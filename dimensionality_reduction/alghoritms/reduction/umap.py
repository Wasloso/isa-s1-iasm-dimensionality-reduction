from typing import Self
import numpy as np
from .base import DimensionalityReductor


class UMAP(DimensionalityReductor):
    """
    Uniform Manifold Approximation and Projection (UMAP).

    Parameters
    ----------
    n_components  : int     Output dimensionality.
    n_neighbors   : int     Size of the local neighbourhood for graph construction.
    min_dist      : float   Minimum distance between points in the embedding.
    n_epochs      : int     Optimisation iterations.
    learning_rate : float   Initial step size.
    random_state  : int | None
    """

    def __init__(
            self,
            n_components: int = 2,
            n_neighbors: int = 15,
            min_dist: float = 0.1,
            n_epochs: int = 200,
            learning_rate: float = 1.0,
            random_state: int | None = None,
    ) -> None:
        super().__init__(n_components)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.embedding_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        nc = int(self.n_components)

        knn_idx, knn_dist = self._knn(X, k)
        rows, cols, weights = self._fuzzy_simplicial_set(knn_idx, knn_dist, n, k)
        a, b = self._fit_ab(self.min_dist)
        embedding = self._spectral_init(rows, cols, weights, n, nc, rng)
        embedding = self._optimize(embedding, rows, cols, weights, a, b, rng)

        self.embedding_ = embedding
        self._is_fitted = True
        return embedding

    # ── graph construction ────────────────────────────────────

    def _knn(self, X: np.ndarray, k: int):
        sq = np.sum(X ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0))
        idx = np.argsort(D, axis=1)[:, 1: k + 1]
        dist = D[np.arange(X.shape[0])[:, None], idx]
        return idx, dist

    def _fuzzy_simplicial_set(self, knn_idx, knn_dist, n, k):
        rho = knn_dist[:, 0]
        sigma = np.array([self._find_sigma(knn_dist[i], rho[i], np.log2(k)) for i in range(n)])

        W = np.zeros((n, n))
        for i in range(n):
            for j_pos in range(k):
                j = knn_idx[i, j_pos]
                d = knn_dist[i, j_pos]
                W[i, j] = np.exp(-max(d - rho[i], 0.0) / (sigma[i] + 1e-10))

        # fuzzy union: w(i,j) = A + B − A*B  where A=W[i,j], B=W[j,i]
        W_sym = W + W.T - W * W.T

        r, c = np.nonzero(W_sym)
        w = W_sym[r, c]
        return r, c, w

    def _find_sigma(self, dists: np.ndarray, rho: float, target: float) -> float:
        lo, hi, sigma = 0.0, np.inf, 1.0
        for _ in range(64):
            val = np.sum(np.exp(-np.maximum(dists - rho, 0.0) / sigma))
            diff = val - target
            if abs(diff) < 1e-5:
                break
            if diff > 0:
                hi = sigma
                sigma = (lo + sigma) / 2.0
            else:
                lo = sigma
                sigma = sigma * 2.0 if hi == np.inf else (sigma + hi) / 2.0
        return sigma

    # ── a, b parameters ──────────────────────────────────────
    # Low-dim similarity: φ(d) = 1 / (1 + a·d^(2b))
    # Fit to: 1 if d < min_dist, else exp(−(d − min_dist))

    def _fit_ab(self, min_dist: float) -> tuple[float, float]:
        x = np.linspace(0.0, 3.0 * (min_dist + 1.0), 300)
        y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist)))

        a, b = 1.0, 1.0
        lr = 1e-3
        for _ in range(2000):
            phi = 1.0 / (1.0 + a * x ** (2 * b))
            err = phi - y
            da = np.mean(err * (-(x ** (2 * b)) * phi ** 2))
            db = np.mean(err * (-2 * a * np.log(x + 1e-10) * x ** (2 * b) * phi ** 2))
            a -= lr * da
            b -= lr * db
            a = max(a, 1e-4)
            b = max(b, 1e-4)
        return float(a), float(b)

    # ── initialisation ────────────────────────────────────────

    def _spectral_init(self, rows, cols, weights, n, nc, rng) -> np.ndarray:
        try:
            W = np.zeros((n, n))
            W[rows, cols] = weights
            deg = W.sum(axis=1) + 1e-10
            D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
            L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
            _, vecs = np.linalg.eigh(L)
            emb = vecs[:, 1: nc + 1]
            emb /= np.std(emb) + 1e-10
            emb *= 1e-4
        except Exception:
            emb = rng.randn(n, nc) * 1e-4
        return emb

    # ── SGD optimisation ─────────────────────────────────────

    def _optimize(self, embedding, rows, cols, weights, a, b, rng) -> np.ndarray:
        n = embedding.shape[0]
        n_neg = 5
        max_w = weights.max() + 1e-10
        epochs_per_sample = max_w / weights
        next_epoch = epochs_per_sample.copy()

        for epoch in range(self.n_epochs):
            alpha = max(self.learning_rate * (1.0 - epoch / self.n_epochs), 1e-4)

            for idx in range(len(rows)):
                if next_epoch[idx] > epoch:
                    continue

                i, j = rows[idx], cols[idx]
                d_sq = np.dot(embedding[i] - embedding[j], embedding[i] - embedding[j]) + 1e-10

                # attractive gradient
                coeff = -2.0 * a * b * d_sq ** (b - 1.0) / (a * d_sq ** b + 1.0)
                delta = alpha * coeff * (embedding[i] - embedding[j])
                embedding[i] += delta
                embedding[j] -= delta

                # repulsive gradients (negative samples)
                for _ in range(n_neg):
                    k_neg = rng.randint(0, n)
                    if k_neg == i:
                        continue
                    d_neg_sq = np.dot(embedding[i] - embedding[k_neg], embedding[i] - embedding[k_neg]) + 1e-10
                    coeff_neg = 2.0 * b / ((0.001 + d_neg_sq) * (a * d_neg_sq ** b + 1.0))
                    embedding[i] += alpha * coeff_neg * (embedding[i] - embedding[k_neg])

                next_epoch[idx] += epochs_per_sample[idx]

        return embedding