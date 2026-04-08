from typing import Self

import numpy as np

from .base import InductiveDimensionalityReductor


class UMAP(InductiveDimensionalityReductor):
    """
    Uniform Manifold Approximation and Projection (UMAP)

    Parameters
    ----------
    n_components : int
        Dimensionality of the embedding space. Default 2.
    n_neighbors : int
        Number of nearest neighbours used to build the local graph.  Larger
        values capture more global structure. Default 15.
    min_dist : float
        Minimum distance between points in the embedding. Controls how tightly
        clusters pack together. Default 0.1.
    spread : float
        Scale of the embedding. Increase to spread clusters further apart.
        Default 1.0.
    n_epochs : int
        Number of optimisation epochs. Default 200.
    learning_rate : float
        Initial SGD step size. Decays linearly to zero. Default 1.0.
    negative_sample_rate : int
        Number of random negative samples per positive edge per epoch.
        Default 5.
    random_state : int | None
        Seed for reproducibility. Default None.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state

        self.embedding_: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._a: float = 1.0
        self._b: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")

        rng = np.random.default_rng(self.random_state)
        self._X_train = X.copy()
        n = X.shape[0]

        k = min(self.n_neighbors, n - 1)
        knn_idx, knn_dist = self._knn(X, k)
        rows, cols, weights = self._fuzzy_simplicial_set(knn_idx, knn_dist)

        self._a, self._b = self._find_ab_params(self.spread, self.min_dist)

        Y = rng.standard_normal((n, self.n_components)) * 1e-4
        Y = self._optimize_layout(Y, rows, cols, weights, self._a, self._b, rng)

        self.embedding_ = Y
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        k = min(self.n_neighbors, self._X_train.shape[0])
        knn_idx, knn_dist = self._knn_cross(X, self._X_train, k)

        w = 1.0 / (knn_dist + 1e-10)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("ij,ijk->ik", w, self.embedding_[knn_idx])

    def _knn(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sq = np.sum(X**2, axis=1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.clip(D2, 0.0, None, out=D2)
        np.fill_diagonal(D2, np.inf)
        idx = np.argsort(D2, axis=1)[:, :k]
        dists = np.sqrt(np.take_along_axis(D2, idx, axis=1))
        return idx, dists

    def _knn_cross(
        self, X_new: np.ndarray, X_ref: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sq_new = np.sum(X_new**2, axis=1)
        sq_ref = np.sum(X_ref**2, axis=1)
        D2 = sq_new[:, None] + sq_ref[None, :] - 2.0 * (X_new @ X_ref.T)
        np.clip(D2, 0.0, None, out=D2)
        k = min(k, X_ref.shape[0])
        idx = np.argsort(D2, axis=1)[:, :k]
        dists = np.sqrt(np.take_along_axis(D2, idx, axis=1))
        return idx, dists

    def _fuzzy_simplicial_set(
        self, knn_idx: np.ndarray, knn_dist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = knn_idx.shape
        rho = knn_dist[:, 0]

        sigmas = np.array([self._binary_search_sigma(knn_dist[i], rho[i], k) for i in range(n)])

        row_i = np.repeat(np.arange(n), k)
        col_j = knn_idx.flatten()
        d_ij = knn_dist.flatten()
        rho_i = np.repeat(rho, k)
        sig_i = np.repeat(sigmas, k)

        vals = np.exp(-np.maximum(0.0, d_ij - rho_i) / sig_i)

        directed: dict[tuple[int, int], float] = {
            (int(row_i[e]), int(col_j[e])): float(vals[e]) for e in range(len(row_i))
        }

        sym_rows, sym_cols, sym_vals = [], [], []
        processed: set[tuple[int, int]] = set()

        for (i, j), v_ij in directed.items():
            if i == j or (i, j) in processed:
                continue
            v_ji = directed.get((j, i), 0.0)
            w = v_ij + v_ji - v_ij * v_ji
            sym_rows += [i, j]
            sym_cols += [j, i]
            sym_vals += [w, w]
            processed.add((i, j))
            processed.add((j, i))

        return (
            np.array(sym_rows, dtype=np.int64),
            np.array(sym_cols, dtype=np.int64),
            np.array(sym_vals, dtype=np.float64),
        )

    def _binary_search_sigma(self, dists: np.ndarray, rho: float, k: int) -> float:

        target = np.log2(k)
        lo, hi = 0.0, np.inf
        sigma = 1.0

        for _ in range(64):
            membership = np.sum(np.exp(-np.maximum(0.0, dists - rho) / sigma))
            if abs(membership - target) < 1e-5:
                break
            if membership > target:
                lo = sigma
                sigma = sigma * 2.0 if hi == np.inf else (sigma + hi) / 2.0
            else:
                hi = sigma
                sigma = (lo + sigma) / 2.0

        return max(sigma, 1e-10)

    def _find_ab_params(self, spread: float, min_dist: float) -> tuple[float, float]:

        xv = np.linspace(1e-6, spread * 3.0, 300)
        yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
        log_xv = np.log(xv)

        a, b = 1.0, 1.0
        eps = 1e-8
        lr = 0.01

        m_a = m_b = v_a = v_b = 0.0
        beta1, beta2 = 0.9, 0.999

        for t in range(1, 3001):
            x2b = xv ** (2.0 * b)
            denom = 1.0 + a * x2b
            f = 1.0 / denom
            res = f - yv

            da = -2.0 * np.mean(res * x2b / (denom**2))
            db = -4.0 * a * np.mean(res * log_xv * x2b / (denom**2))

            m_a = beta1 * m_a + (1 - beta1) * da
            v_a = beta2 * v_a + (1 - beta2) * da**2
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db**2

            a -= lr * (m_a / (1 - beta1**t)) / (np.sqrt(v_a / (1 - beta2**t)) + eps)
            b -= lr * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)

            a = max(a, eps)
            b = max(b, eps)

        return float(a), float(b)

    def _optimize_layout(
        self,
        Y: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        weights: np.ndarray,
        a: float,
        b: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n, nc = Y.shape
        n_edges = len(rows)

        for epoch in range(self.n_epochs):
            alpha = max(
                self.learning_rate * (1.0 - epoch / self.n_epochs),
                1e-4 * self.learning_rate,
            )

            diff = Y[rows] - Y[cols]
            d2 = np.maximum(np.sum(diff**2, axis=1), 1e-12)
            d2b = d2**b
            attr_coeff = alpha * weights * (-2.0 * a * b * d2 ** (b - 1) / (1.0 + a * d2b))
            attr_grad = np.clip(attr_coeff[:, None] * diff, -4.0, 4.0)

            dY = np.zeros((n, nc))
            for d in range(nc):
                dY[:, d] += np.bincount(rows, weights=attr_grad[:, d], minlength=n)
                dY[:, d] -= np.bincount(cols, weights=attr_grad[:, d], minlength=n)

            n_neg = n_edges * self.negative_sample_rate
            ni = rng.integers(0, n, size=n_neg)
            nj = rng.integers(0, n, size=n_neg)

            diff_n = Y[ni] - Y[nj]
            d2n = np.maximum(np.sum(diff_n**2, axis=1), 1e-12)
            d2bn = d2n**b
            rep_coeff = alpha * 2.0 * b / ((0.001 + d2n) * (1.0 + a * d2bn))
            rep_grad = np.clip(rep_coeff[:, None] * diff_n, -4.0, 4.0)

            for d in range(nc):
                dY[:, d] += np.bincount(ni, weights=rep_grad[:, d], minlength=n)

            Y += dY

        return Y
