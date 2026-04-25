from typing import Self
import numpy as np
from .base import DimensionalityReductor


class TSNE(DimensionalityReductor):
    """
    t-distributed Stochastic Neighbour Embedding (t-SNE).

    Parameters
    ----------
    n_components : int
        Output dimensionality. Practically limited to 1–3.
    perplexity   : float
        Effective number of neighbours (5–50 is typical).
    learning_rate: float
        Step size for gradient descent.
    n_iter       : int
        Number of optimisation iterations.
    random_state : int | None
    """

    def __init__(
            self,
            n_components: int = 2,
            perplexity: float = 30.0,
            learning_rate: float = 200.0,
            n_iter: int = 1000,
            random_state: int | None = None,
    ) -> None:
        super().__init__(n_components)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        k = int(self.n_components)

        P = self._joint_probabilities(X)
        P *= 4.0  # early exaggeration

        Y = rng.randn(n, k) * 1e-4
        Y_prev = Y.copy()
        gains = np.ones((n, k))
        momentum = 0.5

        for t in range(self.n_iter):
            if t == 100:
                P /= 4.0  # end early exaggeration
            if t == 250:
                momentum = 0.8

            D = self._sq_distances(Y)
            Q_num = 1.0 / (1.0 + D)
            np.fill_diagonal(Q_num, 0.0)
            Q = Q_num / (np.sum(Q_num) + 1e-12)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            # vectorised gradient: dC/dY_i = 4 Σ_j (p_ij − q_ij)(y_i − y_j)(1+||y_i−y_j||²)⁻¹
            A = PQ * Q_num
            grad = 4.0 * (A.sum(axis=1, keepdims=True) * Y - A @ Y)

            direction_changed = (grad > 0) != ((Y - Y_prev) > 0)
            gains = np.where(direction_changed, gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)

            step = self.learning_rate * gains * grad
            Y_new = Y + momentum * (Y - Y_prev) - step
            Y_prev, Y = Y, Y_new

        self.embedding_ = Y
        self._is_fitted = True
        return Y

    # ── helpers ──────────────────────────────────────────────

    def _sq_distances(self, X: np.ndarray) -> np.ndarray:
        sq = np.sum(X ** 2, axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.fill_diagonal(D, 0.0)
        return np.maximum(D, 0.0)

    def _joint_probabilities(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        D = self._sq_distances(X)
        P = np.zeros((n, n))
        log2_perp = np.log2(self.perplexity)
        for i in range(n):
            P[i] = self._binary_search_sigma(D[i], i, log2_perp)
        P = (P + P.T) / (2.0 * n)
        return np.maximum(P, 1e-12)

    def _binary_search_sigma(
            self, distances: np.ndarray, i: int, target_entropy: float
    ) -> np.ndarray:
        beta_lo, beta_hi = -np.inf, np.inf
        beta = 1.0

        for _ in range(50):
            p = np.exp(-distances * beta)
            p[i] = 0.0
            s = p.sum()
            if s < 1e-12:
                p_norm = np.full(len(p), 1.0 / (len(p) - 1))
                p_norm[i] = 0.0
                return p_norm
            p_norm = p / s
            nz = p_norm > 0
            entropy = -np.sum(p_norm[nz] * np.log2(p_norm[nz]))

            diff = entropy - target_entropy
            if abs(diff) < 1e-5:
                break
            if diff > 0:  # entropy too high → increase beta
                beta_lo = beta
                beta = beta * 2 if beta_hi == np.inf else (beta + beta_hi) / 2
            else:  # entropy too low  → decrease beta
                beta_hi = beta
                beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2

        return p_norm
