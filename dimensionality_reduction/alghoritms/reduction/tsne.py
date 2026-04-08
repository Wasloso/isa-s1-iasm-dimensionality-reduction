from typing import Self

import numpy as np

from .base import DimensionalityReductor


class TSNE(DimensionalityReductor):
    """
    t-Distributed Stochastic Neighbour Embedding (t-SNE).

    Parameters
    ----------
    n_components : int
        Dimensionality of the embedding space (default 2).
    perplexity : float
        Balances attention between local and global structure. Typical values
        are 5-50; default 30.0.
    learning_rate : float
        Gradient-descent step size. Default 200.0.
    n_iter : int
        Number of optimisation iterations. Default 1000.
    early_exaggeration : float
        Factor applied to P in the early phase of optimisation (first
        `n_iter_early_exag` iterations). Default 12.0.
    n_iter_early_exag : int
        Number of iterations with early exaggeration. Default 250.
    momentum : float
        Momentum coefficient for gradient updates. Default 0.8.
    final_momentum : float
        Momentum used after `n_iter_early_exag`. Default 0.8.
    random_state : int | None
        Seed for the random number generator. Default None.
    verbose : bool
        Print KL-divergence every 100 iterations. Default False.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        n_iter_early_exag: int = 250,
        momentum: float = 0.8,
        final_momentum: float = 0.8,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(n_components)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.n_iter_early_exag = n_iter_early_exag
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.random_state = random_state
        self.verbose = verbose

        self.embedding_: np.ndarray | None = None
        self.kl_divergence_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer for TSNE.")

        rng = np.random.default_rng(self.random_state)

        P = self._compute_joint_probabilities(X)

        n_samples = X.shape[0]
        Y = rng.standard_normal((n_samples, self.n_components)) * 1e-4
        velocity = np.zeros_like(Y)

        for iteration in range(self.n_iter):
            exag = self.early_exaggeration if iteration < self.n_iter_early_exag else 1.0
            mom = self.momentum if iteration < self.n_iter_early_exag else self.final_momentum

            Q, inv_distances = self._compute_low_dim_affinities(Y)
            grad = self._compute_gradient(P, Q, Y, inv_distances, exaggeration=exag)

            velocity = mom * velocity - self.learning_rate * grad
            Y = Y + velocity

            Y -= Y.mean(axis=0)

            if self.verbose and (iteration + 1) % 100 == 0:
                kl = self._kl_divergence(P, Q)
                print(f"[t-SNE] iter {iteration + 1:4d} | KL={kl:.4f}")

        self.kl_divergence_ = float(self._kl_divergence(P, self._compute_low_dim_affinities(Y)[0]))
        self.embedding_ = Y
        self._is_fitted = True
        return Y

    def _pairwise_squared_distances(self, X: np.ndarray) -> np.ndarray:
        sum_sq = np.sum(X**2, axis=1)
        D = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2.0 * (X @ X.T)
        np.clip(D, 0, None, out=D)
        return D

    def _perplexity_binary_search(
        self, distances_row: np.ndarray, target_perplexity: float
    ) -> np.ndarray:
        beta = 1.0
        beta_min, beta_max = -np.inf, np.inf
        log_target = np.log(target_perplexity)

        for _ in range(50):
            exp_d = np.exp(-distances_row * beta)
            sum_exp = np.sum(exp_d)
            if sum_exp == 0:
                sum_exp = 1e-10
            p_row = exp_d / sum_exp
            H = -np.sum(p_row * np.log(p_row + 1e-10))

            if log_target < H:
                beta_min = beta
                beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2.0

        exp_d = np.exp(-distances_row * beta)
        sum_exp = np.sum(exp_d)
        return exp_d / (sum_exp if sum_exp > 0 else 1e-10)

    def _compute_joint_probabilities(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        D = self._pairwise_squared_distances(X)

        P = np.zeros((n, n))
        for i in range(n):
            d_i = D[i].copy()
            d_i[i] = np.inf
            P[i] = self._perplexity_binary_search(d_i, self.perplexity)
            P[i, i] = 0.0

        P = (P + P.T) / (2.0 * n)
        np.clip(P, 1e-12, None, out=P)
        return P

    def _compute_low_dim_affinities(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = self._pairwise_squared_distances(Y)
        inv_distances = 1.0 / (1.0 + D)
        np.fill_diagonal(inv_distances, 0.0)
        Q = inv_distances / np.sum(inv_distances)
        np.clip(Q, 1e-12, None, out=Q)
        return Q, inv_distances

    def _compute_gradient(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Y: np.ndarray,
        inv_distances: np.ndarray,
        exaggeration: float,
    ) -> np.ndarray:
        PQ_diff = (exaggeration * P - Q) * inv_distances
        diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        grad = 4.0 * (PQ_diff[:, :, np.newaxis] * diff).sum(axis=1)
        return grad

    def _kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        mask = P > 0
        return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))
