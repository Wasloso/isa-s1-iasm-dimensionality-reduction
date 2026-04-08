"""
Embedding quality metrics.

All implementations are from scratch using only numpy.
"""

from __future__ import annotations

import numpy as np


def trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 5,
) -> float:
    """
    Trustworthiness of a low-dimensional embedding.

    Measures how well the k-nearest-neighbour structure of the original
    high-dimensional space is preserved in the embedding.  A point's
    neighbourhood is considered *trustworthy* if its k nearest neighbours
    in the embedding were also among its close neighbours in the original
    space.

    Score 1.0 means perfect preservation; 0.0 means worst possible.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding to evaluate.
    k : int
        Neighbourhood size. Typical values: 5-15.

    Returns
    -------
    float in [0, 1]

    Notes
    -----
    Complexity: O(n²) time and space due to the pairwise distance matrices.
    For large datasets (n > 5 000) consider sub-sampling before calling this.

    Reference
    ---------
    Venna & Kaski (2006). "Local multidimensional scaling."
    Neural Networks 19(6-7), 889-899.
    """
    n = X_high.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than n_samples={n}.")

    D_high = _pairwise_sq_dist(X_high)
    D_low = _pairwise_sq_dist(X_low)

    # For each point i, rank of every other point j in the *original* space.
    # Because D_high has inf on the diagonal, self is always sorted last.
    # ranks_high[i, j] == 0  →  j is the nearest neighbour of i in X_high.
    # ranks_high[i, j] == k-1  →  j is the k-th nearest neighbour.
    # ranks_high[i, i] == n-1  →  self is last (infinite distance).
    order_high = np.argsort(D_high, axis=1)
    ranks_high = np.empty_like(order_high)
    for i in range(n):
        ranks_high[i, order_high[i]] = np.arange(n)

    # k-nearest neighbours in the LOW-dimensional embedding.
    # D_low also has inf on the diagonal, so [:, :k] correctly skips self.
    nn_low = np.argsort(D_low, axis=1)[:, :k]  # shape (n, k)

    penalty = 0.0
    for i in range(n):
        for j in nn_low[i]:
            r = int(ranks_high[i, j])
            # ranks 0..k-1 are the k true nearest neighbours in original space.
            # rank >= k means j is NOT in the k-NN → contributes to penalty.
            # penalty = r - (k-1) so that the just-outside rank k gives penalty 1,
            # matching the 1-indexed formula in the literature (r_1 - k where r_1 = r+1).
            if r >= k:
                penalty += r - (k - 1)

    normaliser = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - normaliser * penalty)


def _pairwise_sq_dist(Z: np.ndarray) -> np.ndarray:
    """Efficient pairwise squared Euclidean distances via the Gram-matrix identity."""
    sq = np.sum(Z**2, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.clip(D, 0.0, None, out=D)
    np.fill_diagonal(D, np.inf)
    return D
