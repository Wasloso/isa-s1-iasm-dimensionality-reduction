import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

from dimensionality_reduction.alghoritms.metrics import trustworthiness
from dimensionality_reduction.alghoritms.reduction.pca import PCA


def _make_data(n: int = 80, d: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def test_perfect_embedding_scores_one():
    """Embedding identical to the original space should score exactly 1.0."""
    X = _make_data(n=50, d=5)
    # Using the same matrix for both spaces means every rank is perfectly preserved.
    score = trustworthiness(X, X, k=5)
    assert_allclose(score, 1.0, atol=1e-6)


def test_score_in_unit_interval():
    X = _make_data(n=60, d=8)
    rng = np.random.default_rng(1)
    X_low = rng.standard_normal((60, 2))
    score = trustworthiness(X, X_low, k=5)
    assert 0.0 <= score <= 1.0


def test_random_embedding_lower_than_pca():
    """A PCA embedding should be more trustworthy than random noise."""
    X = _make_data(n=100, d=10, seed=7)
    X_pca = PCA(n_components=2).fit_transform(X)
    X_rand = np.random.default_rng(7).standard_normal((100, 2))

    score_pca = trustworthiness(X, X_pca, k=5)
    score_rand = trustworthiness(X, X_rand, k=5)
    assert score_pca > score_rand


def test_matches_sklearn():
    """Our implementation should match sklearn's trustworthiness closely."""
    X = _make_data(n=80, d=6, seed=42)
    X_low = PCA(n_components=2).fit_transform(X)

    ours = trustworthiness(X, X_low, k=5)
    theirs = sklearn_trustworthiness(X, X_low, n_neighbors=5)
    assert_allclose(ours, theirs, atol=1e-4)


def test_invalid_k_raises():
    X = _make_data(n=10, d=3)
    with pytest.raises(ValueError, match="k="):
        trustworthiness(X, X[:, :2], k=10)
