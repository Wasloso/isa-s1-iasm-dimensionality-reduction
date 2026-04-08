import numpy as np
import pytest
from numpy.testing import assert_allclose

from dimensionality_reduction.alghoritms.reduction.tsne import TSNE


def _make_data(n: int = 80, d: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def test_tsne_output_shape():
    X = _make_data()
    Y = TSNE(n_components=2, n_iter=100, random_state=0).fit_transform(X)
    assert Y.shape == (X.shape[0], 2)


def test_tsne_3d_output_shape():
    X = _make_data()
    Y = TSNE(n_components=3, n_iter=100, random_state=0).fit_transform(X)
    assert Y.shape == (X.shape[0], 3)


def test_tsne_fit_stores_embedding():
    X = _make_data()
    tsne = TSNE(n_components=2, n_iter=100, random_state=0).fit(X)
    assert tsne.embedding_ is not None
    assert tsne.embedding_.shape == (X.shape[0], 2)
    assert tsne._is_fitted


def test_tsne_determinism():
    X = _make_data()
    Y1 = TSNE(n_components=2, n_iter=100, random_state=42).fit_transform(X)
    Y2 = TSNE(n_components=2, n_iter=100, random_state=42).fit_transform(X)
    assert_allclose(Y1, Y2)


def test_tsne_different_seeds_differ():
    X = _make_data()
    Y1 = TSNE(n_components=2, n_iter=100, random_state=0).fit_transform(X)
    Y2 = TSNE(n_components=2, n_iter=100, random_state=1).fit_transform(X)
    assert not np.allclose(Y1, Y2)


def test_tsne_kl_divergence_stored():
    X = _make_data()
    tsne = TSNE(n_components=2, n_iter=100, random_state=0)
    tsne.fit_transform(X)
    assert tsne.kl_divergence_ is not None
    assert tsne.kl_divergence_ >= 0.0


def test_tsne_invalid_n_components():
    with pytest.raises(ValueError, match="positive integer"):
        TSNE(n_components=0).fit_transform(np.zeros((10, 5)))


def test_tsne_invalid_input_shape():
    with pytest.raises(ValueError, match="2-D"):
        TSNE().fit_transform(np.zeros(10))


def test_tsne_embedding_centered():
    """The embedding should be roughly zero-centred (centred during optimisation)."""
    X = _make_data(n=100, seed=3)
    Y = TSNE(n_components=2, n_iter=300, random_state=3).fit_transform(X)
    assert_allclose(Y.mean(axis=0), np.zeros(2), atol=0.5)
