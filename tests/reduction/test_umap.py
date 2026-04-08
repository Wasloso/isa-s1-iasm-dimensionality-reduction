import numpy as np
import pytest
from numpy.testing import assert_allclose

from dimensionality_reduction.alghoritms.reduction.umap import UMAP


def _make_data(n: int = 80, d: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def test_umap_output_shape():
    X = _make_data()
    Y = UMAP(n_components=2, n_epochs=5, random_state=0).fit_transform(X)
    assert Y.shape == (X.shape[0], 2)


def test_umap_3d_output_shape():
    X = _make_data()
    Y = UMAP(n_components=3, n_epochs=5, random_state=0).fit_transform(X)
    assert Y.shape == (X.shape[0], 3)


def test_umap_fit_stores_embedding():
    X = _make_data()
    umap = UMAP(n_components=2, n_epochs=5, random_state=0).fit(X)
    assert umap.embedding_ is not None
    assert umap.embedding_.shape == (X.shape[0], 2)
    assert umap._is_fitted


def test_umap_determinism():
    X = _make_data()
    Y1 = UMAP(n_components=2, n_epochs=5, random_state=42).fit_transform(X)
    Y2 = UMAP(n_components=2, n_epochs=5, random_state=42).fit_transform(X)
    assert_allclose(Y1, Y2)


def test_umap_transform_shape():
    X_train = _make_data(n=80, seed=0)
    X_new = _make_data(n=20, seed=1)
    umap = UMAP(n_components=2, n_epochs=5, random_state=0).fit(X_train)
    Y_new = umap.transform(X_new)
    assert Y_new.shape == (20, 2)


def test_umap_transform_requires_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        UMAP().transform(np.zeros((5, 3)))


def test_umap_invalid_input_shape():
    with pytest.raises(ValueError, match="2-D"):
        UMAP(n_epochs=5).fit(np.zeros(10))


def test_umap_ab_params_reasonable():
    """Fitted a, b should be positive and a > 0.5 for typical settings."""
    X = _make_data()
    umap = UMAP(n_components=2, n_epochs=5, random_state=0).fit(X)
    assert umap._a > 0.5
    assert umap._b > 0.0
