import numpy as np
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA as SKLEARN_PCA

from dimensionality_reduction.alghoritms.reduction.pca import PCA


def test_pca_orthogonality():
    X = np.random.rand(50, 10)
    pca = PCA(n_components=3).fit(X)
    comps = pca.components_
    identity_approx = np.dot(comps.T, comps)
    expected_identity = np.eye(3)
    assert_allclose(identity_approx, expected_identity, atol=1e-7)


def test_pca_dimensions():
    N, D = 100, 10
    K = 3
    X = np.random.rand(N, D)

    pca = PCA(n_components=K)
    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape == (N, K)
    assert pca.components_.shape == (D, K)


def test_pca_transformed_mean():
    X = np.random.rand(100, 5)
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    mean_transformed = np.mean(X_transformed, axis=0)
    assert_allclose(mean_transformed, np.zeros(2), atol=1e-7)


def test_compare_with_sklearn():
    X = np.random.rand(100, 5)
    n_components = 2

    pca = PCA(n_components=n_components).fit(X)
    X_transformed = pca.transform(X)

    pca_sklearn = SKLEARN_PCA(n_components=n_components).fit(X)
    X_transformed_sklearn = pca_sklearn.transform(X)
    # Abs values are used to account for possible sign differences in eigenvectors
    np.testing.assert_allclose(np.abs(X_transformed), np.abs(X_transformed_sklearn), atol=1e-7)
