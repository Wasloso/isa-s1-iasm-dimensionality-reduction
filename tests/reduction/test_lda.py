import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA

from dimensionality_reduction.alghoritms.reduction.lda import LDA


def _make_separable(n_per_class: int = 60, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [np.zeros(n_features), np.ones(n_features) * 3, -np.ones(n_features) * 3]
    X = np.vstack([rng.standard_normal((n_per_class, n_features)) + m for m in means])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


def test_lda_output_shape():
    X, y = _make_separable()
    lda = LDA(n_components=2).fit(X, y)
    X_t = lda.transform(X)
    assert X_t.shape == (X.shape[0], 2)
    assert lda.components_.shape == (X.shape[1], 2)


def test_lda_fit_transform_consistent():
    X, y = _make_separable()
    lda = LDA(n_components=2)
    X_ft = lda.fit_transform(X, y)
    X_t = lda.transform(X)
    assert_allclose(X_ft, X_t, atol=1e-10)


def test_lda_n_components_capped():
    """n_components is capped at min(n_classes - 1, n_features)."""
    X, y = _make_separable(n_features=10)
    lda = LDA(n_components=None).fit(X, y)
    # 3 classes → max 2 components
    assert lda.n_components_ == 2


def test_lda_requires_labels():
    X, _ = _make_separable()
    with pytest.raises(ValueError, match="supervised"):
        LDA().fit(X)


def test_lda_invalid_n_components():
    X, y = _make_separable()
    with pytest.raises(ValueError):
        LDA(n_components=10).fit(X, y)  # max is 2 for 3 classes


def test_lda_not_fitted_error():
    lda = LDA()
    with pytest.raises(RuntimeError, match="not fitted"):
        lda.transform(np.zeros((5, 3)))


def test_lda_separates_classes():
    """The embedding should separate the three Gaussian clusters."""
    X, y = _make_separable(n_per_class=100, seed=42)
    lda = LDA(n_components=2).fit(X, y)
    X_t = lda.transform(X)

    # Class centroids in 2-D should be far apart
    centroids = np.array([X_t[y == c].mean(axis=0) for c in [0, 1, 2]])
    dists = [np.linalg.norm(centroids[i] - centroids[j]) for i in range(3) for j in range(i + 1, 3)]
    assert all(d > 1.0 for d in dists), f"Clusters not separated: {dists}"


def test_lda_compare_with_sklearn():
    """Between-class separation quality should be close to sklearn's LDA."""
    X, y = _make_separable(seed=7)
    lda = LDA(n_components=2).fit(X, y)
    X_t = lda.transform(X)

    sk = SklearnLDA(n_components=2).fit(X, y)
    X_sk = sk.transform(X)

    def separation_score(Z: np.ndarray, labels: np.ndarray) -> float:
        """Ratio of between-class variance to within-class variance."""
        overall_mean = Z.mean(axis=0)
        classes = np.unique(labels)
        sb = sum(
            np.sum(labels == c) * np.linalg.norm(Z[labels == c].mean(axis=0) - overall_mean) ** 2
            for c in classes
        )
        sw = sum(
            np.sum(np.linalg.norm(Z[labels == c] - Z[labels == c].mean(axis=0), axis=1) ** 2)
            for c in classes
        )
        return float(sb / (sw + 1e-10))

    score_ours = separation_score(X_t, y)
    score_sk = separation_score(X_sk, y)

    # Our implementation should achieve at least 80 % of sklearn's separation
    assert score_ours >= 0.8 * score_sk, (
        f"Separation score {score_ours:.4f} is too far below sklearn's {score_sk:.4f}"
    )
