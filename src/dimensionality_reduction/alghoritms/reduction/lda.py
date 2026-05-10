from typing import Self

import numpy as np

from .base import InductiveDimensionalityReductor


class LDA(InductiveDimensionalityReductor):
    """
    Linear Discriminant Analysis (LDA).

    Parameters
    ----------
    n_components : int | None
        Number of discriminant components to keep.
        Must satisfy 1 <= n_components <= min(n_classes - 1, n_features).
        If None, all valid components are kept.

    """

    def __init__(self, n_components: int | None = None) -> None:
        super().__init__(n_components)
        self.classes_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        if y is None:
            raise ValueError("LDA is a supervised method and requires labels y.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples; "
                f"got X: {X.shape[0]}, y: {y.shape[0]}."
            )

        _n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        max_components = min(n_classes - 1, n_features)
        k = self._resolve_n_components(max_components)
        self.n_components_ = k

        overall_mean = np.mean(X, axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for cls in self.classes_:
            X_cls = X[y == cls]
            cls_mean = np.mean(X_cls, axis=0)

            X_centered = X_cls - cls_mean
            S_W += X_centered.T @ X_centered

            n_cls = X_cls.shape[0]
            mean_diff = (cls_mean - overall_mean).reshape(-1, 1)
            S_B += n_cls * (mean_diff @ mean_diff.T)

        S_W_reg = S_W + np.eye(n_features) * 1e-8
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_W_reg) @ S_B)

        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        total = np.sum(eigenvalues[:max_components])
        self.explained_variance_ratio_ = eigenvalues[:k] / total if total > 0 else np.zeros(k)

        self.components_ = eigenvectors[:, :k]
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return np.dot(X, self.components_)

    def _resolve_n_components(self, max_components: int) -> int:
        nc = self.n_components
        if nc is None:
            return max_components
        if not isinstance(nc, int):
            raise TypeError(f"n_components must be int or None for LDA; got {type(nc).__name__}.")
        if not (1 <= nc <= max_components):
            raise ValueError(
                f"n_components={nc!r} must satisfy "
                f"1 <= n_components <= min(n_classes-1, n_features)={max_components}."
            )
        return nc
