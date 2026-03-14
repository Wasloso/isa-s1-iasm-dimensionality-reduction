from typing import Self

import numpy as np

from .base import DimensionalityReductor


class PCA(DimensionalityReductor):
    def __init__(self, n_components: int | float | None) -> None:
        """
        Parameters
        ----------
        n_components:
            - int: Number of components to keep.
            - float: Fraction of variance to keep (0.0 to 1.0).
            - None: Keep all components.
        """
        super().__init__(n_components)
        self.mean_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        total_var = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        if self.n_components is None:
            k = X.shape[1]
        elif isinstance(self.n_components, float):
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            k = np.argmax(cumulative_variance >= self.n_components) + 1
        else:
            k = self.n_components
        self.components_ = sorted_eigenvectors[:, :k]
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
