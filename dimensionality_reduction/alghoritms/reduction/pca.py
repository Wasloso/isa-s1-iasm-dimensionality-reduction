from typing import Self

import numpy as np

from .base import InductiveDimensionalityReductor


class PCA(InductiveDimensionalityReductor):
    """
    Principal Component Analysis (PCA).

    Parameters
    ----------
    n_components : int | float | None
        - int: exact number of components to keep (1 <= k <= n_features).
        - float in (0.0, 1.0]: minimum cumulative explained variance ratio to
          retain; the smallest k satisfying the threshold is chosen.
        - None: keep all components.
    """

    def __init__(self, n_components: int | float | None = None) -> None:
        super().__init__(n_components)
        self.mean_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
        _n_samples, n_features = X.shape

        self._validate_n_components(n_features)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        self.eigenvalues_ = eigenvalues
        total_var = np.sum(eigenvalues)

        k = self._resolve_n_components(n_features)
        self.n_components_ = k
        self.components_ = eigenvectors[:, :k]
        self.explained_variance_ratio_ = eigenvalues[:k] / total_var
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def _validate_n_components(self, n_features: int) -> None:
        nc = self.n_components
        if nc is None:
            return
        if isinstance(nc, float):
            if not (0.0 < nc <= 1.0):
                raise ValueError(
                    f"n_components={nc!r} must be in the range (0.0, 1.0] when a float is provided."
                )
        elif isinstance(nc, int):
            if not (1 <= nc <= n_features):
                raise ValueError(
                    f"n_components={nc!r} must satisfy 1 <= n_components <= "
                    f"n_features={n_features}."
                )
        else:
            raise TypeError(f"n_components must be int, float, or None; got {type(nc).__name__}.")

    def _resolve_n_components(self, n_features: int) -> int:
        nc = self.n_components
        if nc is None:
            return n_features
        if isinstance(nc, float):
            cumulative = np.cumsum(self.explained_variance_ratio_)
            return int(np.argmax(cumulative >= nc) + 1)
        return int(nc)
