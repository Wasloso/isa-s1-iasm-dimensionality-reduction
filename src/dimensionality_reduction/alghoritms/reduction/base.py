from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class DimensionalityReductor(ABC):
    """
    Base class for all dimensionality reduction algorithms.

    Parameters
    ----------
    n_components : int | float | None
        Number of components to keep. Interpretation is subclass-specific.
    """

    def __init__(self, n_components: int | float | None) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit_transform.")

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )


class InductiveDimensionalityReductor(DimensionalityReductor):
    """
    Base class for inductive dimensionality reduction algorithms.
    """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
