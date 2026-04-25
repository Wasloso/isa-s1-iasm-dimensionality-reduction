import numpy as np
from sklearn.preprocessing import StandardScaler
import umap as umap_lib


def UMAP(data: np.ndarray, n_components: int) -> np.ndarray:
    n_neighbors = min(15, max(2, data.shape[0] // 10))
    X = StandardScaler().fit_transform(data)
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    return reducer.fit_transform(X)
