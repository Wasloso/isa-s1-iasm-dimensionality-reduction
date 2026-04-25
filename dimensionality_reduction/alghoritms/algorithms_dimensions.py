import numpy as np
from sklearn.preprocessing import StandardScaler

from reduction.pca import PCA
from reduction.lda import LDA
from reduction.tsne import TSNE
from reduction.umap import UMAP


def run_pca(data: np.ndarray, n_components: int) -> np.ndarray:
    n_components = min(n_components, data.shape[0], data.shape[1])
    X = StandardScaler().fit_transform(data)
    return PCA(n_components=n_components).fit_transform(X)


def run_lda(data: np.ndarray, labels: np.ndarray, n_components: int) -> np.ndarray:
    n_classes = len(np.unique(labels))
    n_components = min(n_components, n_classes - 1, data.shape[1])
    if n_components < 1:
        raise ValueError(f"LDA needs at least 2 classes; found {n_classes}.")
    X = StandardScaler().fit_transform(data)
    return LDA(n_components=n_components).fit_transform(X, labels)


def run_tsne(data: np.ndarray, n_components: int) -> np.ndarray:
    n_components = min(n_components, 3)
    perplexity = min(30.0, max(5.0, data.shape[0] / 5.0))
    X = StandardScaler().fit_transform(data)
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=42).fit_transform(X)


# DO Gui zajebiste to jest, ułatwia życie z wykresami
def run_all(
        data: np.ndarray,
        labels: np.ndarray | None,
        n_components: int,
) -> dict[str, np.ndarray | str]:
    results: dict[str, np.ndarray | str] = {}

    for name, fn, needs_labels in [
        ("PCA", lambda d, l: run_pca(d, n_components), False),
        ("LDA", lambda d, l: run_lda(d, l, n_components), True),
        ("t-SNE", lambda d, l: run_tsne(d, n_components), False),
        ("UMAP", lambda d, l: UMAP(d, n_components), False),
    ]:
        if needs_labels and labels is None:
            results[name] = "Requires a label column — select one in the sidebar."
            continue
        try:
            results[name] = fn(data, labels)
        except Exception as exc:
            results[name] = f"Error: {exc}"

    return results



#  Biblioteki wbudowane - do testowania GUI

# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler

# def run_pca(data: np.ndarray, n_components: int) -> np.ndarray:
#
#     n_components = min(n_components, data.shape[0], data.shape[1])
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
#     pca = PCA(n_components=n_components, random_state=42)
#
#     return pca.fit_transform(data_scaled)
#
#
# def run_lda(data: np.ndarray, labels: np.ndarray, n_components: int) -> np.ndarray:
#     """
#     Linear Discriminant Analysis (LDA) — supervised, linear.
#
#     Finds the linear combinations of features that best separate
#     the provided class labels.  Max output dims = (n_classes − 1).
#
#     Parameters
#     ----------
#     data        : 2-D array, shape (n_samples, n_features)
#     labels      : 1-D array, shape (n_samples,)  — class labels
#     n_components: target number of output dimensions
#
#     Returns
#     -------
#     reduced     : 2-D array, shape (n_samples, effective_n_components)
#     """
#     n_classes = len(np.unique(labels))
#     max_components = min(n_classes - 1, data.shape[1], data.shape[0] - 1)
#     n_components = min(n_components, max_components)
#     if n_components < 1:
#         raise ValueError(
#             f"LDA needs at least 2 classes; found {n_classes}."
#         )
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
#     lda = LinearDiscriminantAnalysis(n_components=n_components)
#     return lda.fit_transform(data_scaled, labels)
#
#
# # ─────────────────────────────────────────────
# #  NON-LINEAR METHODS
# # ─────────────────────────────────────────────
#
# def run_tsne(data: np.ndarray, n_components: int) -> np.ndarray:
#     """
#     t-distributed Stochastic Neighbour Embedding (t-SNE) — non-linear.
#
#     Maps high-dimensional data to a low-dimensional space while
#     preserving local neighbourhood structure.  Best for 2-D / 3-D
#     visualisation; not recommended for n_components > 3.
#
#     Parameters
#     ----------
#     data        : 2-D array, shape (n_samples, n_features)
#     n_components: target number of output dimensions (≤ 3 recommended)
#
#     Returns
#     -------
#     reduced     : 2-D array, shape (n_samples, n_components)
#     """
#     n_components = min(n_components, 3)  # t-SNE caps at 3
#     perplexity = min(30, max(5, data.shape[0] // 5))
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
#     tsne = TSNE(
#         n_components=n_components,
#         perplexity=perplexity,
#         max_iter=1000,
#         random_state=42,
#     )
#     return tsne.fit_transform(data_scaled)
#
#
# def run_umap(data: np.ndarray, n_components: int) -> np.ndarray:
#     """
#     Uniform Manifold Approximation and Projection (UMAP) — non-linear.
#
#     Learns the underlying manifold of the data and embeds it in
#     low-dimensional space, preserving both local and global structure
#     better than t-SNE at higher speeds.
#
#     Parameters
#     ----------
#     data        : 2-D array, shape (n_samples, n_features)
#     n_components: target number of output dimensions
#
#     Returns
#     -------
#     reduced     : 2-D array, shape (n_samples, n_components)
#     """
#     try:
#         import umap as umap_lib
#     except ImportError:
#         raise ImportError(
#             "UMAP requires the 'umap-learn' package.  "
#             "Install it with:  pip install umap-learn"
#         )
#     n_neighbors = min(15, max(2, data.shape[0] // 10))
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
#     reducer = umap_lib.UMAP(
#         n_components=n_components,
#         n_neighbors=n_neighbors,
#         min_dist=0.1,
#         random_state=42,
#     )
#     return reducer.fit_transform(data_scaled)
#
#
# # ─────────────────────────────────────────────
# #  CONVENIENCE: run all four at once
# # ─────────────────────────────────────────────
#
# def run_all(
#         data: np.ndarray,
#         labels: np.ndarray | None,
#         n_components: int,
# ) -> dict[str, np.ndarray | str]:
#     """
#     Run PCA, LDA (if labels provided), t-SNE and UMAP.
#
#     Returns a dict mapping method name → reduced array, or an
#     error message string if the method failed.
#     """
#     results: dict[str, np.ndarray | str] = {}
#
#     for name, fn, needs_labels in [
#         ("PCA", lambda d, l: run_pca(d, n_components), False),
#         ("LDA", lambda d, l: run_lda(d, l, n_components), True),
#         ("t-SNE", lambda d, l: run_tsne(d, n_components), False),
#         ("UMAP", lambda d, l: run_umap(d, n_components), False),
#     ]:
#         if needs_labels and labels is None:
#             results[name] = "Requires a label column — select one in the sidebar."
#             continue
#         try:
#             results[name] = fn(data, labels)
#         except Exception as exc:
#             results[name] = f"Error: {exc}"
#
#     return results