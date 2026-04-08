"""
Algorithm runner — maps UI configuration dicts to reducer instances and runs them.

The public entry point is `run_reduction`.  It is decorated with
`@st.cache_data` so that re-running the app with unchanged inputs does not
trigger expensive recomputation.

Because Streamlit's cache serialises arguments, numpy arrays are accepted as
bytes via `X_bytes` / `y_bytes` + shape/dtype metadata to avoid hashing the
raw arrays directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dimensionality_reduction.alghoritms.metrics import trustworthiness
from dimensionality_reduction.alghoritms.reduction.lda import LDA
from dimensionality_reduction.alghoritms.reduction.pca import PCA
from dimensionality_reduction.alghoritms.reduction.tsne import TSNE
from dimensionality_reduction.alghoritms.reduction.umap import UMAP

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_reduction(
    method: str,
    params: dict[str, Any],
    X_bytes: bytes,
    X_shape: tuple[int, int],
    X_dtype: str,
    y_bytes: bytes | None,
    y_shape: tuple[int] | None,
    y_dtype: str | None,
) -> tuple[bytes, tuple[int, int], str, dict[str, Any]]:
    """
    Fit and transform X using the chosen DR algorithm.

    Arguments are passed as raw bytes so Streamlit can hash them cheaply.

    Returns
    -------
    embedding_bytes, embedding_shape, embedding_dtype
        The low-dimensional embedding, serialised as bytes.
    metrics : dict
        Algorithm-specific quality metrics extracted from the fitted model.
    """
    X = np.frombuffer(X_bytes, dtype=X_dtype).reshape(X_shape)
    y = None
    if y_bytes is not None and y_shape is not None and y_dtype is not None:
        y = np.frombuffer(y_bytes, dtype=y_dtype).reshape(y_shape)

    embedding, metrics = _fit_transform(method, params, X, y)

    return (
        embedding.tobytes(),
        embedding.shape,
        str(embedding.dtype),
        metrics,
    )


def decode_embedding(
    embedding_bytes: bytes,
    embedding_shape: tuple[int, int],
    embedding_dtype: str,
) -> np.ndarray:
    return np.frombuffer(embedding_bytes, dtype=embedding_dtype).reshape(embedding_shape)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _fit_transform(
    method: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "PCA":
        embedding, metrics = _run_pca(params, X)
    elif method == "LDA":
        embedding, metrics = _run_lda(params, X, y)
    elif method == "t-SNE":
        embedding, metrics = _run_tsne(params, X)
    elif method == "UMAP":
        embedding, metrics = _run_umap(params, X)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    k = min(5, X.shape[0] - 1)
    metrics["trustworthiness"] = trustworthiness(X, embedding, k=k)
    metrics["trustworthiness_k"] = k
    return embedding, metrics


def _run_pca(params: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, dict]:
    model = PCA(n_components=params["n_components"])
    embedding = model.fit_transform(X)
    metrics = {
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "n_components_": model.n_components_,
    }
    return embedding, metrics


def _run_lda(
    params: dict[str, Any], X: np.ndarray, y: np.ndarray | None
) -> tuple[np.ndarray, dict]:
    if y is None:
        raise ValueError("LDA requires a label column to be selected.")
    model = LDA(n_components=params["n_components"])
    embedding = model.fit_transform(X, y)
    metrics = {
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "n_components_": model.n_components_,
        "classes": model.classes_.tolist(),
    }
    return embedding, metrics


def _run_tsne(params: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, dict]:
    model = TSNE(
        n_components=params["n_components"],
        perplexity=params["perplexity"],
        learning_rate=params["learning_rate"],
        n_iter=params["n_iter"],
        early_exaggeration=params["early_exaggeration"],
        random_state=params["random_state"],
    )
    embedding = model.fit_transform(X)
    metrics = {"kl_divergence": model.kl_divergence_}
    return embedding, metrics


def _run_umap(params: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, dict]:
    model = UMAP(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        spread=params["spread"],
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        random_state=params["random_state"],
    )
    model.fit(X)
    embedding = model.embedding_
    metrics = {"a": model._a, "b": model._b}
    return embedding, metrics
