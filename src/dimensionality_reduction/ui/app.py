"""
Streamlit app — Dimensionality Reduction Explorer.

Run with:
    uv run streamlit run src/dimensionality_reduction/ui/app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets as sklearn_datasets

from dimensionality_reduction.ui.algorithms import (
    AVAILABLE_METHODS,
    decode_embedding,
    run_reduction,
)
from dimensionality_reduction.ui.param_specs import DataContext, render_params
from dimensionality_reduction.ui.plots import make_scatter, make_variance_chart


def _encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Return (y_encoded_for_models, y_original_for_colouring).

    - For string/object labels we map to integer classes (required by LDA).
    - For numeric labels we pass them through (still usable for colouring).
    """

    if y_raw.dtype.kind in ("U", "O"):
        _classes, y = np.unique(y_raw, return_inverse=True)
        return y.astype(np.int64), y_raw
    return y_raw.astype(np.float64), y_raw


def _prepare_model_frame(
    df_raw: pd.DataFrame, feature_cols: list[str], missing_strategy: str
) -> pd.DataFrame:
    df_model = df_raw
    if missing_strategy == "Drop rows with missing values":
        df_model = df_model.dropna(subset=feature_cols)
    else:
        means = df_model[feature_cols].mean(numeric_only=True)
        df_model = df_model.copy()
        df_model[feature_cols] = df_model[feature_cols].fillna(means)
    return df_model


# Page config


st.set_page_config(
    page_title="DR Explorer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dimensionality Reduction Explorer")
st.caption("Load a dataset, select features, pick an algorithm and explore the embedding.")


# Sidebar — Data


BUILTIN_DATASETS = {
    "Iris": "load_iris",
    "Wine": "load_wine",
    "Breast Cancer": "load_breast_cancer",
    "Digits": "load_digits",
}

with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Built-in dataset", "Upload CSV"], horizontal=True)

    df_raw: pd.DataFrame | None = None
    default_label_col: str | None = None
    missing_strategy: str = "Drop rows with missing values"

    if source == "Built-in dataset":
        dataset_name = st.selectbox("Dataset", list(BUILTIN_DATASETS))
        loader = getattr(sklearn_datasets, BUILTIN_DATASETS[dataset_name])
        bunch = loader()
        df_raw = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        target_col = "target"
        if hasattr(bunch, "target_names"):
            df_raw[target_col] = [bunch.target_names[i] for i in bunch.target]
        else:
            df_raw[target_col] = bunch.target.astype(str)
        default_label_col = target_col
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
        else:
            st.info("Upload a CSV file to continue.")

    # Sidebar — Features

    if df_raw is not None:
        st.header("2 · Features")
        numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
        all_cols = df_raw.columns.tolist()

        feature_cols = st.multiselect(
            "Feature columns",
            options=numeric_cols,
            default=numeric_cols,
            help="Columns used as input to the DR algorithm.",
        )

        st.caption(
            "Missing values (NaN) in selected features can break algorithms like PCA/t-SNE/UMAP."
        )
        missing_strategy = st.radio(
            "Missing value handling",
            options=["Drop rows with missing values", "Fill with mean (per column)"],
            index=0,
            help=(
                "Drop removes any row where at least one selected feature is missing. "
                "Fill replaces missing feature values with that feature's mean."
            ),
        )

        label_options = ["(none)", *all_cols]
        label_default_idx = (
            label_options.index(default_label_col) if default_label_col in label_options else 0
        )
        label_col = st.selectbox(
            "Label / colour column",
            options=label_options,
            index=label_default_idx,
            help="Used to colour points in the scatter plot. Required for LDA.",
        )
        if label_col == "(none)":
            label_col = None

        # Sidebar — Algorithm

        st.header("3 · Algorithm")
        method = st.radio("Method", list(AVAILABLE_METHODS))
        n_output_dims = st.radio("Output dimensions", [2, 3], horizontal=True)

        # Sidebar — Hyperparameters

        st.header("4 · Hyperparameters")
        params: dict = {"n_components": n_output_dims}

        ctx = DataContext(
            n_features=len(feature_cols) if feature_cols else 1,
            n_classes=df_raw[label_col].nunique() if label_col else 2,
            n_output_dims=n_output_dims,
        )

        if method == "LDA" and label_col is None:
            st.warning("LDA requires a label column — please select one above.")
        else:
            params.update(render_params(method, ctx))

        # Sidebar — Run button

        st.divider()
        run_btn = st.button("▶ Run", type="primary", width="stretch")

# Main area

if df_raw is None:
    st.stop()

tab_data, tab_results = st.tabs(["Data", "Results"])

# Tab 1 — Data preview

with tab_data:
    st.subheader("Dataset preview")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Rows", df_raw.shape[0])
    col_info2.metric("Columns", df_raw.shape[1])
    col_info3.metric(
        "Classes",
        df_raw[label_col].nunique() if label_col else "—",
    )

    st.dataframe(df_raw.head(10), width="stretch")

    if feature_cols:
        st.subheader("Missing data inspection")
        miss = df_raw[feature_cols].isna()
        rows_with_missing = int(miss.any(axis=1).sum())
        total_rows = int(df_raw.shape[0])
        pct = (rows_with_missing / total_rows * 100.0) if total_rows else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows with any missing selected feature", rows_with_missing)
        m2.metric("Percent of rows affected", f"{pct:.2f}%")
        m3.metric(
            "Recommended handling",
            "Drop" if rows_with_missing > 0 else "None needed",
            help=(
                "If many rows are missing, imputation may preserve sample size but can distort distances. "
                "Dropping is safest for algorithm stability."
            ),
        )

        with st.expander("Per-feature missing counts", expanded=False):
            per_col_missing = miss.sum(axis=0).sort_values(ascending=False)
            st.dataframe(
                per_col_missing.to_frame(name="missing_values"),
                width="stretch",
            )

        if rows_with_missing > 0:
            st.info(
                "To run the algorithms successfully, handle missing values. "
                "Use the sidebar’s **Missing value handling** option: "
                "**Drop rows with missing values** (safest) or **Fill with mean**."
            )

        st.subheader("Feature statistics")
        st.dataframe(df_raw[feature_cols].describe().T, width="stretch")

# Tab 2 — Results

with tab_results:
    if not feature_cols:
        st.warning("Select at least one feature column in the sidebar.")
        st.stop()

    if method == "LDA" and label_col is None:
        st.warning("LDA requires a label column. Please select one in the sidebar.")
        st.stop()

    if not run_btn and "last_embedding" not in st.session_state:
        st.info("Configure the algorithm in the sidebar and press **▶ Run**.")
        st.stop()

    if run_btn:
        df_model = _prepare_model_frame(df_raw, feature_cols, missing_strategy)

        remaining_missing_cols = (
            df_model[feature_cols].isna().sum(axis=0).loc[lambda s: s > 0].index.tolist()
        )
        if remaining_missing_cols:
            st.error(
                "Some selected features still contain missing values after applying the chosen strategy: "
                f"{', '.join(map(str, remaining_missing_cols))}. "
                "Try dropping rows with missing values, or remove those features from the selection."
            )
            st.stop()

        if df_model.shape[0] == 0:
            st.error(
                "After handling missing values, there are no rows left to run the algorithm on. "
                "Try selecting different features or switching the missing value strategy."
            )
            st.stop()

        X = df_model[feature_cols].to_numpy(dtype=np.float64)
        y = None
        if label_col is not None:
            y, label_values = _encode_labels(df_model[label_col].to_numpy())
        else:
            label_values = None

        with st.spinner(f"Running {method}…"):
            try:
                result = run_reduction(
                    method=method,
                    params=params,
                    X_bytes=X.tobytes(),
                    X_shape=X.shape,
                    X_dtype=str(X.dtype),
                    y_bytes=y.tobytes() if y is not None else None,
                    y_shape=y.shape if y is not None else None,
                    y_dtype=str(y.dtype) if y is not None else None,
                )
                st.session_state["last_embedding"] = result
                st.session_state["last_method"] = method
                st.session_state["last_params"] = params
                st.session_state["last_label_col"] = label_col
                st.session_state["last_label_values"] = label_values
                st.session_state["last_n_output_dims"] = n_output_dims
            except Exception as exc:
                st.error(f"Error running {method}: {exc}")
                st.stop()

    # Retrieve from session state (supports re-render without re-run)
    emb_bytes, emb_shape, emb_dtype, metrics = st.session_state["last_embedding"]
    embedding = decode_embedding(emb_bytes, emb_shape, emb_dtype)
    used_method = st.session_state["last_method"]
    used_label_col = st.session_state["last_label_col"]
    used_label_values = st.session_state.get("last_label_values")
    used_dims = st.session_state.get("last_n_output_dims", 2)

    # Scatter plot
    if embedding.shape[1] == 1:
        st.info(
            "The selected variance ratio threshold was met by a single component. "
            "The plot shows that component on the X axis and the sample index on the Y axis."
        )
    title = f"{used_method} — {embedding.shape[0]} samples → {embedding.shape[1]}D"
    fig = make_scatter(
        embedding=embedding,
        labels=used_label_values,
        label_name=used_label_col or "index",
        title=title,
        dims=used_dims,
    )
    st.plotly_chart(fig, width="stretch")

    # Metrics
    st.subheader("Metrics")

    if used_method in ("PCA", "LDA") and "explained_variance_ratio" in metrics:
        evr = metrics["explained_variance_ratio"]
        var_fig = make_variance_chart(evr, used_method)
        st.plotly_chart(var_fig, width="stretch")

        total_pct = sum(evr) * 100
        mc1, mc2 = st.columns(2)
        mc1.metric("Components used", metrics.get("n_components_", len(evr)))
        mc2.metric("Total variance explained", f"{total_pct:.1f}%")

    elif used_method == "t-SNE":
        st.metric("Final KL divergence", f"{metrics['kl_divergence']:.4f}")
        st.caption(
            "Lower KL divergence indicates the low-dimensional distribution "
            "more faithfully represents the high-dimensional one."
        )

    elif used_method == "UMAP":
        mc1, mc2 = st.columns(2)
        mc1.metric("Fitted a", f"{metrics['a']:.4f}")
        mc2.metric("Fitted b", f"{metrics['b']:.4f}")
        st.caption(
            "Parameters a and b shape the low-dimensional kernel "
            "1 / (1 + a·d^(2b)); they are fitted to the chosen min_dist / spread."
        )

    if "trustworthiness" in metrics:
        st.divider()
        k_used = metrics.get("trustworthiness_k", 5)
        tw = metrics["trustworthiness"]
        st.metric(
            label=f"Trustworthiness (k={k_used})",
            value=f"{tw:.4f}",
            help=(
                "Measures how well the local neighbourhood structure of the original "
                "space is preserved in the embedding. "
                f"Of the {k_used} nearest neighbours in the embedding, how many were "
                f"also among the {k_used} nearest in the original space? "
                "Score 1.0 = perfect preservation."
            ),
        )
        if tw >= 0.95:
            st.caption("Excellent — neighbourhood structure is very well preserved.")
        elif tw >= 0.90:
            st.caption("Good — most local structure is preserved.")
        elif tw >= 0.80:
            st.caption("Moderate — some local distortion present.")
        else:
            st.caption("Low — significant local distortion; try adjusting hyperparameters.")

    # Download
    st.subheader("Download embedding")
    emb_df = pd.DataFrame(
        embedding,
        columns=[f"dim_{i + 1}" for i in range(embedding.shape[1])],
    )
    if used_label_values is not None:
        emb_df[used_label_col] = used_label_values

    csv_buf = io.StringIO()
    emb_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_buf.getvalue(),
        file_name=f"{used_method.lower()}_embedding.csv",
        mime="text/csv",
    )
