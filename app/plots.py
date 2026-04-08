"""
Plotting helpers — all functions return Plotly figures ready for st.plotly_chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Scatter plot (2D / 3D)
# ---------------------------------------------------------------------------


def make_scatter(
    embedding: np.ndarray,
    labels: np.ndarray | None,
    label_name: str,
    title: str,
    dims: int[2 | 3] = 2,
) -> go.Figure:
    """
    Build an interactive scatter plot of the DR embedding.

    Parameters
    ----------
    embedding : ndarray of shape (n_samples, n_components)
    labels    : optional 1-D array used for colour.  Strings → discrete palette,
                numbers → continuous colour scale.
    label_name: column name shown in the legend / colour bar.
    title     : plot title.
    dims      : 2 or 3.
    """
    n = embedding.shape[0]
    n_components = embedding.shape[1]
    df = pd.DataFrame({"index": np.arange(n)})

    if n_components == 1:
        # 1-D embedding: plot the single component against sample index (strip plot).
        df["x"] = embedding[:, 0]
        df["y"] = np.arange(n, dtype=float)
        dims = 2  # force 2-D rendering
    elif dims == 3 and n_components >= 3:
        df["x"] = embedding[:, 0]
        df["y"] = embedding[:, 1]
        df["z"] = embedding[:, 2]
    else:
        df["x"] = embedding[:, 0]
        df["y"] = embedding[:, 1]

    color_col = None
    if labels is not None:
        df[label_name] = labels
        color_col = label_name

    is_categorical = color_col is not None and (
        df[color_col].dtype == object or str(df[color_col].dtype).startswith("category")
    )

    color_kwargs: dict = {}
    if color_col is not None:
        if is_categorical:
            color_kwargs = {
                "color": color_col,
                "color_discrete_sequence": px.colors.qualitative.Safe,
            }
        else:
            color_kwargs = {"color": color_col, "color_continuous_scale": "Viridis"}

    hover_data = {"index": True}
    if color_col:
        hover_data[color_col] = True

    is_1d = n_components == 1
    x_label = "Component 1"
    y_label = "Sample index" if is_1d else "Component 2"

    if dims == 3 and "z" in df.columns:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            title=title,
            labels={"x": x_label, "y": y_label, "z": "Component 3"},
            hover_data=hover_data,
            **color_kwargs,
        )
        fig.update_traces(marker={"size": 4, "opacity": 0.85})
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            title=title,
            labels={"x": x_label, "y": y_label},
            hover_data=hover_data,
            **color_kwargs,
        )
        fig.update_traces(
            marker={"size": 7, "opacity": 0.85, "line": {"width": 0.5, "color": "white"}}
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text=label_name,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        font={"size": 13},
        xaxis={"gridcolor": "#e5e5e5", "zeroline": False},
        yaxis={"gridcolor": "#e5e5e5", "zeroline": False},
    )
    return fig


# ---------------------------------------------------------------------------
# Explained variance bar chart (PCA / LDA)
# ---------------------------------------------------------------------------


def make_variance_chart(
    explained_variance_ratio: list[float],
    method: str,
) -> go.Figure:
    """
    Bar chart of per-component explained variance ratio with a cumulative line.
    """
    n = len(explained_variance_ratio)
    evr = explained_variance_ratio
    cumulative = list(pd.Series(evr).cumsum())

    component_labels = [f"{'PC' if method == 'PCA' else 'LD'}{i + 1}" for i in range(n)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=component_labels,
            y=[v * 100 for v in evr],
            name="Individual",
            marker_color="#4C72B0",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=component_labels,
            y=[v * 100 for v in cumulative],
            name="Cumulative",
            mode="lines+markers",
            line={"color": "#DD8452", "width": 2},
            marker={"size": 7},
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            yaxis="y",
        )
    )
    fig.update_layout(
        title=f"{method} — Explained Variance Ratio",
        xaxis_title="Component",
        yaxis={"title": "Variance Explained (%)", "range": [0, 105]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        font={"size": 13},
        xaxis={"gridcolor": "#e5e5e5"},
        yaxis_gridcolor="#e5e5e5",
    )
    return fig
