"""
Start tego syfu:
streamlit run UI_code.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms_dimensions import run_all

#  PAGE CONFIG & GLOBAL STYLE
st.set_page_config(
    page_title="Dim Reduction Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d0d14;
        border-right: 1px solid #1e1e30;
    }
    section[data-testid="stSidebar"] * { color: #c8c8e0 !important; }

    /* ── main bg ── */
    .stApp { background: #080810; color: #e0e0f0; }

    /* ── title ── */
    .lab-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #7b6cff 0%, #00e5c3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .lab-sub {
        font-size: 0.85rem;
        color: #555580;
        margin-top: 2px;
        letter-spacing: 0.05em;
    }

    /* ── method cards (metric row) ── */
    .method-card {
        background: #10101e;
        border: 1px solid #1e1e38;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 6px;
    }
    .method-card .method-name {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #7b6cff;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .method-card .method-type {
        font-size: 0.72rem;
        color: #444466;
        margin-bottom: 6px;
    }
    .method-card .metric-val {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00e5c3;
    }
    .method-card .metric-label {
        font-size: 0.7rem;
        color: #555580;
    }
    .method-card .error-msg {
        font-size: 0.75rem;
        color: #ff5577;
    }

    /* ── viz panels ── */
    .viz-panel {
        background: #0e0e1c;
        border: 1px solid #1e1e38;
        border-radius: 12px;
        padding: 16px;
        min-height: 420px;
    }
    .panel-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #444466;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 8px;
    }

    /* ── swap button ── */
    .swap-area {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding-top: 60px;
    }

    /* ── divider ── */
    hr { border-color: #1e1e30 !important; }

    /* ── streamlit overrides ── */
    .stSelectbox label, .stSlider label, .stFileUploader label {
        color: #9090b0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-baseweb="select"] { background: #10101e; }
    .stButton > button {
        background: linear-gradient(135deg, #7b6cff, #00e5c3);
        color: #000;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        letter-spacing: 0.08em;
    }
    .stButton > button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


#  SESSION STATE INITIALISATION

METHODS = ["PCA", "LDA", "t-SNE", "UMAP"]
METHOD_META = {
    "PCA":   {"type": "Linear · Unsupervised",  "color": "#7b6cff"},
    "LDA":   {"type": "Linear · Supervised",    "color": "#ff9f43"},
    "t-SNE": {"type": "Non-linear · Unsupervised", "color": "#00e5c3"},
    "UMAP":  {"type": "Non-linear · Unsupervised", "color": "#ff5577"},
}

def _init_state():
    defaults = {
        "results": {}, # method → ndarray | error string
        "labels": None, # ndarray dla klasy (bo kolorki potzrebnują)
        "panel_left":  "PCA",
        "panel_right": "t-SNE",
        "ran": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

#  SIDEBAR
with st.sidebar:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown("## ⚗️ Panel do magi")
    st.markdown("---")

    # data upload
    st.markdown("**Źródło danych**")
    uploaded = st.file_uploader(
        "Drop a CSV file",
        type=["csv"],
        help="Pierwszy rząd = column headers. Numeryczne kolumny będą traktowane jak cechy.",
    )

    use_demo = st.checkbox("Demo dataset (Iris)", value=(uploaded is None))

    df: pd.DataFrame | None = None

    if uploaded is not None and not use_demo:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        st.info("Używany jest Iris")

    st.markdown("---")

    # column selection
    label_col = None
    if df is not None:
        all_cols = df.columns.tolist()
        label_col = st.selectbox(
            "Label / target column (for LDA & colouring)",
            options=["— none —"] + all_cols,
            index=0 if "species" not in all_cols else all_cols.index("species") + 1,
        )
        if label_col == "— none —":
            label_col = None

    st.markdown("---")

    #output dimensions
    n_components = st.slider(
        "Wymiary na wyjściu",
        min_value=1,
        max_value=3,
        value=2,
        help="1, 2 i 3 można ładnie narysować",
    )

    st.markdown("---")

    run_btn = st.button("▶  START LICZENIA", use_container_width=True)


#  START ALGORYTMÓW

if run_btn and df is not None:
    numeric_df = df.select_dtypes(include=[np.number])

    if label_col and label_col in df.columns:
        numeric_df = numeric_df.drop(columns=[label_col], errors="ignore")

    if numeric_df.shape[1] < 1:
        st.error("No numeric feature columns found.  Please check your CSV.")
    else:
        data_arr = numeric_df.values
        labels_arr = df[label_col].values if label_col else None

        with st.spinner("Running dimensionality reduction…"):
            results = run_all(data_arr, labels_arr, n_components)

        st.session_state["results"] = results
        st.session_state["labels"] = labels_arr
        st.session_state["ran"] = True

#  HEADER
st.markdown('<h1 class="lab-title">Redukcja Wymiarowości - temat 6</h1>', unsafe_allow_html=True)
st.markdown('<p class="lab-sub">PCA * LDA * t-SNE * UMAP</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


#  METRYTKA EFEKTYWNOŚCI - TRUSTWORTHINESS
st.markdown("### Metryka Efektywności")
st.caption("Trustworthiness zostanie tutaj później wstawione")

metric_cols = st.columns(4)
PLACEHOLDER_SCORES = {"PCA": 3.163, "LDA": 3.163, "t-SNE": 3.163, "UMAP": 3.163}

for col, method in zip(metric_cols, METHODS):
    meta = METHOD_META[method]
    result = st.session_state["results"].get(method)
    is_error = isinstance(result, str)

    with col:
        if is_error:
            st.markdown(
                f"""
                <div class="method-card">
                  <div class="method-name">{method}</div>
                  <div class="method-type">{meta["type"]}</div>
                  <div class="error-msg">⚠ {result[:80]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            score = PLACEHOLDER_SCORES[method] if st.session_state["ran"] else "—"
            score_html = (
                f'<span class="metric-val">{score}</span>'
                if st.session_state["ran"]
                else '<span class="metric-val" style="color:#333355">—</span>'
            )
            shape_html = ""
            if isinstance(result, np.ndarray):
                shape_html = f'<div class="metric-label">output shape: {result.shape[0]} × {result.shape[1]}</div>'
            st.markdown(
                f"""
                <div class="method-card">
                  <div class="method-name">{method}</div>
                  <div class="method-type">{meta["type"]}</div>
                  {score_html}
                  <div class="metric-label">trustworthiness (placeholder)</div>
                  {shape_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<hr>", unsafe_allow_html=True)


#  VISUALISATION HELPERS
_PLOTLY_THEME = dict(
    paper_bgcolor="#0e0e1c",
    plot_bgcolor="#0e0e1c",
    font_color="#c8c8e0",
    margin=dict(l=20, r=20, t=40, b=20),
)

def _color_seq(labels):
    """Return a list of hex colours aligned to each point."""
    if labels is None:
        return None
    unique = list(dict.fromkeys(str(l) for l in labels))
    palette = [
        "#7b6cff", "#00e5c3", "#ff9f43", "#ff5577",
        "#54a0ff", "#5f27cd", "#01cbc4", "#feca57",
        "#ff9ff3", "#48dbfb",
    ]
    cmap = {u: palette[i % len(palette)] for i, u in enumerate(unique)}
    return [cmap[str(l)] for l in labels], [str(l) for l in labels], cmap


def make_figure(method: str, results: dict, labels_arr) -> go.Figure | None:
    """Build a Plotly figure for the given method and n-D result."""
    data = results.get(method)
    if data is None or isinstance(data, str):
        return None

    n_dims = data.shape[1]
    color_info = _color_seq(labels_arr)
    colors = color_info[0] if color_info else None
    label_strs = color_info[1] if color_info else None
    cmap = color_info[2] if color_info else {}

    axis_labels = {
        1: ["Component 1"],
        2: ["Component 1", "Component 2"],
        3: ["Component 1", "Component 2", "Component 3"],
    }.get(n_dims, [f"Dim {i+1}" for i in range(n_dims)])

    meta = METHOD_META[method]
    title = f"<b>{method}</b>  <span style='font-size:11px;color:#444466'>{meta['type']}</span>"

    if n_dims == 1:
        fig = go.Figure()
        for lbl in (list(cmap.keys()) if cmap else [None]):
            mask = (
                np.array([str(l) == lbl for l in labels_arr])
                if labels_arr is not None and lbl is not None
                else np.ones(len(data), dtype=bool)
            )
            fig.add_trace(go.Histogram(
                x=data[mask, 0],
                name=str(lbl) if lbl else "data",
                marker_color=cmap.get(lbl, meta["color"]),
                opacity=0.75,
            ))
        fig.update_layout(
            title=title,
            xaxis_title=axis_labels[0],
            barmode="overlay",
            **_PLOTLY_THEME,
        )

    elif n_dims == 2:
        fig = go.Figure()
        if colors:
            for lbl, clr in cmap.items():
                mask = np.array([str(l) == lbl for l in labels_arr])
                fig.add_trace(go.Scatter(
                    x=data[mask, 0], y=data[mask, 1],
                    mode="markers",
                    name=lbl,
                    marker=dict(color=clr, size=5, opacity=0.75),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data[:, 0], y=data[:, 1],
                mode="markers",
                marker=dict(color=meta["color"], size=5, opacity=0.75),
            ))
        fig.update_layout(
            title=title,
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            **_PLOTLY_THEME,
        )

    elif n_dims == 3:
        fig = go.Figure()
        if colors:
            for lbl, clr in cmap.items():
                mask = np.array([str(l) == lbl for l in labels_arr])
                fig.add_trace(go.Scatter3d(
                    x=data[mask, 0], y=data[mask, 1], z=data[mask, 2],
                    mode="markers",
                    name=lbl,
                    marker=dict(color=clr, size=3, opacity=0.8),
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=data[:, 0], y=data[:, 1], z=data[:, 2],
                mode="markers",
                marker=dict(color=meta["color"], size=3, opacity=0.8),
            ))
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
                bgcolor="#0e0e1c",
                xaxis=dict(color="#444466"),
                yaxis=dict(color="#444466"),
                zaxis=dict(color="#444466"),
            ),
            **_PLOTLY_THEME,
        )
    else:
        return None  # > 3 dims: can't plot

    return fig



#  DUAL VISUALISATION PANELS

st.markdown("### Panel wizualizacji")
st.caption("Wybór metody do pokazania. Wykresy wymagają 1, 2 albo 3 wymiarów.")

# WYBÓR METODY w panelu wizualizacji
sel_col_l, swap_col, sel_col_r = st.columns([5, 1, 5])

with sel_col_l:
    st.session_state["panel_left"] = st.selectbox(
        "Panel A",
        options=METHODS,
        index=METHODS.index(st.session_state["panel_left"]),
        key="sel_left",
    )

with sel_col_r:
    st.session_state["panel_right"] = st.selectbox(
        "Panel B",
        options=METHODS,
        index=METHODS.index(st.session_state["panel_right"]),
        key="sel_right",
    )

# plot panels
panel_l, panel_r = st.columns(2)

for panel_col, method_key in [(panel_l, "panel_left"), (panel_r, "panel_right")]:
    method = st.session_state[method_key]
    panel_label = "A" if method_key == "panel_left" else "B"
    with panel_col:
        st.markdown(
            f'<div class="panel-label">Panel {panel_label} · {method}</div>',
            unsafe_allow_html=True,
        )
        if not st.session_state["ran"]:
            st.markdown(
                '<div class="viz-panel" style="display:flex;align-items:center;'
                'justify-content:center;color:#222240;font-size:0.9rem;">'
                'Run the algorithms to see results</div>',
                unsafe_allow_html=True,
            )
        else:
            result = st.session_state["results"].get(method)
            if isinstance(result, str):
                st.markdown(
                    f'<div class="viz-panel"><span style="color:#ff5577;font-size:0.8rem;">'
                    f'⚠ {result}</span></div>',
                    unsafe_allow_html=True,
                )
            elif isinstance(result, np.ndarray):
                if result.shape[1] > 3:
                    st.markdown(
                        '<div class="viz-panel" style="display:flex;align-items:center;'
                        'justify-content:center;color:#444466;font-size:0.85rem;">'
                        f'Cannot visualise {result.shape[1]}-D output.<br>'
                        'Set output dimensions to 1, 2 or 3.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    fig = make_figure(method, st.session_state["results"], st.session_state["labels"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)


#  FOOTER
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(
#     '<p style="font-size:0.72rem;color:#222240;text-align:center;">'
#     'Dimensionality Reduction Lab · PCA · LDA · t-SNE · UMAP</p>',
#     unsafe_allow_html=True,
# )