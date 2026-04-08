"""
Hyperparameter specifications and widget rendering for each DR algorithm.

Design
------
* ``ParamSpec`` holds all UI metadata for one parameter (label, widget type,
  bounds, step, format, help text).  Static values are used by default; supply
  a ``*_fn`` callable to derive the value at render-time from ``DataContext``
  (e.g. LDA's upper bound depends on the number of classes in the dataset).

* ``PARAM_SPECS`` is a registry that maps algorithm name -> list of ParamSpec.
  This is the *only* place that needs to change when a new algorithm is added
  or an existing one gains a new hyperparameter.

* ``render_params(method, ctx)`` is the single public entry point used by the
  Streamlit app.  It renders all widgets inside a labelled expander and returns
  a plain dict ready to pass to ``run_reduction``.

Extending
---------
To add a new algorithm ``"MyDR"``:
1. Append its ``ParamSpec`` list to ``PARAM_SPECS["MyDR"]``.
2. Optionally add a human-readable title to ``_EXPANDER_TITLES``.
3. If it has non-standard widget logic (like PCA's mode selector), write a
   dedicated ``_render_mydr_params`` function and dispatch to it in
   ``render_params``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import streamlit as st

# ---------------------------------------------------------------------------
# DataContext — runtime constraints derived from the selected dataset
# ---------------------------------------------------------------------------


@dataclass
class DataContext:
    """Snapshot of dataset properties used to resolve dynamic widget bounds."""

    n_features: int = 1
    n_classes: int = 2
    n_output_dims: int = 2


# ---------------------------------------------------------------------------
# ParamSpec — metadata for a single hyperparameter widget
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    """
    Describes one hyperparameter widget.

    Parameters
    ----------
    name       : key in the returned params dict (must match the algorithm's
                 ``__init__`` argument name exactly).
    label      : human-readable Streamlit widget label.
    widget     : ``"slider"`` or ``"number_input"``.
    default    : static default value; ignored when ``default_fn`` is given.
    help       : tooltip text shown next to the widget.
    min_value  : static minimum; ignored when ``min_fn`` is given.
    max_value  : static maximum; ignored when ``max_fn`` is given.
    step       : slider / input step size.
    format_str : printf-style format string passed to ``st.slider``.
    min_fn     : ``(DataContext) -> min_value`` — overrides ``min_value``.
    max_fn     : ``(DataContext) -> max_value`` — overrides ``max_value``.
    default_fn : ``(DataContext) -> default`` — overrides ``default``.
    """

    name: str
    label: str
    widget: Literal["slider", "number_input"]
    default: Any
    help: str = ""
    min_value: Any = None
    max_value: Any = None
    step: Any = None
    format_str: str | None = None
    min_fn: Callable[[DataContext], Any] | None = None
    max_fn: Callable[[DataContext], Any] | None = None
    default_fn: Callable[[DataContext], Any] | None = None


# ---------------------------------------------------------------------------
# Specs registry
# ---------------------------------------------------------------------------


def _lda_max_k(ctx: DataContext) -> int:
    return max(min(ctx.n_classes - 1, ctx.n_features), 1)


PARAM_SPECS: dict[str, list[ParamSpec]] = {
    "LDA": [
        ParamSpec(
            name="n_components",
            label="n_components",
            widget="slider",
            default=2,
            min_value=1,
            max_fn=_lda_max_k,
            default_fn=lambda ctx: min(ctx.n_output_dims, _lda_max_k(ctx)),
            help=(
                "Number of discriminant components to keep. "
                "Hard upper bound is min(n_classes - 1, n_features) "
                "because LDA can only separate C classes with at most C-1 axes."
            ),
        ),
    ],
    "t-SNE": [
        ParamSpec(
            name="perplexity",
            label="Perplexity",
            widget="slider",
            default=30.0,
            min_value=5.0,
            max_value=50.0,
            step=1.0,
            help="Balances local vs. global structure. Typical range: 5-50.",
        ),
        ParamSpec(
            name="learning_rate",
            label="Learning rate",
            widget="slider",
            default=200.0,
            min_value=10.0,
            max_value=1000.0,
            step=10.0,
            help=(
                "Step size for gradient descent. t-SNE gradients are sums over all "
                "point pairs, so they are naturally large - values of 10-1000 are "
                "normal. The original paper recommends 200 as a starting point."
            ),
        ),
        ParamSpec(
            name="n_iter",
            label="Iterations",
            widget="slider",
            default=1000,
            min_value=250,
            max_value=2000,
            step=50,
            help=(
                "Total number of optimisation steps. More iterations improve "
                "convergence but increase runtime. 1000 is sufficient for most datasets."
            ),
        ),
        ParamSpec(
            name="early_exaggeration",
            label="Early exaggeration",
            widget="slider",
            default=12.0,
            min_value=1.0,
            max_value=50.0,
            step=1.0,
            help=(
                "Multiplies the high-dimensional affinities in the first phase of "
                "optimisation, pushing clusters further apart before fine-tuning. "
                "Values of 4-12 are typical; higher values create more separated clusters."
            ),
        ),
        ParamSpec(
            name="random_state",
            label="Random state",
            widget="number_input",
            default=42,
            min_value=0,
            step=1,
            help="Seed for reproducibility. Same seed always produces the same embedding.",
        ),
    ],
    "UMAP": [
        ParamSpec(
            name="n_neighbors",
            label="n_neighbors",
            widget="slider",
            default=15,
            min_value=2,
            max_value=100,
            help=(
                "Number of nearest neighbours used to build the graph. "
                "Larger values preserve more global structure; "
                "smaller values capture finer local patterns."
            ),
        ),
        ParamSpec(
            name="min_dist",
            label="min_dist",
            widget="slider",
            default=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format_str="%.2f",
            help=(
                "Minimum distance between embedded points. "
                "Smaller values create tighter, denser clusters."
            ),
        ),
        ParamSpec(
            name="spread",
            label="Spread",
            widget="slider",
            default=1.0,
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            help=(
                "Controls the overall scale of the embedding. Together with min_dist, "
                "it determines how tightly points are clustered. "
                "Should be >= min_dist; 1.0 is a sensible default."
            ),
        ),
        ParamSpec(
            name="n_epochs",
            label="Epochs",
            widget="slider",
            default=200,
            min_value=50,
            max_value=500,
            step=50,
            help=(
                "Number of SGD passes over the graph edges. More epochs improve "
                "layout quality at the cost of runtime. 200 works well for most datasets."
            ),
        ),
        ParamSpec(
            name="learning_rate",
            label="Learning rate",
            widget="slider",
            default=1.0,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            help=(
                "Step size for the SGD layout optimisation. Unlike t-SNE, UMAP "
                "updates are per-edge (not over all pairs), so gradients stay small "
                "and 1.0 is the standard default."
            ),
        ),
        ParamSpec(
            name="random_state",
            label="Random state",
            widget="number_input",
            default=42,
            min_value=0,
            step=1,
            help="Seed for reproducibility. Same seed always produces the same embedding.",
        ),
    ],
}

_EXPANDER_TITLES: dict[str, str] = {
    "LDA": "LDA settings",
    "t-SNE": "t-SNE settings",
    "UMAP": "UMAP settings",
}


# ---------------------------------------------------------------------------
# Widget rendering helpers
# ---------------------------------------------------------------------------


def _render_one(spec: ParamSpec, ctx: DataContext) -> Any:
    """Render the single widget described by *spec* and return its value."""
    lo = spec.min_fn(ctx) if spec.min_fn else spec.min_value
    hi = spec.max_fn(ctx) if spec.max_fn else spec.max_value
    val = spec.default_fn(ctx) if spec.default_fn else spec.default

    # Clamp default into the valid range to avoid Streamlit errors.
    if lo is not None and val is not None and val < lo:
        val = lo
    if hi is not None and val is not None and val > hi:
        val = hi

    if spec.widget == "slider":
        if lo is not None and lo == hi:
            # Only one valid value exists — skip the slider and show a note.
            st.info(
                f"**{spec.label}** is fixed to **{lo}** — "
                "only one valid value exists for the current dataset configuration."
            )
            return lo
        kwargs: dict[str, Any] = {
            "min_value": lo,
            "max_value": hi,
            "value": val,
            "help": spec.help,
        }
        if spec.step is not None:
            kwargs["step"] = spec.step
        if spec.format_str is not None:
            kwargs["format"] = spec.format_str
        return st.slider(spec.label, **kwargs)

    # number_input
    kwargs = {"min_value": lo, "value": val, "step": spec.step, "help": spec.help}
    if hi is not None:
        kwargs["max_value"] = hi
    return int(st.number_input(spec.label, **kwargs))


def _render_pca_params(ctx: DataContext) -> dict[str, Any]:
    """Render PCA's three-mode n_components selector."""
    params: dict[str, Any] = {}
    with st.expander("PCA settings", expanded=True):
        nc_mode = st.radio(
            "n_components mode",
            ["Integer", "Variance ratio", "All"],
            horizontal=True,
            help=(
                "Integer: keep a fixed number of components. "
                "Variance ratio: keep the fewest components that explain the target "
                "fraction of total variance. "
                "All: keep every component (no reduction in the PCA step itself)."
            ),
        )
        if nc_mode == "Integer":
            params["n_components"] = st.slider(
                "n_components (int)",
                min_value=1,
                max_value=max(ctx.n_features, 1),
                value=min(ctx.n_output_dims, ctx.n_features),
                help=(
                    "Number of principal components to keep. "
                    "Higher values retain more variance but increase dimensionality."
                ),
            )
        elif nc_mode == "Variance ratio":
            params["n_components"] = st.slider(
                "Variance ratio to retain",
                min_value=0.50,
                max_value=1.00,
                value=0.95,
                step=0.01,
                format="%.2f",
                help=(
                    "Keep the minimum number of components whose cumulative explained "
                    "variance reaches this threshold. E.g. 0.95 retains 95% of the "
                    "total variance."
                ),
            )
        else:
            params["n_components"] = None
            st.caption("All components are kept — PCA will not reduce dimensionality.")
    return params


def _render_lda_params(ctx: DataContext) -> dict[str, Any]:
    """Render LDA params with an informational caption about the component limit."""
    max_k = _lda_max_k(ctx)
    params: dict[str, Any] = {}
    with st.expander("LDA settings", expanded=True):
        st.caption(
            f"Max components for this dataset: **{max_k}** "
            f"(= min(n_classes-1, n_features) "
            f"= min({ctx.n_classes - 1}, {ctx.n_features}))"
        )
        for spec in PARAM_SPECS["LDA"]:
            params[spec.name] = _render_one(spec, ctx)
    return params


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_params(method: str, ctx: DataContext) -> dict[str, Any]:
    """
    Render all hyperparameter widgets for *method* inside a labelled expander.

    Parameters
    ----------
    method : algorithm name — must be one of ``"PCA"``, ``"LDA"``,
             ``"t-SNE"``, ``"UMAP"``.
    ctx    : runtime constraints (feature count, class count, output dims).

    Returns
    -------
    dict
        Hyperparameter values keyed by the algorithm's ``__init__`` argument
        names.  Merge with ``{"n_components": n_output_dims}`` before passing
        to ``run_reduction``.
    """
    if method == "PCA":
        return _render_pca_params(ctx)
    if method == "LDA":
        return _render_lda_params(ctx)

    specs = PARAM_SPECS.get(method)
    if not specs:
        return {}

    title = _EXPANDER_TITLES.get(method, f"{method} settings")
    params: dict[str, Any] = {}
    with st.expander(title, expanded=True):
        for spec in specs:
            params[spec.name] = _render_one(spec, ctx)
    return params
