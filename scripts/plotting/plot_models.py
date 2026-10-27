"""Compare model results across metrics for a given hospital.

Usage examples
--------------
Internal validation (single center):

    python scripts/plotting/plot_models.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV

Hero sub-cohort:

    python scripts/plotting/plot_models.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV --hero

Multi-center comparison (two output figures):

    python scripts/plotting/plot_models.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV USZ KISPI

Restrict metrics / models / columns:

    python scripts/plotting/plot_models.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV \\
        --metrics roc_auc auprc recall \\
        --models xgb lr tabpfn \\
        --ncols 3
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adonis_ml.machine_learning.utils import get_results_path, get_figures_path
from adonis_ml.machine_learning.plotting import (
    MODEL_LABELS, METRIC_LABELS, MODEL_COLORS, CENTER_COLORS,
    MODEL_GROUPS, GROUP_ORDER, PUB_RC, _save_fig, style_axes,
)
from adonis_ml.utils import load_and_parse_config, AdonisConfig


logger = logging.getLogger(__name__)

INTERNAL_HOSPITAL = "CHUV"

# Figure geometry constants (inches). Axes are fixed size; the figure grows
# only when rows/columns are added, so panels stay comparable across figures.
PANEL_W: float = 3.5
PANEL_H: float = 2.5

# Bar cap for performance metrics (all in [0, 1]).
_YMAX_CAP: float = 1.0
_BAR_HEADROOM: float = 1.06  # multiplier of max(mean+std) for axes top


# ── helpers ──────────────────────────────────────────────────────────────────

def _sorted_models(model_names: list[str]) -> list[str]:
    def _rank(m):
        g = MODEL_GROUPS.get(m, "")
        return GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
    return sorted(model_names, key=_rank)


def _group_positions(model_names: list[str], gap: float = 0.5) -> np.ndarray:
    positions = []
    pos = 0.0
    prev_group = None
    for m in model_names:
        group = MODEL_GROUPS.get(m, "")
        if prev_group is not None and group != prev_group:
            pos += gap
        positions.append(pos)
        pos += 1.0
        prev_group = group
    return np.array(positions)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model results across metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--hospital", required=True, nargs="+",
                        help="Hospital name(s): CHUV, USZ, KISPI.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model types (default: all from config).")
    parser.add_argument("--ncols", type=int, default=2,
                        help="Number of subplot columns (default: 2).")
    parser.add_argument("--split", default="test", choices=["test", "vali"])
    parser.add_argument("--external-suffix", default=None)
    parser.add_argument("--hero", action="store_true",
                        help="Compare on the HeRO sub-cohort.")
    parser.add_argument("--add-stat", action="store_true",
                        help="Overlay significance brackets from statistical_tests_center_*.parquet.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _substitute_user(config: AdonisConfig) -> None:
    if getattr(config, "environment", None) == "BioMedIT":
        user = os.getlogin()
        for k, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[k] = v.replace("$USER", user)


def _pkl_suffix(hospital: str, external_suffix: str | None) -> str:
    if hospital == INTERNAL_HOSPITAL:
        return ""
    if external_suffix is not None:
        return external_suffix
    return f"_external_{hospital}_cv"


def load_all_results(
    config: AdonisConfig,
    hospital: str,
    split: str,
    external_suffix: str | None,
    requested_models: list[str] | None = None,
) -> dict[str, dict]:
    results_path = get_results_path(config)
    suffix = _pkl_suffix(hospital, external_suffix)
    model_results: dict[str, dict] = {}
    model_names = requested_models or list(config.model["models"].keys())

    for model_name in model_names:
        pkl_path = os.path.join(results_path, f"{model_name}_sklearn{suffix}.pkl")
        if not os.path.exists(pkl_path):
            logger.warning("Results file not found, skipping: %s", pkl_path)
            continue
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

        results = saved.get("results", {})
        if split in results:
            results = results[split]
        elif isinstance(next(iter(results.values()), None), dict) and "mean" not in next(iter(results.values())):
            logger.warning("Split '%s' not found for '%s'", split, model_name)
            continue

        prefix = f"{split}_"
        results = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in results.items()}
        model_results[model_name] = results

    return model_results


def load_hero_results(
    config: AdonisConfig,
    requested_models: list[str] | None = None,
) -> dict[str, dict]:
    results_path = get_results_path(config)
    model_results: dict[str, dict] = {}
    hero_score_loaded = False
    model_names = requested_models or list(config.model["models"].keys())

    for model_name in model_names:
        pkl_path = os.path.join(results_path, f"{model_name}_sklearn.pkl")
        if not os.path.exists(pkl_path):
            logger.warning("Results file not found, skipping: %s", pkl_path)
            continue
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

        if "results_hero_ml" not in saved:
            logger.warning("No hero results in '%s'", pkl_path)
            continue
        model_results[model_name] = saved["results_hero_ml"]

        if not hero_score_loaded and "results_hero_score" in saved:
            model_results["hero_predictor"] = saved["results_hero_score"]
            hero_score_loaded = True

    return model_results


def _select_metrics(config: AdonisConfig, requested: list[str] | None) -> list[str]:
    available = list(config.metrics.keys())
    if requested is None:
        return available
    return [m for m in requested if m in available]


def write_markdown(model_names, metrics, rows, output_path, hospital, split):
    display_names = [MODEL_LABELS.get(m, m) for m in model_names]
    lines = [
        f"# Model comparison — {hospital} ({split})\n",
        "| Metric | " + " | ".join(display_names) + " |",
        "| --- |" + " --- |" * len(display_names),
    ]
    for metric, row in zip(metrics, rows):
        lines.append("| " + metric + " | " + " | ".join(row) + " |")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown saved to %s", output_path)


def build_table(model_results, metrics):
    model_names = list(model_results.keys())
    rows, used_metrics = [], []
    for metric in metrics:
        row = []
        for model in model_names:
            m = model_results[model].get(metric)
            row.append(f"{m['mean']:.3f} ({m['std']:.3f})" if m else "—")
        if any(c != "—" for c in row):
            rows.append(row)
            used_metrics.append(metric)
    return model_names, used_metrics, rows


def write_hero_summary(model_results, metrics, output_path, hospital):
    display = {m: MODEL_LABELS.get(m, m) for m in model_results}
    ml_models = [m for m in model_results if m != "hero_predictor"]
    hero = model_results.get("hero_predictor", {})
    lines = [
        f"# HeRO sub-cohort comparison — {hospital}\n",
        "| Metric | HeRO Score | " + " | ".join(display[m] for m in ml_models) + " | Best ML Δ vs HeRO |",
        "| --- | --- |" + " --- |" * len(ml_models) + " --- |",
    ]
    for metric in metrics:
        hero_m = hero.get(metric)
        hero_str = f"{hero_m['mean']:.3f} ({hero_m['std']:.3f})" if hero_m else "—"
        ml_cells, ml_means = [], []
        for m in ml_models:
            v = model_results[m].get(metric)
            if v:
                ml_cells.append(f"{v['mean']:.3f} ({v['std']:.3f})")
                ml_means.append(v["mean"])
            else:
                ml_cells.append("—")
        delta_str = f"{max(ml_means) - hero_m['mean']:+.3f}" if hero_m and ml_means else "—"
        lines.append("| " + metric + " | " + hero_str + " | " + " | ".join(ml_cells) + " | " + delta_str + " |")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Hero summary saved to %s", output_path)


# ── statistical significance helpers ─────────────────────────────────────────

def load_stat_results(
    results_path: str,
    hospital: str,
    hero: bool = False,
) -> pd.DataFrame | None:
    """Load significant rows from a statistical_tests_center_*.parquet file."""
    suffix = "_hero" if hero else ""
    path = os.path.join(results_path, f"statistical_tests_center_{hospital}{suffix}.parquet")
    if not os.path.isfile(path):
        logger.warning("Stat results not found (run run_statistical_tests.py first): %s", path)
        return None
    df = pd.read_parquet(path)
    sig = df[df["reject_h0"] == True].reset_index(drop=True)
    return sig if not sig.empty else None


def _significance_label(p_adj: float) -> str:
    if pd.isna(p_adj) or p_adj >= 0.05:
        return "*"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    return "*"


def _draw_brackets(
    ax,
    brackets: list[tuple[float, float, str]],
    y_top: float,
    gap_frac: float = 0.04,
    step_frac: float = 0.07,
) -> float:
    """Draw stacked significance brackets above ``y_top`` with clip_on=False.

    Brackets are rendered above the axes (they may exceed the axis limits, which
    is desirable for bounded [0, 1] performance metrics). The returned value is
    the top data-y coordinate reached by the highest bracket, so the caller can
    grow the figure's top margin to keep them inside the image.
    """
    if not brackets:
        return 0.0
    brackets = sorted(brackets, key=lambda b: abs(b[1] - b[0]))

    tick_h = y_top * 0.012
    y0 = y_top * (1.0 + gap_frac)
    step = y_top * step_frac

    y_max = y0
    for i, (x1, x2, label) in enumerate(brackets):
        y = y0 + i * step
        ax.plot([x1, x1, x2, x2], [y - tick_h, y, y, y - tick_h],
                color="black", linewidth=0.7, clip_on=False)
        ax.text((x1 + x2) / 2, y + tick_h * 0.5, label,
                ha="center", va="bottom", fontsize=7, clip_on=False)
        y_max = max(y_max, y + tick_h * 4)

    return y_max


def _expand_margins_for_brackets(
    fig: plt.Figure,
    bracket_overflows: list[tuple[plt.Axes, float]],
    pad_in: float = 0.06,
) -> None:
    """Grow the figure's top margin (and row gap, if needed) to fit brackets.

    Unlike growing the plot area, growing the margins keeps the axes size and
    y-limits intact — so brackets drawn with ``clip_on=False`` above bounded
    metrics stay visible without stretching the [0, 1] axis.

    Parameters
    ----------
    bracket_overflows : list of (ax, y_bracket_max)
        One entry per axes that has brackets; ``y_bracket_max`` is the highest
        bracket top in data coordinates for that axes.
    pad_in : float
        Extra padding above the tallest bracket, in inches.
    """
    if not bracket_overflows:
        return

    fig_w, fig_h = fig.get_size_inches()
    sp = fig.subplotpars

    # Per-row overflow above the axes top, in inches.
    row_overflow_in: dict[int, float] = {}
    total_rows = 1
    for ax, y_bmax in bracket_overflows:
        ylim_top = ax.get_ylim()[1]
        if ylim_top <= 0 or y_bmax <= ylim_top:
            continue
        bbox = ax.get_position()
        ax_h_in = bbox.height * fig_h
        overflow_in = (y_bmax - ylim_top) * ax_h_in / ylim_top
        # Row index within the GridSpec (0 = top row).
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        total_rows = max(total_rows, gs.nrows)
        row = ss.rowspan.start
        row_overflow_in[row] = max(row_overflow_in.get(row, 0.0), overflow_in)

    if not row_overflow_in:
        return

    # Fixed margins & spacing in inches (recovered from current layout).
    left_in   = sp.left * fig_w
    right_in  = (1 - sp.right) * fig_w
    bottom_in = sp.bottom * fig_h
    top_in    = (1 - sp.top) * fig_h
    plot_h    = fig_h - bottom_in - top_in
    ax_h_in   = plot_h / (total_rows + (total_rows - 1) * sp.hspace)
    hspace_in = ax_h_in * sp.hspace  # inches between rows

    # Grow top margin for row 0 overflow; grow hspace for lower rows.
    top_extra   = row_overflow_in.get(0, 0.0)
    row_extra   = max((v for r, v in row_overflow_in.items() if r > 0), default=0.0)
    new_top_in  = top_in + (top_extra + pad_in if top_extra > 0 else 0.0)
    new_hsp_in  = hspace_in + (row_extra + pad_in if row_extra > 0 else 0.0)

    if new_top_in == top_in and new_hsp_in == hspace_in:
        return

    new_plot_h  = total_rows * ax_h_in + (total_rows - 1) * new_hsp_in
    new_fig_h   = bottom_in + new_plot_h + new_top_in
    new_hspace  = new_hsp_in / ax_h_in if ax_h_in > 0 else sp.hspace

    fig.set_size_inches(fig_w, new_fig_h)
    fig.subplots_adjust(
        left=left_in / fig_w,
        right=1 - right_in / fig_w,
        bottom=bottom_in / new_fig_h,
        top=1 - new_top_in / new_fig_h,
        hspace=new_hspace,
        wspace=sp.wspace,
    )


def _make_fig(
    nrows: int,
    ncols: int,
    panel_w: float,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with fixed-size subplots.

    Each subplot has dimensions:
        panel_w × PANEL_H (inches)

    The figure expands only by adding rows/columns while keeping the
    subplot sizes constant.
    """

    # ----- Fixed margins (inches) -----
    LEFT = 0.80
    RIGHT = 0.20
    TOP = 0.25       # no titles are drawn; brackets extend the top via
                     # _expand_margins_for_brackets when needed
    BOTTOM = 1.20    # room for rotated x-tick labels + figure-level legend

    # ----- Fixed spacing between subplots (inches) -----
    WSPACE = 0.60
    HSPACE = 1.4

    fig_w = LEFT + ncols * panel_w + (ncols - 1) * WSPACE + RIGHT
    fig_h = TOP + nrows * PANEL_H + (nrows - 1) * HSPACE + BOTTOM

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        dpi=300,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=LEFT / fig_w,
        right=1 - RIGHT / fig_w,
        bottom=BOTTOM / fig_h,
        top=1 - TOP / fig_h,
        wspace=WSPACE / panel_w,
        hspace=HSPACE / PANEL_H,
    )

    return fig, axes


def _finalize(
    fig: plt.Figure,
    handles: list,
    labels: list,
    ncol: int,
    fname: str,
) -> None:
    """Place legend at figure bottom and save without bbox_inches expansion."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, 0))
    _save_fig(fig, fname, dpi=300)
    plt.close(fig)


def _cap_ylim(y_top: float) -> float:
    """Return the axes upper limit, clamped to the metric cap (1.0)."""
    if y_top <= 0:
        return _YMAX_CAP
    return min(_YMAX_CAP, y_top * _BAR_HEADROOM)


# ── single-center bar chart ───────────────────────────────────────────────────

def plot_comparison(
    model_results: dict[str, dict],
    metrics: list[str],
    output_dir: str,
    hospital: str,
    suffix: str = "",
    ncols: int = 2,
    stat_df: pd.DataFrame | None = None,
) -> None:
    """One figure with a subplot grid: models on x-axis, bar color = MODEL_COLORS."""
    model_names = _sorted_models(list(model_results.keys()))
    x = _group_positions(model_names)

    n_metrics = len(metrics)
    nrows = math.ceil(n_metrics / ncols)
    panel_w = max(PANEL_W, len(model_names) * 0.65)

    with plt.rc_context(PUB_RC):
        model_handles = [
            mpatches.Patch(facecolor=MODEL_COLORS.get(m, "#999999"), edgecolor="black",
                           linewidth=0.5, label=MODEL_LABELS.get(m, m))
            for m in model_names
        ]

        fig, axes = _make_fig(nrows, ncols, panel_w)
        axes_flat = axes.flatten()

        bracket_overflows: list[tuple[plt.Axes, float]] = []
        for ax, metric in zip(axes_flat, metrics):
            bar_colors = [MODEL_COLORS.get(m, "#999999") for m in model_names]
            means = [model_results[m][metric]["mean"] if model_results[m].get(metric) else 0.0 for m in model_names]
            stds  = [model_results[m][metric]["std"]  if model_results[m].get(metric) else 0.0 for m in model_names]

            ax.bar(x, means, yerr=stds, color=bar_colors, capsize=2.5, width=0.6,
                   edgecolor="black", linewidth=0.5, ecolor="black",
                   error_kw={"linewidth": 0.7})
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_names],
                               rotation=30, ha="right")
            ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
            y_top = max(m + s for m, s in zip(means, stds)) if any(m > 0 for m in means) else 1.0
            ax.set_ylim(bottom=0, top=_cap_ylim(y_top))
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

            if stat_df is not None:
                sig = stat_df[
                    (stat_df["metric"] == metric) &
                    (stat_df["comparison_type"] == "hero_vs_others")
                ]
                x_by_model = dict(zip(model_names, x))
                brackets = []
                for _, row in sig.iterrows():
                    mA, mB = row["model_A"], row["model_B"]
                    if mA in x_by_model and mB in x_by_model:
                        brackets.append((x_by_model[mA], x_by_model[mB],
                                         _significance_label(row["p_adj_bh"])))
                if brackets:
                    y_bmax = _draw_brackets(ax, brackets, y_top)
                    if y_bmax > ax.get_ylim()[1]:
                        bracket_overflows.append((ax, y_bmax))

        _expand_margins_for_brackets(fig, bracket_overflows)

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        tag = f"_{suffix}" if suffix else ""
        fname = os.path.join(output_dir, f"compare_{hospital}{tag}.png")
        legend_ncol = min(len(model_handles), 3)
        _finalize(fig, model_handles, [h.get_label() for h in model_handles], legend_ncol, fname)


# ── multi-center bar charts ───────────────────────────────────────────────────

def plot_multi_by_center(
    multi: dict[str, dict[str, dict]],
    hospitals: list[str],
    metrics: list[str],
    model_names: list[str],
    output_dir: str,
    ncols: int = 2,
    stat_dfs: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Figure A: centers on x-axis, one bar per model, colored by MODEL_COLORS.

    Models ordered by group with gap between groups within each center cluster.
    """
    n_models = len(model_names)
    bar_width = 0.7 / max(n_models, 1)

    intra_positions = _group_positions(model_names, gap=0.3)
    intra_positions = intra_positions - intra_positions.mean()
    intra_positions *= bar_width

    n_metrics = len(metrics)
    nrows = math.ceil(n_metrics / ncols)
    n_hospitals = len(hospitals)
    panel_w = max(PANEL_W, n_hospitals * max(1.2, n_models * 0.25))
    model_idx = {m: mi for mi, m in enumerate(model_names)}

    with plt.rc_context(PUB_RC):
        model_handles = [
            mpatches.Patch(facecolor=MODEL_COLORS.get(m, "#999999"), edgecolor="black",
                           linewidth=0.5, label=MODEL_LABELS.get(m, m))
            for m in model_names
        ]

        fig, axes = _make_fig(nrows, ncols, panel_w)
        axes_flat = axes.flatten()

        bracket_overflows: list[tuple[plt.Axes, float]] = []
        for ax, metric in zip(axes_flat, metrics):
            y_top = 0.0
            for hi, hospital in enumerate(hospitals):
                for mi, model in enumerate(model_names):
                    v = multi[hospital].get(model, {}).get(metric)
                    if v is None:
                        continue
                    color = MODEL_COLORS.get(model, "#999999")
                    ax.bar(hi + intra_positions[mi], v["mean"], width=bar_width,
                           yerr=v["std"], capsize=2.5, color=color,
                           edgecolor="black", linewidth=0.5,
                           ecolor="black", error_kw={"linewidth": 0.7})
                    y_top = max(y_top, v["mean"] + v["std"])

            ax.set_xticks(range(n_hospitals))
            ax.set_xticklabels(hospitals)
            ax.set_xlim(-0.7, n_hospitals - 0.3)
            ax.set_ylim(bottom=0, top=_cap_ylim(y_top))
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

            if stat_dfs is not None:
                ax_y_bmax = 0.0
                for hi, hospital in enumerate(hospitals):
                    stat_df = stat_dfs.get(hospital)
                    if stat_df is None:
                        continue
                    sig = stat_df[
                        (stat_df["metric"] == metric) &
                        (stat_df["comparison_type"] == "best_vs_others")
                    ]
                    brackets = []
                    for _, row in sig.iterrows():
                        mA, mB = row["model_A"], row["model_B"]
                        if mA in model_idx and mB in model_idx:
                            x1 = hi + intra_positions[model_idx[mA]]
                            x2 = hi + intra_positions[model_idx[mB]]
                            brackets.append((x1, x2, _significance_label(row["p_adj_bh"])))
                    if brackets:
                        ax_y_bmax = max(ax_y_bmax, _draw_brackets(ax, brackets, y_top))
                if ax_y_bmax > ax.get_ylim()[1]:
                    bracket_overflows.append((ax, ax_y_bmax))

        _expand_margins_for_brackets(fig, bracket_overflows)

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        fname = os.path.join(output_dir, "compare_multicentre_by_center.png")
        legend_ncol = min(len(model_handles), 3)
        _finalize(fig, model_handles, [h.get_label() for h in model_handles], legend_ncol, fname)


def plot_multi_by_model(
    multi: dict[str, dict[str, dict]],
    hospitals: list[str],
    metrics: list[str],
    model_names: list[str],
    output_dir: str,
    ncols: int = 2,
) -> None:
    """Figure B: models on x-axis, one bar per hospital, colored by CENTER_COLORS.

    Models ordered by group with gap between groups.
    """
    x = _group_positions(model_names)
    n_hospitals = len(hospitals)
    bar_width = 0.7 / max(n_hospitals, 1)

    n_metrics = len(metrics)
    nrows = math.ceil(n_metrics / ncols)
    panel_w = max(PANEL_W, len(model_names) * n_hospitals * 0.22)

    with plt.rc_context(PUB_RC):
        center_handles = [
            mpatches.Patch(facecolor=CENTER_COLORS.get(h, "#999999"), edgecolor="black",
                           linewidth=0.5, label=h)
            for h in hospitals
        ]

        fig, axes = _make_fig(nrows, ncols, panel_w)
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, metrics):
            y_top = 0.0
            for ki, hospital in enumerate(hospitals):
                x_offset = (ki - (n_hospitals - 1) / 2) * bar_width
                color = CENTER_COLORS.get(hospital, "#999999")
                for mi, model in enumerate(model_names):
                    v = multi[hospital].get(model, {}).get(metric)
                    if v is None:
                        continue
                    ax.bar(x[mi] + x_offset, v["mean"], width=bar_width,
                           yerr=v["std"], capsize=2.5, color=color,
                           edgecolor="black", linewidth=0.5,
                           ecolor="black", error_kw={"linewidth": 0.7})
                    y_top = max(y_top, v["mean"] + v["std"])

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_names],
                               rotation=30, ha="right")
            ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
            ax.set_ylim(bottom=0, top=_cap_ylim(y_top))
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        fname = os.path.join(output_dir, "compare_multicentre_by_model.png")
        legend_ncol = min(len(center_handles), 5)
        _finalize(fig, center_handles, [h.get_label() for h in center_handles], legend_ncol, fname)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)
    _substitute_user(config)

    output_dir = args.output_dir or get_figures_path(config)
    os.makedirs(output_dir, exist_ok=True)
    metrics = _select_metrics(config, args.metrics)

    results_path = get_results_path(config)

    if len(args.hospital) > 1:
        hospitals = args.hospital
        multi = {h: load_all_results(config, h, args.split, args.external_suffix, args.models)
                 for h in hospitals}
        if not any(multi.values()):
            logger.error("No results loaded for any center. Exiting.")
            sys.exit(1)

        # Union of available models across all centers, sorted by group
        all_model_names: dict[str, None] = {}
        for center_data in multi.values():
            for m in center_data:
                all_model_names[m] = None
        model_names = _sorted_models(list(all_model_names.keys()))

        stat_dfs = None
        if args.add_stat:
            stat_dfs = {h: sd for h in hospitals
                        if (sd := load_stat_results(results_path, h)) is not None}
            if not stat_dfs:
                logger.warning("--add-stat: no statistical test results found for any center.")
                stat_dfs = None

        plot_multi_by_center(multi, hospitals, metrics, model_names, output_dir, args.ncols,
                             stat_dfs=stat_dfs)
        plot_multi_by_model(multi, hospitals, metrics, model_names, output_dir, args.ncols)
        print(f"\nFigures written to: {output_dir}")

    elif args.hero:
        hospital = args.hospital[0]
        model_results = load_hero_results(config, args.models)
        if not model_results:
            logger.error("No hero results loaded. Exiting.")
            sys.exit(1)

        stat_df = None
        if args.add_stat:
            stat_df = load_stat_results(results_path, hospital, hero=True)
            if stat_df is None:
                logger.warning("--add-stat: no hero stat results found for %s.", hospital)

        model_names, used_metrics, rows = build_table(model_results, metrics)
        stem = f"compare_{hospital}_hero"
        md_path      = os.path.join(output_dir, f"{stem}.md")
        summary_path = os.path.join(output_dir, f"{stem}_summary.md")
        write_markdown(model_names, used_metrics, rows, md_path, hospital, "hero sub-cohort")
        write_hero_summary(model_results, used_metrics, summary_path, hospital)
        plot_comparison(model_results, used_metrics, output_dir, hospital, suffix="hero",
                        ncols=args.ncols, stat_df=stat_df)
        print(f"\nOutputs written to:\n  {md_path}\n  {summary_path}\n  {output_dir}/")

    else:
        hospital = args.hospital[0]
        model_results = load_all_results(config, hospital, args.split, args.external_suffix, args.models)
        if not model_results:
            logger.error("No results loaded. Exiting.")
            sys.exit(1)

        model_names, used_metrics, rows = build_table(model_results, metrics)
        md_path = os.path.join(output_dir, f"compare_{hospital}.md")
        write_markdown(model_names, used_metrics, rows, md_path, hospital, args.split)

        stat_df = None
        if args.add_stat:
            stat_df = load_stat_results(results_path, hospital)
            if stat_df is None:
                logger.warning("--add-stat: no stat results found for %s.", hospital)

        plot_comparison(model_results, used_metrics, output_dir, hospital, suffix="",
                        ncols=args.ncols, stat_df=stat_df)
        print(f"\nOutputs written to:\n  {md_path}\n  {output_dir}/")


if __name__ == "__main__":
    main()
