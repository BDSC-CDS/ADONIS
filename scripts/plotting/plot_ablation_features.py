"""Plot feature-group ablation results as grouped bar charts.

One subplot per metric; bars grouped by model type, one bar group per
feature-group ablation tag.  Optionally includes an "all features" baseline
from the standard (non-ablation) per-model pkl.

Usage
-----
    python scripts/plotting/plot_ablation_features.py \\
        --config config/machine_learning/config.json \\
        --groups vitals lab demographic \\
        --group-labels "Vitals" "Lab values" "Demographics"

    python scripts/plotting/plot_ablation_features.py \\
        --config config/machine_learning/config.json \\
        --groups vitals lab \\
        --no-baseline \\
        --metrics roc_auc auprc \\
        --models xgb lr \\
        --ncols 3 \\
        --out ablation_features.png
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from adonis_ml.machine_learning.plotting import (
    MODEL_COLORS, MODEL_LABELS, METRIC_LABELS, SUBGROUP_PALETTE, PUB_RC, _save_fig,
)
from adonis_ml.machine_learning.utils import get_results_path, get_figures_path
from adonis_ml.utils import load_and_parse_config, AdonisConfig

logger = logging.getLogger(__name__)

# ── figure geometry ────────────────────────────────────────────────────────────
PANEL_W: float = 3.5
PANEL_H: float = 2.5
_OVERHEAD_TOP: float = 0.80
_XLABEL_IN: float = 0.75                                  # rotated x-tick labels
_LEGEND_W: float = 1.10                                   # right-side legend band (inches)
_WSPACE: float = 0.60
_HSPACE: float = 1.4


def _make_fig(nrows: int, ncols: int, panel_w: float = PANEL_W) -> tuple[plt.Figure, np.ndarray]:
    fig_w = _OVERHEAD_TOP + ncols * panel_w + (ncols - 1) * _WSPACE + _LEGEND_W
    fig_h = _OVERHEAD_TOP + nrows * PANEL_H + (nrows - 1) * _HSPACE + _XLABEL_IN
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        dpi=300, squeeze=False,
    )
    fig.subplots_adjust(
        left=_OVERHEAD_TOP / fig_w,
        right=1 - _LEGEND_W / fig_w,
        bottom=_XLABEL_IN / fig_h,
        top=1 - _OVERHEAD_TOP / fig_h,
        wspace=_WSPACE / panel_w,
        hspace=_HSPACE / PANEL_H,
    )
    return fig, axes


# Radar panels are square-ish and need extra breathing room around the perimeter
# so angled model tick labels don't collide with the title, the legend band,
# or neighboring subplots.
_RADAR_PANEL_W: float = 4.4
_RADAR_PANEL_H: float = 4.0
_RADAR_WSPACE: float = 1.6
_RADAR_HSPACE: float = 1.4
_RADAR_TOP:    float = 1.00
_RADAR_BOTTOM: float = 0.60


def _make_radar_fig(nrows: int, ncols: int) -> tuple[plt.Figure, np.ndarray]:
    """Polar subplots sized for a paper figure.

    Panels are deliberately square-ish and generously spaced so each spoke
    label stays within its own cell and the radial value annotations don't
    collide with the title, the legend band, or neighboring panels.
    """
    fig_w = _OVERHEAD_TOP + ncols * _RADAR_PANEL_W + (ncols - 1) * _RADAR_WSPACE + _LEGEND_W
    fig_h = _RADAR_TOP + nrows * _RADAR_PANEL_H + (nrows - 1) * _RADAR_HSPACE + _RADAR_BOTTOM
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        dpi=300, squeeze=False,
        subplot_kw={"projection": "polar"},
    )
    fig.subplots_adjust(
        left=_OVERHEAD_TOP / fig_w,
        right=1 - _LEGEND_W / fig_w,
        bottom=_RADAR_BOTTOM / fig_h,
        top=1 - _RADAR_TOP / fig_h,
        wspace=_RADAR_WSPACE / _RADAR_PANEL_W,
        hspace=_RADAR_HSPACE / _RADAR_PANEL_H,
    )
    return fig, axes


def _draw_radar_panel(ax, present_models, group_tags, group_displs, data, metric):
    """Draw one radar subplot: spokes = models, overlaid polygons = groups.

    At every spoke, the dot with the higher value gets its numeric label
    plotted radially outside the point; the lower value's label is plotted
    radially inside — same color as the polygon so the pairing is obvious.
    """
    n_models = len(present_models)
    theta = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    # Collect per-polygon values across spokes
    polygon_vals: dict[int, np.ndarray] = {}
    for gi, tag in enumerate(group_tags):
        vals = [data[tag].get(m, {}).get(metric, {}).get("mean", np.nan)
                for m in present_models]
        polygon_vals[gi] = np.array(vals, dtype=float)

    all_finite = np.array([v for arr in polygon_vals.values() for v in arr if np.isfinite(v)])
    if all_finite.size == 0:
        return
    r_max = float(all_finite.max())
    r_min = float(all_finite.min())
    r_span = max(r_max - r_min, r_max * 0.5, 1e-6)
    r_offset = r_span * 0.22                      # base label clearance from the polygon line
    r_sep    = r_span * 0.11                      # min separation between stacked labels at a spoke
    r_top    = r_max + r_offset * 3.0

    # Polygons + markers
    for gi in range(len(group_tags)):
        color = SUBGROUP_PALETTE[gi % len(SUBGROUP_PALETTE)]
        vals_arr = polygon_vals[gi]
        if np.all(np.isnan(vals_arr)):
            continue
        vals_closed = np.concatenate([vals_arr, vals_arr[:1]])
        ax.plot(theta_closed, vals_closed, color=color, linewidth=1.4,
                marker="o", markersize=3.5, alpha=0.75, zorder=3)
        ax.fill(theta_closed, vals_closed, color=color, alpha=0.08, zorder=2)

    # Value annotations: at each spoke, sort by value and place labels in
    # evenly-spaced radial slots so close-together polygons never overlap.
    # Highest value → outermost slot, lowest → innermost — preserving the
    # "high outside / low inside the polygon" semantic even when the dots
    # themselves are nearly on top of each other.
    for s in range(n_models):
        spoke_vals = sorted(
            [(gi, polygon_vals[gi][s]) for gi in range(len(group_tags))
             if np.isfinite(polygon_vals[gi][s])],
            key=lambda t: t[1],
        )
        if not spoke_vals:
            continue
        n = len(spoke_vals)
        center = float(np.mean([v for _, v in spoke_vals]))
        # Half-range in radial units; include +offset so top label clears the
        # outer polygon line and bottom label dips below the inner one.
        half = (n - 1) / 2.0
        targets = np.array([
            center + (i - half) * max(r_sep, r_offset) for i in range(n)
        ])
        # Push the extreme labels further out / in by r_offset for clear separation
        targets[0]  -= r_offset * 0.6
        targets[-1] += r_offset * 0.6
        # Clamp to axes range
        targets = np.clip(targets, 0.0, r_top)
        for (gi, val), r_text in zip(spoke_vals, targets):
            color = SUBGROUP_PALETTE[gi % len(SUBGROUP_PALETTE)]
            ax.text(theta[s], r_text, f"{val:.2f}",
                    color=color, fontsize=6.5, ha="center", va="center",
                    fontweight="bold", zorder=5,
                    path_effects=[path_effects.withStroke(linewidth=2.0, foreground="white")])

    ax.set_ylim(0, r_top)
    ax.set_xticks(theta)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present_models])
    ax.tick_params(axis="x", pad=8, labelsize=8)
    ax.tick_params(axis="y", labelsize=6, colors="#555555")
    ax.set_rlabel_position(90 + 180 / max(n_models, 1))   # tuck radial labels between first two spokes
    ax.grid(linestyle="--", alpha=0.35, linewidth=0.5)
    ax.spines["polar"].set_linewidth(0.6)
    ax.spines["polar"].set_color("#888888")


def _finalize(fig, handles, labels, fname):
    fig_w = fig.get_figwidth()
    fig.legend(handles, labels, loc="center right", ncol=1, fontsize=8,
               frameon=False,
               bbox_to_anchor=(1 - 0.10 / fig_w, 0.5))
    _save_fig(fig, fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── helpers ────────────────────────────────────────────────────────────────────

def _substitute_user(config: AdonisConfig) -> None:
    if getattr(config, "environment", None) == "BioMedIT":
        user = os.getlogin()
        for k, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[k] = v.replace("$USER", user)


def load_ablation_group_results(
    results_path: str,
    model: str,
    tag: str,
    split: str = "test",
) -> dict[str, dict] | None:
    """Load {metric: {mean, std}} from a feature-group ablation pkl."""
    pkl = os.path.join(results_path, f"ablation_{tag}", f"{model}_sklearn.pkl")
    if not os.path.exists(pkl):
        logger.warning("Ablation pkl not found: %s", pkl)
        return None
    with open(pkl, "rb") as f:
        saved = pickle.load(f)
    results = saved.get("results", {}).get(split, saved.get("results", {}))
    prefix = f"{split}_"
    out = {}
    for k, v in results.items():
        metric = k.removeprefix(prefix) if k.startswith(prefix) else k
        values = v.get("values")
        has_values = values is not None and len(values) > 0
        mean = v.get("mean", float(np.mean(values)) if has_values else float("nan"))
        std = float(np.std(values)) if has_values else v.get("std", 0.0)
        out[metric] = {"mean": mean, "std": std}
    return out


def load_baseline_results(
    results_path: str,
    model: str,
    split: str = "test",
) -> dict[str, dict] | None:
    """Load {metric: {mean, std}} from the standard (all-features) pkl."""
    pkl = os.path.join(results_path, f"{model}_sklearn.pkl")
    if not os.path.exists(pkl):
        logger.warning("Standard pkl not found: %s", pkl)
        return None
    with open(pkl, "rb") as f:
        saved = pickle.load(f)
    results = saved.get("results", {}).get(split, saved.get("results", {}))
    prefix = f"{split}_"
    return {
        (k.removeprefix(prefix) if k.startswith(prefix) else k): v
        for k, v in results.items()
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot feature-group ablation results as grouped bar charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--groups", nargs="+", required=True,
                   help="Ablation tags to include (e.g. vitals lab demographic).")
    p.add_argument("--group-labels", nargs="+", default=None,
                   help="Display labels for each group (same order as --groups).")
    p.add_argument("--no-baseline", action="store_true",
                   help="Omit the 'all features' baseline bar.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Metrics to plot (default: all from config).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model types to include (default: all from config).")
    p.add_argument("--split", default="test", choices=["test", "vali"])
    p.add_argument("--ncols", type=int, default=2,
                   help="Number of subplot columns.")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: figures dir).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)
    _substitute_user(config)

    results_path = get_results_path(config)
    figures_path = get_figures_path(config)

    model_names = args.models or list(config.model["models"].keys())
    available_metrics = list(config.metrics.keys())
    metrics = [m for m in (args.metrics or available_metrics) if m in available_metrics]

    # Build ordered list of (tag, display_label)
    group_labels = args.group_labels or args.groups
    if len(group_labels) != len(args.groups):
        sys.exit("--group-labels must have the same number of entries as --groups")
    groups: list[tuple[str, str]] = list(zip(args.groups, group_labels))

    # Optionally append baseline ("all features") as the rightmost group.
    _BASELINE_TAG = "__baseline__"
    if not args.no_baseline:
        groups = groups + [(_BASELINE_TAG, "Full")]

    # Load data: tag → model → metric → {mean, std}
    data: dict[str, dict[str, dict]] = {}
    for tag, _label in groups:
        data[tag] = {}
        for model in model_names:
            if tag == _BASELINE_TAG:
                res = load_baseline_results(results_path, model, args.split)
            else:
                res = load_ablation_group_results(results_path, model, tag, args.split)
            if res is not None:
                data[tag][model] = res

    present_models = [m for m in model_names if any(m in data[t] for t, _ in groups)]
    if not present_models:
        sys.exit("No results found for any model/group combination.")

    present_metrics = [
        mt for mt in metrics
        if any(mt in data[t].get(m, {}) for t, _ in groups for m in present_models)
    ]
    if not present_metrics:
        sys.exit("No metrics found in loaded results.")

    # ── shared layout params ────────────────────────────────────────────────
    n_groups  = len(groups)
    n_models  = len(present_models)
    ncols     = min(args.ncols, len(present_metrics))
    nrows     = math.ceil(len(present_metrics) / ncols)

    group_tags   = [t for t, _ in groups]
    group_displs = [l for _, l in groups]

    # Output paths: derive triplet outputs (by-group + by-model + radar) from optional --out.
    if args.out:
        base, ext = os.path.splitext(args.out)
        ext = ext or ".png"
        out_by_group = f"{base}_by_group{ext}"
        out_by_model = f"{base}_by_model{ext}"
        out_radar    = f"{base}_radar{ext}"
    else:
        out_by_group = os.path.join(figures_path, "ablation_features_by_group.png")
        out_by_model = os.path.join(figures_path, "ablation_features_by_model.png")
        out_radar    = os.path.join(figures_path, "ablation_features_radar.png")
    for p in (out_by_group, out_by_model, out_radar):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    # ── figure 1: x = feature group, bars per model (MODEL_COLORS) ──────────
    bar_width_g = 0.7 / max(n_models, 1)
    panel_w_g   = max(PANEL_W, n_groups * max(1.2, n_models * 0.28))
    fig_g, axes_g = _make_fig(nrows, ncols, panel_w_g)
    axes_g_flat   = axes_g.flatten()
    x_centers_g   = np.arange(n_groups, dtype=float)
    offsets_g     = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width_g

    for ax, metric in zip(axes_g_flat, present_metrics):
        y_top = 0.0
        for mi, model in enumerate(present_models):
            color = MODEL_COLORS.get(model, "#999999")
            for gi, tag in enumerate(group_tags):
                v = data[tag].get(model, {}).get(metric)
                if v is None:
                    continue
                xpos = x_centers_g[gi] + offsets_g[mi]
                ax.bar(
                    xpos, v["mean"],
                    width=bar_width_g,
                    yerr=v["std"],
                    color=color,
                    capsize=2,
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                    ecolor="black",
                    error_kw={"linewidth": 0.6},
                )
                y_top = max(y_top, v["mean"] + v["std"])

        ax.set_xticks(x_centers_g)
        ax.set_xticklabels(group_displs, fontsize=7)
        ax.set_xlim(x_centers_g[0] - 0.6, x_centers_g[-1] + 0.6)
        ax.set_ylim(bottom=0, top=y_top * 1.08 if y_top > 0 else 1.0)
        ax.set_xlabel("Feature group", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

    for ax in axes_g_flat[len(present_metrics):]:
        ax.set_visible(False)

    handles_g = [
        mpatches.Patch(
            facecolor=MODEL_COLORS.get(m, "#999999"),
            edgecolor="black", linewidth=0.5,
            label=MODEL_LABELS.get(m, m),
        )
        for m in present_models
    ]
    _finalize(fig_g, handles_g, [h.get_label() for h in handles_g], out_by_group)
    print(f"Saved → {out_by_group}")

    # ── figure 2: x = model, bars per feature group (SUBGROUP_PALETTE) ──────
    bar_width_m = 0.8 / max(n_groups, 1)
    panel_w_m   = max(PANEL_W, n_models * max(1.2, n_groups * 0.28))
    fig_m, axes_m = _make_fig(nrows, ncols, panel_w_m)
    axes_m_flat   = axes_m.flatten()
    x_centers_m   = np.arange(n_models, dtype=float)
    offsets_m     = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_width_m

    for ax, metric in zip(axes_m_flat, present_metrics):
        y_top = 0.0
        for gi, tag in enumerate(group_tags):
            color = SUBGROUP_PALETTE[gi % len(SUBGROUP_PALETTE)]
            for mi, model in enumerate(present_models):
                v = data[tag].get(model, {}).get(metric)
                if v is None:
                    continue
                xpos = x_centers_m[mi] + offsets_m[gi]
                ax.bar(
                    xpos, v["mean"],
                    width=bar_width_m,
                    yerr=v["std"],
                    color=color,
                    capsize=2,
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                    ecolor="black",
                    error_kw={"linewidth": 0.6},
                )
                y_top = max(y_top, v["mean"] + v["std"])

        ax.set_xticks(x_centers_m)
        ax.set_xticklabels(
            [MODEL_LABELS.get(m, m) for m in present_models],
            rotation=25, ha="right", fontsize=7,
        )
        ax.set_xlim(x_centers_m[0] - 0.6, x_centers_m[-1] + 0.6)
        ax.set_ylim(bottom=0, top=y_top * 1.08 if y_top > 0 else 1.0)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

    for ax in axes_m_flat[len(present_metrics):]:
        ax.set_visible(False)

    handles_m = [
        mpatches.Patch(
            facecolor=SUBGROUP_PALETTE[gi % len(SUBGROUP_PALETTE)],
            edgecolor="black", linewidth=0.5,
            label=label,
        )
        for gi, label in enumerate(group_displs)
    ]
    _finalize(fig_m, handles_m, [h.get_label() for h in handles_m], out_by_model)
    print(f"Saved → {out_by_model}")

    # ── figure 3: radar/spider plot — spokes = models, polygons = feature groups ──
    if n_models < 3:
        logger.warning("Radar plot needs ≥ 3 models to form a polygon; skipping.")
    else:
        with plt.rc_context(PUB_RC):
            fig_r, axes_r = _make_radar_fig(nrows, ncols)
            axes_r_flat = axes_r.flatten()

            for ax, metric in zip(axes_r_flat, present_metrics):
                _draw_radar_panel(ax, present_models, group_tags, group_displs, data, metric)
                ax.set_title(METRIC_LABELS.get(metric, metric), pad=18)

            for ax in axes_r_flat[len(present_metrics):]:
                ax.set_visible(False)

            handles_r = [
                mpatches.Patch(
                    facecolor=SUBGROUP_PALETTE[gi % len(SUBGROUP_PALETTE)],
                    edgecolor="black", linewidth=0.5, alpha=0.6,
                    label=label,
                )
                for gi, label in enumerate(group_displs)
            ]
            _finalize(fig_r, handles_r, [h.get_label() for h in handles_r], out_radar)
            print(f"Saved → {out_radar}")


if __name__ == "__main__":
    main()
