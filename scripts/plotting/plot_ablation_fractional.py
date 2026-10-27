"""Plot learning-curve metrics from fractional data-ablation results.

For each requested data fraction a line is drawn per model, with error bars
equal to the std across outer folds (from the by_fold aggregation).

Fractions < 1.0 are loaded from  ablation_data_frac_{N}/{model}_sklearn_aggregated.pkl,
produced by  scripts/machine_learning/training/run_training_tuning.py
            --do-fractional --data-fraction <f>.
Fraction 1.0 is loaded from the base  {model}_sklearn.pkl  (--do-cv without --data-fraction).

Usage
-----
    python scripts/plotting/plot_ablation_fractional.py \\
        --config config/machine_learning/config.json \\
        --fractions 0.1 0.25 0.5 1.0

    python scripts/plotting/plot_ablation_fractional.py \\
        --config config/machine_learning/config.json \\
        --fractions 0.1 0.25 0.5 1.0 \\
        --metrics roc_auc auprc recall \\
        --models xgb lr tabpfn \\
        --ncols 3 \\
        --out ablation_fractional.png
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

# ── figure geometry (mirrors plot_models.py) ──────────────────────────────
PANEL_W: float = 3.5
PANEL_H: float = 2.5
_OVERHEAD_TOP: float = 0.80
_XLABEL_IN: float = 0.55                                  # x-label + tick labels
_LEGEND_W: float = 1.10                                   # right-side legend band (inches)
_WSPACE: float = 0.60
_HSPACE: float = 1.4


def _make_fig(nrows: int, ncols: int) -> tuple[plt.Figure, np.ndarray]:
    fig_w = _OVERHEAD_TOP + ncols * PANEL_W + (ncols - 1) * _WSPACE + _LEGEND_W
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
        wspace=_WSPACE / PANEL_W,
        hspace=_HSPACE / PANEL_H,
    )
    return fig, axes


def _finalize(fig, handles, labels, fname):
    fig_w = fig.get_figwidth()
    fig.legend(handles, labels, loc="center right", ncol=1, fontsize=8,
               frameon=False,
               bbox_to_anchor=(1 - 0.10 / fig_w, 0.5))
    _save_fig(fig, fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── radar figure geometry ─────────────────────────────────────────────────────
_RADAR_PANEL_W: float = 4.4
_RADAR_PANEL_H: float = 4.0
_RADAR_WSPACE: float = 1.6
_RADAR_HSPACE: float = 1.4
_RADAR_TOP:    float = 1.00
_RADAR_BOTTOM: float = 0.60


def _make_radar_fig(nrows: int, ncols: int) -> tuple[plt.Figure, np.ndarray]:
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


def _draw_radar_panel(ax, present_models, fractions, data, metric):
    """One radar subplot: spokes = models, overlaid polygons = data fractions.

    At each spoke the highest value's label is drawn radially outside its dot
    and the lowest inside — colored to match its polygon.
    """
    n_models = len(present_models)
    theta = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    polygon_vals: dict[int, np.ndarray] = {}
    for fi, frac in enumerate(fractions):
        vals = [data[frac].get(m, {}).get(metric, {}).get("mean", np.nan)
                for m in present_models]
        polygon_vals[fi] = np.array(vals, dtype=float)

    all_finite = np.array([v for arr in polygon_vals.values() for v in arr if np.isfinite(v)])
    if all_finite.size == 0:
        return
    r_max = float(all_finite.max())
    r_min = float(all_finite.min())
    r_span = max(r_max - r_min, r_max * 0.5, 1e-6)
    r_offset = r_span * 0.22
    r_sep    = r_span * 0.11
    r_top    = r_max + r_offset * 3.0

    for fi in range(len(fractions)):
        color = SUBGROUP_PALETTE[fi % len(SUBGROUP_PALETTE)]
        vals_arr = polygon_vals[fi]
        if np.all(np.isnan(vals_arr)):
            continue
        vals_closed = np.concatenate([vals_arr, vals_arr[:1]])
        ax.plot(theta_closed, vals_closed, color=color, linewidth=1.4,
                marker="o", markersize=3.5, alpha=0.75, zorder=3)
        ax.fill(theta_closed, vals_closed, color=color, alpha=0.08, zorder=2)

    # Stacked labels per spoke — guarantees no overlap even when fractions
    # produce nearly identical metric values at the same model.
    for s in range(n_models):
        spoke_vals = sorted(
            [(fi, polygon_vals[fi][s]) for fi in range(len(fractions))
             if np.isfinite(polygon_vals[fi][s])],
            key=lambda t: t[1],
        )
        if not spoke_vals:
            continue
        n = len(spoke_vals)
        center = float(np.mean([v for _, v in spoke_vals]))
        half = (n - 1) / 2.0
        targets = np.array([
            center + (i - half) * max(r_sep, r_offset) for i in range(n)
        ])
        targets[0]  -= r_offset * 0.6
        targets[-1] += r_offset * 0.6
        targets = np.clip(targets, 0.0, r_top)
        for (fi, val), r_text in zip(spoke_vals, targets):
            color = SUBGROUP_PALETTE[fi % len(SUBGROUP_PALETTE)]
            ax.text(theta[s], r_text, f"{val:.2f}",
                    color=color, fontsize=6.5, ha="center", va="center",
                    fontweight="bold", zorder=5,
                    path_effects=[path_effects.withStroke(linewidth=2.0, foreground="white")])

    ax.set_ylim(0, r_top)
    ax.set_xticks(theta)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present_models])
    ax.tick_params(axis="x", pad=8, labelsize=8)
    ax.tick_params(axis="y", labelsize=6, colors="#555555")
    ax.set_rlabel_position(90 + 180 / max(n_models, 1))
    ax.grid(linestyle="--", alpha=0.35, linewidth=0.5)
    ax.spines["polar"].set_linewidth(0.6)
    ax.spines["polar"].set_color("#888888")


# ── helpers ───────────────────────────────────────────────────────────────────

def _substitute_user(config: AdonisConfig) -> None:
    if getattr(config, "environment", None) == "BioMedIT":
        user = os.getlogin()
        for k, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[k] = v.replace("$USER", user)



def _load_standard_pkl(pkl: str, split: str) -> dict[str, dict] | None:
    """Load {metric: {mean, std}} from a standard _sklearn.pkl."""
    with open(pkl, "rb") as f:
        saved = pickle.load(f)
    results = saved.get("results", {})
    if split in results:
        results = results[split]
    prefix = f"{split}_"
    return {
        (k.removeprefix(prefix) if k.startswith(prefix) else k): v
        for k, v in results.items()
    }


def load_fraction_results(
    results_path: str,
    model: str,
    fraction: float,
    split: str = "test",
) -> dict[str, dict] | None:
    """Return {metric: {"mean": float, "std": float}} for one model/fraction.

    For fraction < 1.0: reads from ablation_data_frac_{N}/{model}_sklearn_aggregated.pkl.
    For fraction == 1.0: reads from the base-path {model}_sklearn.pkl (standard CV results).
    """
    if fraction >= 1.0:
        pkl = os.path.join(results_path, f"{model}_sklearn.pkl")
        if not os.path.exists(pkl):
            logger.warning("100%% pkl not found (run --do-cv without --data-fraction): %s", pkl)
            return None
        return _load_standard_pkl(pkl, split)
    else:
        pct = int(round(fraction * 100))
        pkl = os.path.join(
            results_path, f"ablation_data_frac_{pct}", f"{model}_sklearn_aggregated.pkl"
        )
        if not os.path.exists(pkl):
            logger.warning(
                "Aggregated pkl not found for %.0f%% (run --do-fractional --data-fraction %.2f): %s",
                fraction * 100, fraction, pkl,
            )
            return None
        with open(pkl, "rb") as f:
            saved = pickle.load(f)
        agg = saved.get("aggregated", {})
        if not agg:
            logger.warning("'aggregated' key missing or empty in %s", pkl)
            return None
        # Metric keys in the aggregated pkl carry sklearn's "test_" prefix;
        # strip it so names match the config metrics and the standard pkl.
        prefix = "test_"
        return {
            (metric.removeprefix(prefix) if metric.startswith(prefix) else metric): {
                "mean": v["by_fold"]["mean"],
                "std":  v["by_fold"]["std"],
            }
            for metric, v in agg.items()
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot learning curves from fractional data-ablation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--fractions", nargs="+", type=float, required=True,
                   help="Data fractions to plot, e.g. 0.1 0.25 0.5 1.0")
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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)
    _substitute_user(config)

    results_path = get_results_path(config)
    figures_path = get_figures_path(config)

    model_names = args.models or list(config.model["models"].keys())

    fractions = sorted(set(args.fractions))
    print(f"Results path : {results_path}")
    print(f"Fractions    : {[f'{f*100:.0f}%' for f in fractions]}")

    available_metrics = list(config.metrics.keys())
    metrics = [m for m in (args.metrics or available_metrics) if m in available_metrics]

    # Load data: fraction → model → metric → {mean, std}
    data: dict[float, dict[str, dict]] = {}
    for frac in fractions:
        data[frac] = {}
        for model in model_names:
            res = load_fraction_results(results_path, model, frac, args.split)
            if res is not None:
                data[frac][model] = res

    # Keep only models that have at least one fraction loaded
    present_models = [m for m in model_names if any(m in data[f] for f in fractions)]
    if not present_models:
        sys.exit("No results found for any model/fraction combination.")

    present_metrics = [
        mt for mt in metrics
        if any(mt in data[f].get(m, {}) for f in fractions for m in present_models)
    ]
    if not present_metrics:
        sys.exit("No metrics found in loaded results.")

    # ── build figure ─────────────────────────────────────────────────────────
    ncols = min(args.ncols, len(present_metrics))
    nrows = math.ceil(len(present_metrics) / ncols)
    fig, axes = _make_fig(nrows, ncols)
    axes_flat = axes.flatten()

    ticks_pct = sorted({int(round(f * 100)) for f in fractions})

    for ax, metric in zip(axes_flat, present_metrics):
        ax.set_xscale("log")

        for model in present_models:
            means, stds, xs = [], [], []
            for frac in fractions:
                v = data[frac].get(model, {}).get(metric)
                if v is None:
                    continue
                xs.append(frac * 100)
                means.append(v["mean"])
                stds.append(v["std"])
            if not xs:
                continue
            xs_arr    = np.array(xs)
            means_arr = np.array(means)
            stds_arr  = np.array(stds)
            color = MODEL_COLORS.get(model, "#999999")
            ax.errorbar(
                xs_arr, means_arr, yerr=stds_arr,
                color=color, marker="o", markersize=4,
                linewidth=1.2, capsize=3, elinewidth=0.8, capthick=0.8,
            )

        ax.set_xlim(ticks_pct[0] * 0.7, ticks_pct[-1] * 1.2)
        ax.set_xticks(ticks_pct)
        ax.set_xticklabels([str(t) for t in ticks_pct], fontsize=8)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Training data (%)", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)

    for ax in axes_flat[len(present_metrics):]:
        ax.set_visible(False)

    # Legend: Line2D handles so the glyph matches the actual plot
    handles = [
        plt.Line2D(
            [0], [0],
            color=MODEL_COLORS.get(m, "#999999"),
            marker="o", markersize=4, linewidth=1.2,
            label=MODEL_LABELS.get(m, m),
        )
        for m in present_models
    ]

    out_path = args.out or os.path.join(figures_path, "ablation_fractional.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _finalize(fig, handles, [h.get_label() for h in handles], out_path)
    print(f"Saved → {out_path}")

    # ── figure 2: radar/spider — spokes = models, polygons = data fractions ──
    n_models_r = len(present_models)
    if n_models_r < 3:
        logger.warning("Radar plot needs ≥ 3 models to form a polygon; skipping.")
    else:
        if args.out:
            base, ext = os.path.splitext(out_path)
            ext = ext or ".png"
            out_radar = f"{base}_radar{ext}"
        else:
            out_radar = os.path.join(figures_path, "ablation_fractional_radar.png")
        os.makedirs(os.path.dirname(out_radar) or ".", exist_ok=True)

        with plt.rc_context(PUB_RC):
            fig_r, axes_r = _make_radar_fig(nrows, ncols)
            axes_r_flat = axes_r.flatten()

            for ax, metric in zip(axes_r_flat, present_metrics):
                _draw_radar_panel(ax, present_models, fractions, data, metric)
                ax.set_title(METRIC_LABELS.get(metric, metric), pad=18)

            for ax in axes_r_flat[len(present_metrics):]:
                ax.set_visible(False)

            import matplotlib.patches as mpatches
            handles_r = [
                mpatches.Patch(
                    facecolor=SUBGROUP_PALETTE[fi % len(SUBGROUP_PALETTE)],
                    edgecolor="black", linewidth=0.5, alpha=0.6,
                    label=f"{int(round(frac * 100))}%",
                )
                for fi, frac in enumerate(fractions)
            ]
            _finalize(fig_r, handles_r, [h.get_label() for h in handles_r], out_radar)
            print(f"Saved → {out_radar}")


if __name__ == "__main__":
    main()
