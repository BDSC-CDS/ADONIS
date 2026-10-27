"""Plot OOF calibration curves with Wilson error bars.

Reads the OOF calibration parquets written by
``scripts/machine_learning/training/run_training_tuning.py --do-curves``:

    {results_path}/{model_type}/calibration_oof_bins[_hero|_external_<H>].parquet
    {results_path}/{model_type}/calibration_oof_scores[_hero|_external_<H>].parquet

The bins parquet has one row per (strategy, bin) with prob_true, prob_pred,
Wilson interval, count, positive. The scores parquet has one row per strategy
with brier / ce / ece / relative_ece / symmetric_relative_ece.

Usage
-----
Single-strategy figure:

    python scripts/plotting/plot_calibration.py \\
        --config config/machine_learning/config.json \\
        --strategy quantile

Both strategies side-by-side:

    python scripts/plotting/plot_calibration.py \\
        --config config/machine_learning/config.json \\
        --strategy all

Hero sub-cohort:

    python scripts/plotting/plot_calibration.py \\
        --config config/machine_learning/config.json \\
        --strategy all --hero
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from adonis_ml.machine_learning.plotting import (
    MODEL_COLORS,
    MODEL_LABELS,
    _save_fig,
    _sorted_model_types,
)
from adonis_ml.machine_learning.utils import get_figures_path, get_results_path
from adonis_ml.utils import AdonisConfig, load_and_parse_config

logger = logging.getLogger(__name__)

INTERNAL_HOSPITAL = "CHUV"
STRATEGIES = ("linear", "quantile")


def _parquet_suffix(hospital: str, hero: bool) -> str:
    if hero:
        return "_hero"
    if hospital != INTERNAL_HOSPITAL:
        return f"_external_{hospital}_cv"
    return ""


def _load_model_calibration(
    results_path: str,
    model_type: str,
    suffix: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    model_dir = os.path.join(results_path, model_type)
    bins_path = os.path.join(model_dir, f"calibration_oof_bins{suffix}.parquet")
    scores_path = os.path.join(model_dir, f"calibration_oof_scores{suffix}.parquet")
    probs_path = os.path.join(model_dir, f"calibration_oof_probs{suffix}.parquet")

    if not os.path.exists(bins_path) or not os.path.exists(scores_path):
        logger.warning("Missing OOF calibration parquets for '%s' (%s)", model_type, suffix)
        return None, None, None

    probs_df = pd.read_parquet(probs_path) if os.path.exists(probs_path) else None
    if probs_df is None:
        logger.warning("Missing raw OOF probs for '%s' — histogram panel will be empty", model_type)

    return pd.read_parquet(bins_path), pd.read_parquet(scores_path), probs_df


def _plot_strategy(
    ax,
    strategy: str,
    per_model_bins: dict[str, pd.DataFrame],
    per_model_scores: dict[str, pd.DataFrame],
) -> None:
    all_x = []

    for model_type, bins_df in per_model_bins.items():
        df = bins_df[bins_df["strategy"] == strategy]
        if df.empty:
            continue
        df = df.sort_values("prob_pred")

        prob_pred = df["prob_pred"].to_numpy()
        prob_true = df["prob_true"].to_numpy()
        lower = df["wilson_lower"].to_numpy()
        upper = df["wilson_upper"].to_numpy()

        color = MODEL_COLORS.get(model_type, "#999999")
        label = MODEL_LABELS.get(model_type, model_type)

        ax.errorbar(
            prob_pred, prob_true,
            yerr=[np.maximum(prob_true - lower, 0), np.maximum(upper - prob_true, 0)],
            fmt="none", color=color, alpha=0.35,
        )
        ax.plot(prob_pred, prob_true, marker="o", linestyle="-",
                color=color, label=label, linewidth=1.5, markersize=4)

        all_x.extend(prob_pred.tolist())

    if not all_x:
        ax.set_visible(False)
        return

    # Reference diagonal spanning the data range on this panel
    lo = min(all_x)
    hi = max(all_x)
    if strategy == "quantile":
        lo = max(lo, 1e-6)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"{strategy.capitalize()} bins")
    ax.grid(alpha=0.3, which="both")

    # Error-score annotations
    lines = []
    for model_type, scores_df in per_model_scores.items():
        row = scores_df[scores_df["strategy"] == strategy]
        if row.empty:
            continue
        r = row.iloc[0]
        label = MODEL_LABELS.get(model_type, model_type)
        lines.append(
            f"{label}: Brier={r['brier']:.2e}  CE={r['ce']:.2e}  "
            f"ECE={r['ece']:.2e}  R-ECE={100 * r['symmetric_relative_ece']:.1f}%"
        )
    if lines:
        ax.text(
            0.0, -0.18, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
        )


def _plot_prob_histogram(ax, per_model_probs: dict[str, pd.DataFrame]) -> None:
    positive_min = None
    for probs_df in per_model_probs.values():
        if probs_df is None or probs_df.empty:
            continue
        positive = probs_df["y_prob"].to_numpy()
        positive = positive[positive > 0]
        if positive.size == 0:
            continue
        m = float(positive.min())
        positive_min = m if positive_min is None else min(positive_min, m)

    if positive_min is None:
        ax.set_visible(False)
        return

    bins = np.logspace(np.log10(positive_min), 0, 50)

    for model_type, probs_df in per_model_probs.items():
        if probs_df is None or probs_df.empty:
            continue
        positive = probs_df["y_prob"].to_numpy()
        positive = positive[positive > 0]
        if positive.size == 0:
            continue
        color = MODEL_COLORS.get(model_type, "#999999")
        label = MODEL_LABELS.get(model_type, model_type)
        ax.hist(positive, bins=bins, alpha=0.35, color=color, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Predicted probability (log scale)")
    ax.set_ylabel("Counts")
    ax.set_title("Prediction distribution")
    ax.grid(alpha=0.3, which="both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot OOF calibration curves with Wilson intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types to plot (default: all in config).")
    parser.add_argument("--strategy", default="all",
                        choices=list(STRATEGIES) + ["all"],
                        help="Which binning strategy to plot; 'all' shows both side-by-side. "
                             "Quantile panels are drawn on log-log axes.")
    parser.add_argument("--hero", action="store_true",
                        help="Load *_hero parquets.")
    parser.add_argument("--hospital", default=INTERNAL_HOSPITAL,
                        help="Hospital tag (default: internal).")
    parser.add_argument("--out-suffix", default="",
                        help="Extra suffix appended to output filename.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)

    if config.environment == "BioMedIT":
        user = os.getlogin()
        for k, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[k] = v.replace("$USER", user)

    results_path = get_results_path(config)
    figures_path = get_figures_path(config)

    selected_models = args.models or list(config.model["models"].keys())
    selected_models = _sorted_model_types(selected_models)

    suffix = _parquet_suffix(args.hospital, args.hero)

    per_model_bins: dict[str, pd.DataFrame] = {}
    per_model_scores: dict[str, pd.DataFrame] = {}
    per_model_probs: dict[str, pd.DataFrame | None] = {}
    for model_type in selected_models:
        bins_df, scores_df, probs_df = _load_model_calibration(results_path, model_type, suffix)
        if bins_df is None or scores_df is None:
            continue
        per_model_bins[model_type] = bins_df
        per_model_scores[model_type] = scores_df
        per_model_probs[model_type] = probs_df

    if not per_model_bins:
        logger.error("No calibration parquets found — nothing to plot.")
        return

    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    n_panels = len(strategies) + 1  # +1 for the histogram

    sns.set_theme(style="whitegrid", font_scale=1)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), dpi=150, squeeze=False)
    axes_flat = axes.flatten()

    for ax, strategy in zip(axes_flat, strategies):
        _plot_strategy(ax, strategy, per_model_bins, per_model_scores)

    _plot_prob_histogram(axes_flat[-1], per_model_probs)

    axes_flat[-1].legend(loc="best", fontsize=8, frameon=True)

    tag = args.strategy if args.strategy != "all" else "all"
    out_name = f"calibration_{tag}{suffix}{args.out_suffix}.png"
    out_path = os.path.join(figures_path, out_name)

    fig.tight_layout()
    _save_fig(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
