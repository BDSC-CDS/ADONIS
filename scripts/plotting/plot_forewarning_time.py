"""Plot the threshold-swept PPV vs forewarning-time curve.

For each model, one curve is drawn by sweeping the decision threshold over
all distinct OOF probabilities. x = coverage-weighted mean forewarning time
(hours) across sepsis patients, y = PPV over all windows (sepsis + controls).

The curve is precomputed by
scripts/machine_learning/training/run_training_tuning.py --do-forewarning-time
and saved as forewarning_curve_{H}h[_hero].parquet under each model directory.

Usage
-----
    python scripts/plotting/plot_forewarning_time.py \\
        --config config/machine_learning/config.json

    # HeRO sub-cohort
    python scripts/plotting/plot_forewarning_time.py \\
        --config config/machine_learning/config.json --hero
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adonis_ml.machine_learning.utils import get_results_path, get_figures_path
from adonis_ml.machine_learning.plotting import (
    MODEL_LABELS, MODEL_COLORS, PUB_RC, _save_fig, style_axes,
)
from adonis_ml.utils import load_and_parse_config, AdonisConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PPV vs forewarning-time curve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--hero", action="store_true", help="Use hero sub-cohort parquet.")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model keys to plot (default: all models in config).",
    )
    parser.add_argument(
        "--forewarning-time", type=int, default=72,
        help="Lookback window in hours (must match extraction).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _load_curves(
    results_path: str, model_types: list[str], hero: bool, forewarning_time: int,
) -> dict[str, pd.DataFrame]:
    hero_suffix = "_hero" if hero else ""
    fw_tag = f"{forewarning_time}h"
    curves: dict[str, pd.DataFrame] = {}
    for mt in model_types:
        path = os.path.join(results_path, mt, f"forewarning_curve_{fw_tag}{hero_suffix}.parquet")
        if not os.path.isfile(path):
            logger.warning("Forewarning curve not found for model '%s': %s", mt, path)
            continue
        curves[mt] = pd.read_parquet(path)
    if not curves:
        raise FileNotFoundError(
            f"No forewarning_curve_{fw_tag}{hero_suffix} parquets found. "
            f"Run: python scripts/run_training_tuning.py --config ... "
            f"--do-forewarning-time --forewarning-time {forewarning_time}"
        )
    return curves


def plot_curve(
    curves: dict[str, pd.DataFrame],
    figures_path: str,
    hero: bool,
    forewarning_time: int,
) -> None:
    with plt.rc_context(PUB_RC):
        # fig_w compensates for the right-side legend band reserved by
        # tight_layout(rect=...) so the drawn axes match a 4.2×3.4" panel.
        fig, ax = plt.subplots(figsize=(4.2 / 0.78, 3.4), dpi=300)

        max_fw = 0.0
        prevalences: list[float] = []
        for mt, curve in curves.items():
            df = curve.sort_values("fw_time", kind="stable")
            ax.plot(
                df["fw_time"].values,
                df["ppv"].values,
                color=MODEL_COLORS.get(mt, "#999999"),
                linestyle="-",
                linewidth=2,
                label=MODEL_LABELS.get(mt, mt),
            )
            max_fw = max(max_fw, float(np.nanmax(df["fw_time"].values)))
            if "prevalence" in df.columns:
                p = float(df["prevalence"].iloc[0])
                if np.isfinite(p):
                    prevalences.append(p)

        if prevalences:
            prev = float(np.mean(prevalences))
            ax.axhline(
                prev, linestyle="--", color="black", linewidth=1
            )

        x_upper = min(forewarning_time, max_fw * 1.05) if max_fw > 0 else forewarning_time
        ax.set_xlim(0.02, x_upper)
        ax.set_ylim(0, 0.25)
        ax.set_xlabel("Forewarning time (hours)")
        ax.set_ylabel("PPV")
        style_axes(ax)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center right", ncol=1,
            frameon=False, bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=[0, 0, 0.78, 1])

        suffix = "_hero" if hero else ""
        fw_tag = f"{forewarning_time}h"
        out = os.path.join(figures_path, f"forewarning_curve_{fw_tag}{suffix}.png")
        _save_fig(fig, out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved forewarning curve → %s", out)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
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
        config.mlflow["path_to_checkpoint"] = config.mlflow["path_to_checkpoint"].replace("$USER", user)

    results_path = get_results_path(config)
    figures_path = get_figures_path(config)
    model_types = list(config.model["models"].keys())

    if args.models is not None:
        unknown = [m for m in args.models if m not in model_types]
        if unknown:
            logger.warning("Unknown model(s) ignored: %s", unknown)
        model_types = [m for m in args.models if m in model_types]

    if args.hero and "hero_predictor" not in model_types:
        model_types.append("hero_predictor")

    curves = _load_curves(results_path, model_types, args.hero, args.forewarning_time)
    plot_curve(curves, figures_path, args.hero, args.forewarning_time)


if __name__ == "__main__":
    main()
