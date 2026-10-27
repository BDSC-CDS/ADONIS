"""Plot ROC, PRC, DCA, and calibration curves from pre-computed parquet files.

Usage
-----
Internal validation, all four curves in a 2-column grid:

    python scripts/plotting/plot_curves.py \\
        --config config/machine_learning/config.json

Select curve types and columns:

    python scripts/plotting/plot_curves.py \\
        --config config/machine_learning/config.json \\
        --curves roc prc dca \\
        --ncols 3

Hero sub-cohort:

    python scripts/plotting/plot_curves.py \\
        --config config/machine_learning/config.json \\
        --hero

Single external hospital:

    python scripts/plotting/plot_curves.py \\
        --config config/machine_learning/config.json \\
        --hospital USZ

Multi-hospital comparison (two output figures):

    python scripts/plotting/plot_curves.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV USZ KISPI

Parquet files are read from:
    {results_path}/{model_type}/roc_curve[_hero|_external_{hospital}].parquet
    {results_path}/{model_type}/prc_curve[...].parquet
    {results_path}/{model_type}/dca_curve[...].parquet
    {results_path}/{model_type}/calibration_oof_bins[...].parquet
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
from scipy.interpolate import interp1d

from adonis_ml.machine_learning.plotting import (
    CENTER_COLORS,
    CENTER_LINESTYLES,
    MODEL_COLORS,
    MODEL_GROUPS,
    MODEL_LABELS,
    PUB_RC,
    _GROUP_LINESTYLES,
    _save_fig,
    _sorted_model_types,
    plot_calibration_curve,
    plot_curve,
    plot_dca,
    style_axes,
)
from adonis_ml.machine_learning.utils import get_figures_path, get_results_path
from adonis_ml.utils import AdonisConfig, load_and_parse_config

logger = logging.getLogger(__name__)

CURVE_CHOICES = ["roc", "prc", "dca", "dca_or", "cal"]
INTERNAL_HOSPITAL = "CHUV"


# ── helpers ───────────────────────────────────────────────────────────────────

def _parquet_suffix(hospital: str, hero: bool) -> str:
    if hero:
        return "_hero"
    if hospital != INTERNAL_HOSPITAL:
        return f"_external_{hospital}_cv"
    return ""


def _pkl_appendix(hospital: str, file_appendix_override: str) -> str:
    if file_appendix_override:
        return file_appendix_override
    if hospital != INTERNAL_HOSPITAL:
        return f"_external_{hospital}_cv"
    return ""


def _load_hospital(
    results_path: str,
    hospital: str,
    selected_models: list[str],
    curves: list[str],
    hero: bool,
    file_appendix_override: str,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict]:
    """Load parquets + pkl results for one hospital.

    Returns (curve_data, results_sklearn) where
    curve_data[curve_key][model_type] = DataFrame.
    """
    pq_suffix = _parquet_suffix(hospital, hero)
    appendix  = _pkl_appendix(hospital, file_appendix_override)

    curve_data: dict[str, dict[str, pd.DataFrame]] = {c: {} for c in CURVE_CHOICES}
    results_sklearn: dict = {}

    for model_type in selected_models:
        model_dir = os.path.join(results_path, model_type)

        if model_type != "hero_predictor":
            pkl_path = os.path.join(results_path, f"{model_type}_sklearn{appendix}.pkl")
            if not os.path.exists(pkl_path):
                logger.warning("Pickle not found for '%s' / '%s': %s", hospital, model_type, pkl_path)
                continue
            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            if hero:
                hero_ml = saved.get("results_hero_ml", {})
                results_sklearn[model_type] = {
                    "test": {f"test_{k}_hero": v for k, v in hero_ml.items()}
                }
                if "results_hero_score" in saved and "hero_predictor" not in results_sklearn:
                    results_sklearn["hero_predictor"] = {
                        "test": {f"test_{k}_hero": v for k, v in saved["results_hero_score"].items()}
                    }
            else:
                results_sklearn[model_type] = saved["results"]

        fname_map = {
            "roc": f"roc_curve{pq_suffix}.parquet",
            "prc": f"prc_curve{pq_suffix}.parquet",
            "dca": f"dca_curve{pq_suffix}.parquet",
            "cal": f"calibration_oof_bins{pq_suffix}.parquet",
        }
        for curve_key, fname in fname_map.items():
            if curve_key not in curves and not (curve_key == "dca" and "dca_or" in curves):
                continue
            path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                curve_data[curve_key][model_type] = pd.read_parquet(path)
                logger.info("Loaded %s for %s / %s", fname, hospital, model_type)
            else:
                logger.warning("Parquet not found: %s", path)

    # dca_or is a second panel on the same data — share the loaded dca dict
    if "dca_or" in curves:
        curve_data["dca_or"] = curve_data["dca"]

    return curve_data, results_sklearn


def _mean_curve(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_x, mean_y) interpolated across folds."""
    mean_x = np.linspace(0, 1, 100)
    ys = []
    for fold in df["fold"].unique():
        fold_df = df[df["fold"] == fold].sort_values(x_col)
        f = interp1d(fold_df[x_col].values, fold_df[y_col].values,
                     kind="linear", bounds_error=False, fill_value=(0, 1))
        ys.append(f(mean_x))
    return mean_x, np.array(ys).mean(axis=0)


def _dca_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (thresholds, net_benefit, prevalence) from a pooled DCA parquet."""
    df = df.sort_values("threshold")
    thresholds = df["threshold"].to_numpy()
    nb = df["net_benefit"].to_numpy()
    prevalence = _prevalence_from_df(df)
    return thresholds, nb, prevalence


def _prevalence_from_df(df: pd.DataFrame) -> float | None:
    if "prevalence" in df.columns and not df["prevalence"].empty:
        return float(df["prevalence"].iloc[0])
    min_t = df["threshold"].min()
    vals = df.loc[df["threshold"] == min_t, "treat_all"]
    if not vals.empty:
        return float(vals.mean())
    return None


def _prevalence_from_dca(dca_data: dict[str, pd.DataFrame]) -> float | None:
    """First non-empty DCA dataframe wins; all share the same pooled test prevalence."""
    for df in dca_data.values():
        if df is None or df.empty:
            continue
        prev = _prevalence_from_df(df)
        if prev is not None:
            return prev
    return None


def _model_linestyle(model_type: str) -> str:
    """Return the fixed group linestyle for a model."""
    group = MODEL_GROUPS.get(model_type, "baselines")
    return _GROUP_LINESTYLES.get(group, "-")


def _build_panel_specs(
    curves: list[str],
    data_by_hosp: dict[str, dict[str, dict[str, pd.DataFrame]]],
) -> list[tuple[str, str]]:
    """Build ordered list of (spec_name, curve_type) for subplot grid.

    "dca"    → probability x-axis panel  (spec_name="dca_prob", curve_type="dca")
    "dca_or" → OR-rescaled x-axis panel  (spec_name="dca_or",   curve_type="dca")
    """
    specs: list[tuple[str, str]] = []
    for curve in curves:
        # dca and dca_or both check the "dca" data bucket
        data_key = "dca" if curve in ("dca", "dca_or") else curve
        has_data = any(data_by_hosp[h][data_key] for h in data_by_hosp)
        if not has_data:
            continue
        if curve == "dca":
            specs.append(("dca_prob", "dca"))
        elif curve == "dca_or":
            specs.append(("dca_or", "dca"))
        else:
            specs.append((curve, curve))
    return specs


# ── single-hospital figure ────────────────────────────────────────────────────

def _plot_single(
    config,
    curve_data: dict[str, dict],
    results_sklearn: dict,
    panel_specs: list[tuple[str, str]],
    metric_suffix: str,
    dca_ylim: tuple,
    ncols: int,
    out_path: str,
) -> None:
    n_panels = len(panel_specs)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                                 dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        dca_axes: dict[str, object] = {}
        for idx, (spec_name, curve_type) in enumerate(panel_specs):
            ax = axes_flat[idx]
            if curve_type == "dca":
                dca_axes[spec_name] = ax
            elif curve_type == "roc":
                plot_curve(config, curve_data["roc"], x_col="fpr", y_col="tpr",
                           df_results=results_sklearn, filename="", ref_line=True,
                           metric_suffix=metric_suffix, ax=ax)
            elif curve_type == "prc":
                plot_curve(config, curve_data["prc"], x_col="recall", y_col="precision",
                           df_results=results_sklearn, filename="", ref_line=False,
                           metric_suffix=metric_suffix, ax=ax)
            elif curve_type == "cal":
                plot_calibration_curve(config, curve_data["cal"], filename="", ax=ax)

        if curve_data["dca"] and dca_axes:
            plot_dca(config, curve_data["dca"], filename="", ylim=dca_ylim,
                     ax_prob=dca_axes.get("dca_prob"), ax_or=dca_axes.get("dca_or"))

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        # Normalise delegate-drawn axes: strip grid, hide top/right spines.
        for ax in axes_flat[:n_panels]:
            style_axes(ax)

        fig.tight_layout()
        _save_fig(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ── multi-hospital figures ────────────────────────────────────────────────────

def _plot_multi_by_hospital(
    hospitals: list[str],
    model_names: list[str],
    data_by_hosp: dict[str, dict[str, dict[str, pd.DataFrame]]],
    panel_specs: list[tuple[str, str]],
    dca_ylim: tuple,
    ncols: int,
    out_path: str,
) -> None:
    """Figure 1: color = hospital, linestyle = model."""
    n_panels = len(panel_specs)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                                 dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        for idx, (spec_name, curve_type) in enumerate(panel_specs):
            ax = axes_flat[idx]

            for hospital in hospitals:
                color = CENTER_COLORS.get(hospital, "#999999")

                for model_type in model_names:
                    linestyle = _model_linestyle(model_type)

                    if curve_type in ("roc", "prc"):
                        x_col = "fpr"   if curve_type == "roc" else "recall"
                        y_col = "tpr"   if curve_type == "roc" else "precision"
                        df = data_by_hosp[hospital][curve_type].get(model_type)
                        if df is None or df.empty:
                            continue
                        mean_x, mean_y = _mean_curve(df, x_col, y_col)
                        ax.plot(mean_x, mean_y, color=color, linestyle=linestyle, linewidth=1.2)
                        if curve_type == "roc":
                            ax.set_xlabel("False positive rate")
                            ax.set_ylabel("True positive rate")
                        else:
                            ax.set_xlabel("Recall")
                            ax.set_ylabel("Precision")

                    elif curve_type == "cal":
                        df = data_by_hosp[hospital]["cal"].get(model_type)
                        if df is None or df.empty:
                            continue
                        df = df[df["strategy"] == "quantile"].sort_values("prob_pred")
                        if df.empty:
                            continue
                        ax.plot(df["prob_pred"], df["prob_true"], marker="o",
                                color=color, linestyle=linestyle, linewidth=1.2, markersize=3)
                        ax.set_xlabel("Mean predicted probability")
                        ax.set_ylabel("Observed frequency")

                    elif curve_type == "dca":
                        df = data_by_hosp[hospital]["dca"].get(model_type)
                        if df is None or df.empty:
                            continue
                        thresholds, mean_nb, prevalence = _dca_curve(df)
                        if spec_name == "dca_prob":
                            ax.plot(thresholds, mean_nb, color=color, linestyle=linestyle, linewidth=1.2)
                            ax.set_xlabel("Threshold probability")
                            ax.set_ylabel("Net benefit")
                            ax.set_ylim(*dca_ylim)
                            pt_max = 0.05
                            if prevalence and 0 < prevalence < 1:
                                prior_odds = prevalence / (1 - prevalence)
                                pt_max = (7 * prior_odds) / (1 + 7 * prior_odds)
                            ax.set_xlim(0, pt_max)
                        else:  # dca_or
                            if prevalence and prevalence > 0:
                                prior_odds = prevalence / (1 - prevalence)
                                or_x = (thresholds / (1 - thresholds)) / prior_odds
                                ax.plot(or_x, mean_nb, color=color, linestyle=linestyle, linewidth=1.2)
                            ax.set_xlabel("Odds ratio relative to prior")
                            ax.set_ylabel("Net benefit")
                            ax.set_xlim(0, 7)
                            ax.set_ylim(*dca_ylim)

            # Reference lines
            if curve_type == "roc":
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
            elif curve_type == "cal":
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(left=1e-4)
                ax.set_ylim(bottom=1e-4)
                xl, xr = ax.get_xlim()
                ax.plot([xl, xr], [xl, xr], "k--", linewidth=0.7)
            elif curve_type == "dca":
                ax.axhline(0, color="black", linewidth=0.7, linestyle=":")

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        for ax in axes_flat[:n_panels]:
            style_axes(ax)

        hosp_handles = [
            mpatches.Patch(facecolor=CENTER_COLORS.get(h, "#999999"), edgecolor="black",
                           linewidth=0.5, label=h)
            for h in hospitals
        ]
        model_handles = [
            plt.Line2D([0], [0], color="black",
                       linestyle=_model_linestyle(mt),
                       linewidth=1.2, label=MODEL_LABELS.get(mt, mt))
            for mt in model_names
        ]
        all_handles = hosp_handles + model_handles
        fig.legend(all_handles, [h.get_label() for h in all_handles],
                   loc="lower center", ncol=min(len(all_handles), 6),
                   frameon=False, bbox_to_anchor=(0.5, 0))
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        _save_fig(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_multi_by_model(
    hospitals: list[str],
    model_names: list[str],
    data_by_hosp: dict[str, dict[str, dict[str, pd.DataFrame]]],
    panel_specs: list[tuple[str, str]],
    dca_ylim: tuple,
    ncols: int,
    out_path: str,
) -> None:
    """Figure 2: color = model, linestyle = hospital."""
    n_panels = len(panel_specs)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                                 dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        for idx, (spec_name, curve_type) in enumerate(panel_specs):
            ax = axes_flat[idx]

            for model_type in model_names:
                color = MODEL_COLORS.get(model_type, "#999999")

                for hospital in hospitals:
                    linestyle = CENTER_LINESTYLES.get(hospital, "-")

                    if curve_type in ("roc", "prc"):
                        x_col = "fpr"   if curve_type == "roc" else "recall"
                        y_col = "tpr"   if curve_type == "roc" else "precision"
                        df = data_by_hosp[hospital][curve_type].get(model_type)
                        if df is None or df.empty:
                            continue
                        mean_x, mean_y = _mean_curve(df, x_col, y_col)
                        ax.plot(mean_x, mean_y, color=color, linestyle=linestyle, linewidth=1.2)
                        if curve_type == "roc":
                            ax.set_xlabel("False positive rate")
                            ax.set_ylabel("True positive rate")
                        else:
                            ax.set_xlabel("Recall")
                            ax.set_ylabel("Precision")

                    elif curve_type == "cal":
                        df = data_by_hosp[hospital]["cal"].get(model_type)
                        if df is None or df.empty:
                            continue
                        df = df[df["strategy"] == "quantile"].sort_values("prob_pred")
                        if df.empty:
                            continue
                        ax.plot(df["prob_pred"], df["prob_true"], marker="o",
                                color=color, linestyle=linestyle, linewidth=1.2, markersize=3)
                        ax.set_xlabel("Mean predicted probability")
                        ax.set_ylabel("Observed frequency")

                    elif curve_type == "dca":
                        df = data_by_hosp[hospital]["dca"].get(model_type)
                        if df is None or df.empty:
                            continue
                        thresholds, mean_nb, prevalence = _dca_curve(df)
                        if spec_name == "dca_prob":
                            ax.plot(thresholds, mean_nb, color=color, linestyle=linestyle, linewidth=1.2)
                            ax.set_xlabel("Threshold probability")
                            ax.set_ylabel("Net benefit")
                            ax.set_ylim(*dca_ylim)
                            pt_max = 0.05
                            if prevalence and 0 < prevalence < 1:
                                prior_odds = prevalence / (1 - prevalence)
                                pt_max = (7 * prior_odds) / (1 + 7 * prior_odds)
                            ax.set_xlim(0, pt_max)
                        else:
                            if prevalence and prevalence > 0:
                                prior_odds = prevalence / (1 - prevalence)
                                or_x = (thresholds / (1 - thresholds)) / prior_odds
                                ax.plot(or_x, mean_nb, color=color, linestyle=linestyle, linewidth=1.2)
                            ax.set_xlabel("Odds ratio relative to prior")
                            ax.set_ylabel("Net benefit")
                            ax.set_xlim(0, 7)
                            ax.set_ylim(*dca_ylim)

            if curve_type == "roc":
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
            elif curve_type == "cal":
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(left=1e-4)
                ax.set_ylim(bottom=1e-4)
                xl, xr = ax.get_xlim()
                ax.plot([xl, xr], [xl, xr], "k--", linewidth=0.7)
            elif curve_type == "dca":
                ax.axhline(0, color="black", linewidth=0.7, linestyle=":")

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        for ax in axes_flat[:n_panels]:
            style_axes(ax)

        model_handles = [
            mpatches.Patch(facecolor=MODEL_COLORS.get(mt, "#999999"), edgecolor="black",
                           linewidth=0.5, label=MODEL_LABELS.get(mt, mt))
            for mt in model_names
        ]
        hosp_handles = [
            plt.Line2D([0], [0], color="black",
                       linestyle=CENTER_LINESTYLES.get(h, "-"),
                       linewidth=1.2, label=h)
            for h in hospitals
        ]
        all_handles = model_handles + hosp_handles
        fig.legend(all_handles, [h.get_label() for h in all_handles],
                   loc="lower center", ncol=min(len(all_handles), 6),
                   frameon=False, bbox_to_anchor=(0.5, 0))
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        _save_fig(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot curve figures from saved parquets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types to plot (default: all in config).")
    parser.add_argument("--curves", nargs="+", default=CURVE_CHOICES,
                        choices=CURVE_CHOICES, metavar="CURVE",
                        help="Curve types: roc prc dca dca_or cal (default: all).")
    parser.add_argument("--ncols", type=int, default=2,
                        help="Number of subplot columns (default: 2).")
    parser.add_argument("--file-appendix", default="",
                        help="Override pkl suffix, e.g. '_external_USZ_cv'.")
    parser.add_argument("--hero", action="store_true",
                        help="Plot curves for the hero sub-cohort.")
    parser.add_argument("--hospital", nargs="+", default=None,
                        help="Hospital(s): CHUV USZ KISPI. "
                             "One → single-hospital figure. "
                             "Multiple → two multi-hospital comparison figures.")
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
    if args.hero:
        selected_models = selected_models + ["hero_predictor"]
    selected_models = _sorted_model_types(selected_models)

    hospitals = args.hospital or [INTERNAL_HOSPITAL]

    # ── multi-hospital mode ────────────────────────────────────────────────
    if len(hospitals) > 1:
        data_by_hosp: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        for h in hospitals:
            cd, _ = _load_hospital(
                results_path, h, selected_models, args.curves, args.hero, args.file_appendix
            )
            data_by_hosp[h] = cd

        panel_specs = _build_panel_specs(args.curves, data_by_hosp)
        if not panel_specs:
            logger.error("No data loaded for any requested curve type.")
            return

        prevalence = next(
            (_prevalence_from_dca(data_by_hosp[h]["dca"]) for h in hospitals
             if _prevalence_from_dca(data_by_hosp[h]["dca"]) is not None),
            None,
        )
        dca_y_top = (prevalence * 1.1) if prevalence is not None else 0.06
        dca_ylim = (-0.001, dca_y_top)
        tag = "_hero" if args.hero else ""

        _plot_multi_by_hospital(
            hospitals, selected_models, data_by_hosp, panel_specs, dca_ylim, args.ncols,
            out_path=os.path.join(figures_path, f"curves_multihosp_by_hospital{tag}.png"),
        )
        _plot_multi_by_model(
            hospitals, selected_models, data_by_hosp, panel_specs, dca_ylim, args.ncols,
            out_path=os.path.join(figures_path, f"curves_multihosp_by_model{tag}.png"),
        )
        return

    # ── single-hospital mode ───────────────────────────────────────────────
    hospital = hospitals[0]
    curve_data, results_sklearn = _load_hospital(
        results_path, hospital, selected_models, args.curves, args.hero, args.file_appendix
    )

    if not results_sklearn:
        logger.error("No models loaded — nothing to plot.")
        return

    metric_suffix = "_hero" if args.hero else ""
    prevalence = _prevalence_from_dca(curve_data.get("dca", {}))
    dca_y_top = (prevalence * 1.1) if prevalence is not None else 0.06
    dca_ylim = (-0.001, dca_y_top)

    panel_specs = _build_panel_specs(args.curves, {hospital: curve_data})
    if not panel_specs:
        logger.error("No data loaded for any requested curve type.")
        return

    pq_tag = _parquet_suffix(hospital, args.hero)
    out_path = os.path.join(figures_path, f"curves{pq_tag}.png")

    _plot_single(
        config, curve_data, results_sklearn, panel_specs,
        metric_suffix, dca_ylim, args.ncols, out_path,
    )


if __name__ == "__main__":
    main()
