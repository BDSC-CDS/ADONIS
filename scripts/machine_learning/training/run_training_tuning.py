"""Train and tune ADONIS-ML models via nested cross-validation.

Usage examples
--------------
Run cross-validation:

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --do-cv

Run on GPU 0 with CV + subgroups + curves:

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --gpu 0 --do-cv --do-subgroups --do-curves --log-level DEBUG

Fractional data (learning-curve) analysis — partition the training set into
K = round(1/data_fraction) disjoint subsets and run nested CV per subset:

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --do-fractional --data-fraction 0.5

Feature-group ablation — train on 'always' + the listed groups:

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --do-cv --feature-groups demographics basic

CLI flags
---------
--config              Path to JSON config file (required).
--gpu                 GPU index (default: 0).
--do-cv               Run nested cross-validation.
--do-subgroups        Run subgroup evaluation (requires saved results).
--do-curves           Plot ROC and PRC curves (requires saved results).
--do-fractional       Run fractional data (learning-curve) analysis
                      (requires --data-fraction < 1.0).
--data-fraction       Fraction of training groups to use (0 < f <= 1.0).
--feature-groups      Feature-group ablation: train on 'always' + selected
                      groups (e.g. demographics basic).
--do-metrics-over-time  Compute metrics per relative window hour and save parquets.
--do-shap             Compute SHAP values and save plots (requires saved results).
                      Uses the fold-0 model; explains the fold-0 test set for
                      internal CV, or the full dataset for external validation.
                      Dispatches to the fastest available explainer per model:
                        TreeExplainer   → xgb, rf  (seconds)
                        LinearExplainer → lr        (seconds)
                        PermutationExplainer → all others (tabpfn, tabicl, …)
                      Outputs per model (under figures_path/{model_type}/):
                        shap_beeswarm.png          — top-N feature beeswarm
                        shap_scatter/{rank}_{feat}.png — scatter + marginals per feature
                      Saved SHAP results: results_path/{model_type}_shap.pkl
--shap-n-explain      Cap on explained samples. Recommended for slow models:
                        tabpfn / tabicl / svm: 200–500
                        xgb / rf / lr:         no cap needed (leave unset)
--shap-n-background   Background dataset size for the SHAP explainer (default: 100).
--log-level           Logging verbosity: DEBUG / INFO / WARNING / ERROR (default: INFO).

SHAP quick-start
----------------
Step 1 — train and save models (once):

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --do-cv

Step 2 — compute SHAP (fast models):

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --do-shap --gpu 0

Step 2 — compute SHAP (slow models such as TabPFN):

    python scripts/machine_learning/training/run_training_tuning.py \\
        --config config/machine_learning/config.json \\
        --do-shap --gpu 0 --shap-n-explain 300 --shap-n-background 100

Log file
--------
If ``paths.logs`` is set in the config, a timestamped log file
``training_YYYYMMDD_HHMMSS.log`` is written to that directory in addition
to stdout.
"""
from __future__ import annotations

import os

# Disable telemetry before any TabPFN import
os.environ["TABPFN_NO_ANALYTICS"] = "true"

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd

from adonis_ml.machine_learning.data_loader import DataLoader
from adonis_ml.machine_learning.train_tune import (
    AdonisTrainerTuner,
    AdonisTrainerFractional,
)
from adonis_ml.machine_learning.utils import set_seed, setup_cuda, get_results_path, get_figures_path, load_fold_models, filter_features_by_groups
from adonis_ml.machine_learning.plotting import plot_shap_beeswarm, plot_shap_scatter
from adonis_ml.machine_learning.calibration import compute_calibration_curve
from adonis_ml.utils import AdonisConfig, load_and_parse_config


def wrap_single_shot_as_aggregate(results_path: str, model_types, split: str = "test") -> None:
    """Write single-shot CV results in the aggregated format read by
    ``plot_ablation_fractional.py``.

    Reads ``{model}_sklearn.pkl`` in ``results_path`` and writes a sibling
    ``{model}_sklearn_aggregated.pkl`` whose ``aggregated[metric]["by_fold"]``
    carries the CV fold statistics.
    """
    logger = logging.getLogger(__name__)
    for model in model_types:
        src = os.path.join(results_path, f"{model}_sklearn.pkl")
        dst = os.path.join(results_path, f"{model}_sklearn_aggregated.pkl")
        if not os.path.exists(src):
            logger.warning("wrap_single_shot: missing %s — skipped", src)
            continue
        with open(src, "rb") as fh:
            saved = pickle.load(fh)
        results = saved.get("results", {})
        test = results.get(split, results)
        aggregated = {}
        for metric, v in test.items():
            values = v.get("values", [])
            mean = v.get("mean", float("nan"))
            std = v.get("std", 0.0)
            aggregated[metric] = {
                "by_fold":   {"values": list(values), "mean": float(mean), "std": float(std)},
                "by_subset": {"values": [float(mean)], "mean": float(mean), "std": 0.0},
            }
        payload = {
            "n_subsets": 1,
            "per_subset_metrics": {0: test},
            "aggregated": aggregated,
        }
        with open(dst, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Wrote single-shot aggregate → %s", dst)


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=handlers,
    )
    logging.getLogger("posthog").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and tune ADONIS-ML models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--do-cv", action="store_true", help="Run nested cross-validation.")
    parser.add_argument("--do-subgroups", action="store_true", help="Run subgroup evaluation.")
    parser.add_argument("--do-curves", action="store_true", help="Plot ROC and PRC curves.")
    parser.add_argument(
        "--do-fractional", action="store_true",
        help="Cover the training set with K=round(1/data_fraction) disjoint group subsets, "
             "run full nested CV per subset, and aggregate metrics across subsets. "
             "Requires --data-fraction < 1.0.",
    )
    parser.add_argument("--do-metrics-over-time", action="store_true",
                        help="Compute metrics per relative window hour and save parquets.")
    parser.add_argument("--do-forewarning-time", action="store_true",
                        help="Compute forewarning time per sepsis patient per fold and save parquets.")
    parser.add_argument("--forewarning-time", type=int, default=72,
                        help="Lookback window in hours for the fw-time streak search "
                             "(default: 72). Independent of the labelling horizon; "
                             "encoded in the output filename.")
    parser.add_argument(
        "--feature-groups", nargs="+", default=None,
        metavar="GROUP",
        help="Feature group ablation: train on 'always' + selected groups. "
             "Available: basic, advanced, ventilation. "
             "Results saved under ablation_<tag>/ inside the main results dir.",
    )
    parser.add_argument(
        "--ablation-tag", default=None,
        help="Override the ablation subdirectory name (default: '_'.join(feature_groups)).",
    )
    parser.add_argument(
        "--data-fraction", type=float, default=1.0,
        help="Fraction of training groups to use (0 < f <= 1.0). Test set is always 100%%. "
             "Without --do-fractional: a single random subsample is drawn (legacy behavior). "
             "With --do-fractional: the training set is partitioned into K=round(1/f) "
             "disjoint group subsets, each yielding one full nested-CV run.",
    )
    parser.add_argument("--do-shap", action="store_true", help="Compute SHAP values (fold 0) and save beeswarm plot.")
    parser.add_argument("--shap-n-explain", type=int, default=None, help="Cap on explained samples for slow models (PermutationExplainer). Default: all.")
    parser.add_argument("--shap-n-background", type=int, default=100, help="Background dataset size for SHAP explainer.")
    parser.add_argument("--cal-n-bins", type=int, default=20,
                        help="Number of bins for OOF calibration curves (--do-curves).")
    parser.add_argument("--cal-alpha", type=float, default=0.05,
                        help="Significance level for Wilson intervals on OOF calibration curves.")
    parser.add_argument("--cal-threshold-region-factor", type=float, default=5.0,
                        help="Linear-strategy zoom factor around median threshold.")
    parser.add_argument("--threshold-max", type=float, default=None,
                        help="Upper bound for threshold tuning grid (e.g. 0.025). "
                             "If set, uses np.linspace(0, threshold-max, threshold-n) "
                             "instead of the sklearn default linspace(0, 1, threshold-n).")
    parser.add_argument("--threshold-n", type=int, default=1000,
                        help="Number of candidate thresholds in the tuning grid (default: 1000).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)

    # $USER substitution for all string paths
    if config.environment == "BioMedIT":
        user = os.getlogin()
        for k, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[k] = v.replace("$USER", user)
        config.mlflow["path_to_checkpoint"] = config.mlflow["path_to_checkpoint"].replace(
            "$USER", user
        )

    log_dir = config.paths.get("logs")
    setup_logging(level=args.log_level, log_dir=log_dir)
    logger = logging.getLogger(__name__)

    active_flags = [k for k, v in vars(args).items() if k.startswith("do_") and v]
    logger.info("Config:       %s", args.config)
    logger.info("GPU index:    %d", args.gpu)
    logger.info("Active flags: %s", active_flags)
    logger.info("Log dir:      %s", log_dir)

    config.params["threshold_max"] = args.threshold_max
    config.params["threshold_n"]   = args.threshold_n

    set_seed(config.seed)
    config, device = setup_cuda(config, device_idx=str(args.gpu))

    data_loader = DataLoader(config)
    df, list_of_variables = data_loader.load_adonis_feature_dataset()
    demog = data_loader.load_demog()

    logger.info("Dataset shape:   %s", df.shape)
    logger.info("Output path:     %s", config.paths.get("output_path"))

    # --- Ablation feature filtering ---
    ablation_prefixes: list[str] | None = None
    ablation_tag: str | None = None
    if args.feature_groups:
        group_defs = getattr(config, "ablation_variable_groups", {})
        unknown = [g for g in args.feature_groups if g not in group_defs and g != "always"]
        if unknown:
            logger.warning("Unknown feature groups (ignored): %s", unknown)
        ablation_prefixes = list(group_defs.get("always", []))
        for g in args.feature_groups:
            ablation_prefixes.extend(group_defs.get(g, []))
        ablation_tag = args.ablation_tag or "_".join(args.feature_groups)
        logger.info("Ablation groups: %s  →  tag: %s  |  prefixes: %s",
                    args.feature_groups, ablation_tag, ablation_prefixes)

    n_fractional_subsets = 1
    single_shot_fraction = False
    if args.data_fraction != 1.0:
        frac_tag = f"data_frac_{int(args.data_fraction * 100)}"
        ablation_tag = f"{ablation_tag}_{frac_tag}" if ablation_tag else frac_tag
        if args.do_fractional:
            n_fractional_subsets = max(2, int(round(1.0 / args.data_fraction)))
            logger.info(
                "Fractional mode: fraction=%.2f  →  %d disjoint subsets  →  tag=%s",
                args.data_fraction, n_fractional_subsets, ablation_tag,
            )
        else:
            config.model["undersampler"].set_params(data_fraction=args.data_fraction)
            single_shot_fraction = True
            logger.info(
                "Data fraction (random, single-shot): %.2f  →  tag=%s  "
                "(will wrap output as *_sklearn_aggregated.pkl for plotter compatibility)",
                args.data_fraction, ablation_tag,
            )
    elif args.do_fractional:
        raise SystemExit("--do-fractional requires --data-fraction < 1.0")

    do_cv               = args.do_cv
    do_evaluate_subgroups = args.do_subgroups
    do_curves           = args.do_curves
    do_fractional_data  = args.do_fractional

    if do_cv:
        ATT = AdonisTrainerTuner(
            df,
            demog,
            config,
            ablation_prefixes=ablation_prefixes,
            ablation_tag=ablation_tag,
        )
        fit_models_sklearn, results_sklearn, results_predict, best_params_sklearn, best_threshold_sklearn = ATT.run_nested_cross_validation_sklearn(
            selected_models=None,
        )
        if single_shot_fraction:
            wrap_single_shot_as_aggregate(
                get_results_path(config, ablation_tag),
                list(config.model["models"].keys()),
            )

    if do_fractional_data:
        ATT = AdonisTrainerFractional(
            df,
            demog,
            config,
            ablation_prefixes=ablation_prefixes,
            ablation_tag=ablation_tag,
        )
        ATT.run_nested_cv_fractional(
            n_subsets=n_fractional_subsets,
            selected_models=None,
            base_ablation_tag=ablation_tag,
        )

    if do_evaluate_subgroups or do_curves or args.do_metrics_over_time or args.do_forewarning_time:
        ATT = AdonisTrainerTuner(
            df,
            demog,
            config,
            ablation_prefixes=ablation_prefixes,
            ablation_tag=ablation_tag,
        )

        results_sklearn = {}
        results_predict = {}
        best_threshold_sklearn = {}
        for model_type, model in config.model["models"].items():
            with open(os.path.join(get_results_path(config, ablation_tag), f"{model_type}_sklearn.pkl"), "rb") as f:
                results = pickle.load(f)

            results_sklearn[model_type] = results["results"]
            results_predict[model_type] = results["results_predict"]
            best_threshold_sklearn[model_type] = results.get("best_threshold")

    if do_evaluate_subgroups:
        ATT.run_evaluation_subgroups(results_predict)

    if args.do_forewarning_time:
        ATT.run_forewarning_time(results_predict, forewarning_time=args.forewarning_time)

    if args.do_metrics_over_time:
        for model_type in config.model["models"]:
            ATT.compute_and_save_metrics_per_window_hour(model_type, results_predict[model_type])

    if do_curves:
        results_roc_curve, results_precision_recall_curve, results_dca_curve = ATT.run_evaluation_curve(results_predict)
        results_path = get_results_path(config, ablation_tag)

        cal_strategies = ("quantile", "linear")

        def _write_oof_calibration(model_dir, fold_predictions, thresholds, suffix):
            import numpy as np
            y_true = np.concatenate([np.asarray(fd["y_true"]) for fd in fold_predictions.values()])
            y_prob = np.concatenate([np.asarray(fd["y_prob"]) for fd in fold_predictions.values()])
            bin_rows, score_rows = [], []
            for strategy in cal_strategies:
                res = compute_calibration_curve(
                    y_true, y_prob,
                    n_bins=args.cal_n_bins,
                    strategy=strategy,
                    thresholds=thresholds if strategy == "linear" else None,
                    threshold_region_factor=args.cal_threshold_region_factor,
                    alpha=args.cal_alpha,
                )
                for i in range(len(res["prob_pred"])):
                    bin_rows.append({
                        "strategy":     strategy,
                        "bin_idx":      i,
                        "prob_pred":    float(res["prob_pred"][i]),
                        "prob_true":    float(res["prob_true"][i]),
                        "wilson_lower": float(res["wilson_lower"][i]),
                        "wilson_upper": float(res["wilson_upper"][i]),
                        "count":        int(res["counts"][i]),
                        "positive":     int(res["positives"][i]),
                    })
                score_rows.append({"strategy": strategy, **res["error_scores"]})
            pd.DataFrame(bin_rows).to_parquet(
                os.path.join(model_dir, f"calibration_oof_bins{suffix}.parquet"), index=False
            )
            pd.DataFrame(score_rows).to_parquet(
                os.path.join(model_dir, f"calibration_oof_scores{suffix}.parquet"), index=False
            )
            pd.DataFrame({"y_true": y_true.astype(np.int8), "y_prob": y_prob.astype(np.float32)}).to_parquet(
                os.path.join(model_dir, f"calibration_oof_probs{suffix}.parquet"), index=False
            )

        for model_type in results_roc_curve:
            model_dir = os.path.join(results_path, model_type)
            os.makedirs(model_dir, exist_ok=True)
            results_roc_curve[model_type].to_parquet(os.path.join(model_dir, "roc_curve.parquet"), index=False)
            results_precision_recall_curve[model_type].to_parquet(os.path.join(model_dir, "prc_curve.parquet"), index=False)
            results_dca_curve[model_type].to_parquet(os.path.join(model_dir, "dca_curve.parquet"), index=False)

            # OOF calibration parquets (consumed by plot_curves.py and plot_calibration.py)
            thresholds_model = best_threshold_sklearn.get(model_type)
            _write_oof_calibration(model_dir, results_predict[model_type]["test"], thresholds_model, suffix="")
            logger.info("Saved curve parquets for model '%s' → %s", model_type, model_dir)

        # Hero sub-cohort curves (runs automatically when hero_max data is available)
        if ATT.hero_max is not None:
            results_predict_hero = ATT.get_hero_predict_dict(results_predict, apply_prior_shift=True)
            if results_predict_hero:
                roc_hero, prc_hero, dca_hero = ATT.run_evaluation_curve(results_predict_hero, max_threshold=0.2)
                for model_type in roc_hero:
                    model_dir = os.path.join(results_path, model_type)
                    os.makedirs(model_dir, exist_ok=True)
                    roc_hero[model_type].to_parquet(os.path.join(model_dir, "roc_curve_hero.parquet"), index=False)
                    prc_hero[model_type].to_parquet(os.path.join(model_dir, "prc_curve_hero.parquet"), index=False)
                    dca_hero[model_type].to_parquet(os.path.join(model_dir, "dca_curve_hero.parquet"), index=False)

                    thresholds_model = best_threshold_sklearn.get(model_type)
                    _write_oof_calibration(
                        model_dir, results_predict_hero[model_type]["test"], thresholds_model, suffix="_hero"
                    )
                    logger.info("Saved hero curve parquets for '%s' → %s", model_type, model_dir)

    if args.do_shap:
        results_path = get_results_path(config, ablation_tag)
        results_shap = {}

        for model_type in config.model["models"]:
            shap_pkl = os.path.join(results_path, f"{model_type}_shap.pkl")
            if os.path.exists(shap_pkl):
                logger.info("Loading cached SHAP results for '%s' from %s", model_type, shap_pkl)
                with open(shap_pkl, "rb") as f:
                    results_shap[model_type] = pickle.load(f)
            else:
                logger.info("No cached SHAP results for '%s', computing now", model_type)
                sklearn_pkl = os.path.join(results_path, f"{model_type}_sklearn.pkl")
                with open(sklearn_pkl, "rb") as f:
                    saved = pickle.load(f)
                fit_models_shap = {model_type: load_fold_models(sklearn_pkl, saved, device=str(args.gpu))}
                results_predict_shap = {model_type: saved["results_predict"]}

                ATT_shap = AdonisTrainerTuner(df, demog, config, ablation_prefixes=ablation_prefixes, ablation_tag=ablation_tag)
                results_shap.update(ATT_shap.run_shap_analysis(
                    fit_models_shap,
                    results_predict_shap,
                    n_background=args.shap_n_background,
                    n_explain=args.shap_n_explain,
                    source="internal",
                ))

        for model_type, shap_result in results_shap.items():
            plot_shap_beeswarm(config, shap_result, model_type)
            plot_shap_scatter(config, shap_result, model_type)


if __name__ == "__main__":
    main()
