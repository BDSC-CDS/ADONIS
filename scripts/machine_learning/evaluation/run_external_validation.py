"""External validation of ADONIS-ML models on held-out hospitals.

Train the model on CHUV (--do-cv in run_training_tuning.py), then run this
script to apply the saved per-fold models to USZ or KISPI.

Usage examples
--------------
Run CV evaluation on external dataset:

    python scripts/machine_learning/evaluation/run_external_validation.py \\
        --config config/machine_learning/config.json \\
        --hospital USZ \\
        --do-cv --log-level INFO

Feature-group ablation (loads models from ablation_basic_advanced/ subdirectory):

    python scripts/machine_learning/evaluation/run_external_validation.py \\
        --config config/machine_learning/config.json \\
        --hospital USZ \\
        --feature-groups basic advanced \\
        --do-cv --log-level INFO

Fractional ablation (K CHUV subset models, each applied to full external data):

    python scripts/machine_learning/evaluation/run_external_validation.py \\
        --config config/machine_learning/config.json \\
        --hospital USZ \\
        --data-fraction 0.5 --do-fractional \\
        --do-cv --log-level INFO

Re-run subgroup analysis and save curve parquets from saved results (fast — no inference):

    python scripts/machine_learning/evaluation/run_external_validation.py \\
        --config config/machine_learning/config.json \\
        --hospital USZ \\
        --do-subgroups --do-curves

Validate on both external hospitals sequentially:

    for HOSP in USZ KISPI; do
        python scripts/machine_learning/evaluation/run_external_validation.py \\
            --config config/machine_learning/config.json \\
            --hospital $HOSP \\
            --do-cv --do-subgroups --do-curves
    done

CLI flags
---------
--config          Path to JSON config file (required).
--hospital        External hospital to validate on: USZ or KISPI (required).
--gpu             GPU index (default: 0).
--do-cv           Split external dataset into stratified folds and aggregate
                  metrics across fold models per test fold.
--do-subgroups    Run subgroup evaluation (requires --do-cv or a previously
                  saved external pkl). Not supported with --do-fractional.
--do-curves       Save ROC, PRC, DCA, and calibration parquets for plot_curves.py
                  (requires --do-cv or a previously saved external pkl).
                  Not supported with --do-fractional.
--feature-groups  Feature group ablation: loads models from ablation_{tag}/
                  and filters features to "always" + selected groups.
--ablation-tag    Override the ablation subdirectory name.
--do-fractional   Apply each CHUV subset model k to the full external dataset
                  and aggregate metrics across subsets.
--data-fraction   Fraction determining K = round(1/f) subsets (required with
                  --do-fractional, must be < 1.0).
--log-level       Logging verbosity: DEBUG / INFO / WARNING / ERROR (default: INFO).

Input / Output
--------------
Fold models are read from:
    get_results_path(config, ablation_tag) / {model_type}_sklearn.pkl

External dataset is read from:
    paths.path_to_checkpoint / df_cases_controls_{hospital}.parquet

Results are saved with a ``_external_{hospital}_cv`` suffix:
    get_results_path(config, ablation_tag) / {model_type}_sklearn_external_{hospital}_cv.pkl

Fractional aggregated results are saved as:
    get_results_path(config, ablation_tag) / {model_type}_sklearn_external_{hospital}_fractional_agg.pkl

Curve parquets are saved under:
    get_results_path(config, ablation_tag) / {model_type} / {curve}_curve_external_{hospital}.parquet
"""
from __future__ import annotations

import os

# Disable telemetry before any TabPFN import
os.environ["TABPFN_NO_ANALYTICS"] = "true"

import argparse
import logging
import pickle
import sys
from datetime import datetime

import numpy as np

from adonis_ml.machine_learning.data_loader import DataLoader
from adonis_ml.machine_learning.train_tune import AdonisTrainerTuner
from adonis_ml.machine_learning.utils import (
    set_seed,
    setup_cuda,
    get_results_path,
    load_fold_models,
)
from adonis_ml.machine_learning.calibration import compute_calibration_curve
from adonis_ml.machine_learning.plotting import (
    plot_results_subgroups,
    plot_shap_beeswarm,
)
from adonis_ml.utils import AdonisConfig, load_and_parse_config


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"external_validation_{timestamp}.log")
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
        description="External validation of ADONIS-ML models on held-out hospitals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--hospital", required=True, help="External hospital to validate on (e.g. USZ, KISPI).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--do-cv", action="store_true",
                        help="Run CV evaluation on the external dataset.")
    parser.add_argument("--do-subgroups", action="store_true",
                        help="Run subgroup evaluation and generate subgroup plots.")
    parser.add_argument("--do-curves", action="store_true", help="Plot ROC and PRC curves.")
    parser.add_argument("--do-shap", action="store_true",
                        help="Compute SHAP values (fold 0) on external patients and save beeswarm plot.")
    parser.add_argument("--shap-n-explain", type=int, default=None,
                        help="Cap on explained samples for slow models. Default: all.")
    parser.add_argument("--shap-n-background", type=int, default=100,
                        help="Background dataset size for SHAP explainer.")
    parser.add_argument("--feature-groups", nargs="+", default=None,
                        help="Feature group ablation (e.g. basic advanced). Loads models from "
                             "ablation_{tag}/ and filters features to 'always' + selected groups.")
    parser.add_argument("--ablation-tag", default=None,
                        help="Override the ablation subdirectory name.")
    parser.add_argument("--do-fractional", action="store_true",
                        help="Apply each CHUV subset model to the full external dataset and "
                             "aggregate metrics across subsets.")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction determining K=round(1/f) subsets. Must be <1.0 with --do-fractional.")
    parser.add_argument("--cal-n-bins", type=int, default=20,
                        help="Number of bins for OOF calibration curve.")
    parser.add_argument("--cal-alpha", type=float, default=0.05,
                        help="Alpha for Wilson confidence intervals on calibration curve.")
    parser.add_argument("--cal-threshold-region-factor", type=float, default=5.0,
                        help="Multiplier on median threshold for linear calibration bin upper bound.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_models(
    results_path: str,
    model_types: list[str],
    device: str,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Load all fold models and thresholds from a results directory."""
    fit_models: dict = {}
    thresholds: dict = {}
    for model_type in model_types:
        pkl_path = os.path.join(results_path, f"{model_type}_sklearn.pkl")
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(
                f"Trained model pickle not found: {pkl_path}\n"
                f"Run run_training_tuning.py --do-cv first."
            )
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        fit_models[model_type] = load_fold_models(pkl_path, saved, device=device)
        thresholds[model_type] = saved["best_threshold"]
        logger.info(
            "Loaded %d fold models for '%s' from %s",
            len(fit_models[model_type]), model_type, pkl_path,
        )
    return fit_models, thresholds


def _load_external_results(results_path: str, model_types: list[str], suffix: str) -> tuple[dict, dict]:
    results_sklearn: dict = {}
    results_predict: dict = {}
    for model_type in model_types:
        pkl_path = os.path.join(results_path, f"{model_type}_sklearn_external_{suffix}.pkl")
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(
                f"External results not found: {pkl_path}\n"
                f"Run with --do-cv first."
            )
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        results_sklearn[model_type] = saved["results"]
        results_predict[model_type] = saved["results_predict"]
    return results_sklearn, results_predict


def _aggregate_external_subsets(per_subset_results: dict[int, dict], n_subsets: int) -> dict:
    """Mirror AdonisTrainerFractional._aggregate_across_subsets for external results.

    Two views per model & metric:
    - by_fold  : mean across K subsets for each test fold → mean/std across folds.
    - by_subset: mean over test folds for each subset → mean/std across K subsets.
    """
    models = list(per_subset_results[0].keys())
    metrics = list(per_subset_results[0][models[0]]["test"].keys())
    agg = {}
    for model in models:
        agg[model] = {}
        for metric in metrics:
            mat = np.array([
                per_subset_results[k][model]["test"][metric]["values"]
                for k in range(n_subsets)
            ])
            fold_means = mat.mean(axis=0)
            subset_means = mat.mean(axis=1)
            agg[model][metric] = {
                "by_fold": {
                    "values": fold_means.tolist(),
                    "mean":   float(fold_means.mean()),
                    "std":    float(fold_means.std()),
                },
                "by_subset": {
                    "values": subset_means.tolist(),
                    "mean":   float(subset_means.mean()),
                    "std":    float(subset_means.std()),
                },
            }
    return agg


def _save_fractional_external_aggregate(
    aggregated: dict,
    per_subset_results: dict[int, dict],
    n_subsets: int,
    results_path: str,
    hospital: str,
    logger: logging.Logger,
) -> None:
    for model, model_agg in aggregated.items():
        payload = {
            "n_subsets": n_subsets,
            "hospital": hospital,
            "per_subset_metrics": {
                k: per_subset_results[k][model]["test"] for k in range(n_subsets)
            },
            "aggregated": model_agg,
        }
        path = os.path.join(
            results_path, f"{model}_sklearn_external_{hospital}_fractional_agg.pkl"
        )
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Saved fractional aggregate for '%s' → %s", model, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)

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
    logger.info("Hospital:     %s", args.hospital)
    logger.info("GPU index:    %d", args.gpu)
    logger.info("Active flags: %s", active_flags)
    logger.info("Log dir:      %s", log_dir)

    set_seed(config.seed)
    config, device = setup_cuda(config, device_idx=str(args.gpu))

    # ------------------------------------------------------------------
    # Ablation tag / prefixes  (mirrors run_training_tuning.py logic)
    # ------------------------------------------------------------------
    feature_groups = args.feature_groups or []
    ablation_prefixes = ["always"] + feature_groups if feature_groups else None
    ablation_tag = args.ablation_tag or ("_".join(feature_groups) if feature_groups else None)
    data_fraction = args.data_fraction
    do_fractional = args.do_fractional

    if do_fractional and data_fraction >= 1.0:
        raise ValueError("--do-fractional requires --data-fraction < 1.0")

    if data_fraction != 1.0:
        frac_tag = f"data_frac_{int(data_fraction * 100)}"
        ablation_tag = f"{ablation_tag}_{frac_tag}" if ablation_tag else frac_tag

    if ablation_tag:
        logger.info("Ablation tag: %s", ablation_tag)
    if ablation_prefixes:
        logger.info("Ablation prefixes: %s", ablation_prefixes)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    config.hospital = args.hospital

    data_loader = DataLoader(config)
    df, list_of_variables = data_loader.load_adonis_feature_dataset()
    demog = data_loader.load_demog()

    if "hospital" in demog.columns:
        demog = demog[demog["hospital"] == args.hospital].reset_index(drop=True)
        logger.info("Demog filtered to %s — %d rows", args.hospital, len(demog))
    else:
        logger.warning("No 'hospital' column in demog — using full demog table.")

    logger.info("External dataset shape: %s", df.shape)

    results_path = get_results_path(config, ablation_tag=ablation_tag)
    model_types = list(config.model["models"].keys())
    logger.info("Results path: %s", results_path)

    # ------------------------------------------------------------------
    # --do-cv
    # ------------------------------------------------------------------
    results_sklearn: dict = {}
    results_predict: dict = {}
    ATT = None

    if args.do_cv:
        if do_fractional:
            if args.do_subgroups or args.do_curves:
                logger.warning(
                    "--do-subgroups and --do-curves are not supported together with "
                    "--do-fractional; those flags will be skipped."
                )
                args.do_subgroups = False
                args.do_curves = False

            n_subsets = max(2, round(1.0 / data_fraction))
            logger.info("Fractional external validation: K=%d subsets", n_subsets)

            per_subset_results: dict[int, dict] = {}
            for k in range(n_subsets):
                subset_tag = f"{ablation_tag}/subset_{k}" if ablation_tag else f"subset_{k}"
                subset_path = get_results_path(config, ablation_tag=subset_tag)
                logger.info("=== Fractional subset %d/%d — tag=%s ===", k + 1, n_subsets, subset_tag)

                fit_models_sub, thresholds_sub = _load_models(
                    subset_path, model_types, str(args.gpu), logger
                )
                ATT_sub = AdonisTrainerTuner(
                    df, demog, config,
                    ablation_prefixes=ablation_prefixes,
                    ablation_tag=subset_tag,
                )
                r_sklearn, _ = ATT_sub.run_external_cv_validation(
                    fit_models_sub,
                    thresholds_sub,
                    file_appendix=f"_external_{args.hospital}_subset_{k}",
                    do_save_results=True,
                )
                per_subset_results[k] = r_sklearn

            aggregated = _aggregate_external_subsets(per_subset_results, n_subsets)
            _save_fractional_external_aggregate(
                aggregated, per_subset_results, n_subsets, results_path, args.hospital, logger
            )
            # Expose by_fold view as the primary results for any downstream logging
            results_sklearn = {
                model: {
                    "test": {
                        metric: agg_val["by_fold"]
                        for metric, agg_val in aggregated[model].items()
                    }
                }
                for model in aggregated
            }

        else:
            fit_models_sklearn, best_threshold_sklearn = _load_models(
                results_path, model_types, str(args.gpu), logger
            )
            ATT = AdonisTrainerTuner(
                df, demog, config,
                ablation_prefixes=ablation_prefixes,
                ablation_tag=ablation_tag,
            )
            results_sklearn, results_predict = ATT.run_external_cv_validation(
                fit_models_sklearn,
                best_threshold_sklearn,
                file_appendix=f"_external_{args.hospital}_cv",
                do_save_results=True,
            )

    # ------------------------------------------------------------------
    # --do-subgroups / --do-curves without --do-cv
    # ------------------------------------------------------------------
    active_suffix = f"{args.hospital}_cv"

    if (args.do_subgroups or args.do_curves) and not args.do_cv:
        ATT = AdonisTrainerTuner(
            df, demog, config,
            ablation_prefixes=ablation_prefixes,
            ablation_tag=ablation_tag,
        )
        results_sklearn, results_predict = _load_external_results(
            results_path, model_types, active_suffix
        )

    if args.do_subgroups:
        results_subgroups = ATT.run_evaluation_subgroups(results_predict)
        plot_results_subgroups(config, results_subgroups, results_sklearn)

    if args.do_curves:
        import pandas as pd

        results_roc_curve, results_precision_recall_curve, results_dca_curve = ATT.run_evaluation_curve(results_predict)
        curve_suffix = f"_external_{args.hospital}_cv"
        cal_strategies = ("quantile", "linear")

        for model_type in results_roc_curve:
            model_dir = os.path.join(results_path, model_type)
            os.makedirs(model_dir, exist_ok=True)
            results_roc_curve[model_type].to_parquet(
                os.path.join(model_dir, f"roc_curve{curve_suffix}.parquet"), index=False)
            results_precision_recall_curve[model_type].to_parquet(
                os.path.join(model_dir, f"prc_curve{curve_suffix}.parquet"), index=False)
            results_dca_curve[model_type].to_parquet(
                os.path.join(model_dir, f"dca_curve{curve_suffix}.parquet"), index=False)

            # OOF calibration: pool all fold predictions and compute bins with Wilson intervals
            fold_preds = results_predict[model_type]["test"]
            y_true_pool = np.concatenate([np.asarray(fd["y_true"]) for fd in fold_preds.values()])
            y_prob_pool = np.concatenate([np.asarray(fd["y_prob"]) for fd in fold_preds.values()])
            bin_rows, score_rows = [], []
            for strategy in cal_strategies:
                res = compute_calibration_curve(
                    y_true_pool, y_prob_pool,
                    n_bins=args.cal_n_bins,
                    strategy=strategy,
                    thresholds=None,
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
                os.path.join(model_dir, f"calibration_oof_bins{curve_suffix}.parquet"), index=False)
            pd.DataFrame(score_rows).to_parquet(
                os.path.join(model_dir, f"calibration_oof_scores{curve_suffix}.parquet"), index=False)
            pd.DataFrame({
                "y_true": y_true_pool.astype(np.int8),
                "y_prob": y_prob_pool.astype(np.float32),
            }).to_parquet(
                os.path.join(model_dir, f"calibration_oof_probs{curve_suffix}.parquet"), index=False)
            logger.info("Saved curve parquets for '%s' (suffix=%s)", model_type, curve_suffix)

    # ------------------------------------------------------------------
    # --do-shap
    # ------------------------------------------------------------------
    if args.do_shap:
        if do_fractional:
            logger.warning("--do-shap is not supported with --do-fractional; skipping.")
        else:
            fit_models_sklearn_shap: dict = {}
            if not args.do_cv:
                fit_models_sklearn_shap, _ = _load_models(
                    results_path, model_types, str(args.gpu), logger
                )
            else:
                fit_models_sklearn_shap = fit_models_sklearn  # already loaded above

            ATT_shap = AdonisTrainerTuner(
                df, demog, config,
                ablation_prefixes=ablation_prefixes,
                ablation_tag=ablation_tag,
            )
            results_shap = ATT_shap.run_shap_analysis(
                fit_models_sklearn_shap,
                results_predict={},
                n_background=args.shap_n_background,
                n_explain=args.shap_n_explain,
                source="external",
            )
            for model_type, shap_result in results_shap.items():
                plot_shap_beeswarm(config, shap_result, model_type)

    logger.info("External validation complete for hospital: %s", args.hospital)


if __name__ == "__main__":
    main()
