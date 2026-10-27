import logging
import os
import pickle
import random
import re

import numpy as np
import torch

from imblearn.pipeline import Pipeline

from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model
from tabicl import TabICLClassifier

from adonis_ml.machine_learning.parallel import ParallelEstimatorWrapper

logger = logging.getLogger(__name__)


def set_seed(_seed=123):
    os.environ["PYTHONHASHSEED"] = str(_seed)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    logger.info("Seed set to %d", _seed)


def setup_cuda(config, device_idx="3"):
    # Select the GPU(s) you want to use
    os.environ["CUDA_VISIBLE_DEVICES"] = device_idx

    config.use_gpu = config.use_gpu and torch.cuda.is_available()

    if config.use_gpu:
        device = torch.device("cuda:{}".format(config.gpu))
        logger.info("Use GPU: cuda:%s", config.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
        logger.info("Use CPU")

    return config, device


def print_results_summary(results):
    def _dict_depth(dic, level=1):
        if not isinstance(dic, dict) or not dic:
            return level
        return max(_dict_depth(dic[key], level + 1) for key in dic)

    def _print_results_summary(results):
        print(
            ", ".join(
                [
                    f'{metric}: {result["mean"]:.3f}({result["std"]:.3f})'
                    for metric, result in results.items()
                ]
            )
        )

    depth_of_results_dict = _dict_depth(results)
    match depth_of_results_dict:
        case 2:
            print(
                ", ".join(
                    [
                        f'{metric}: {result:.3f}'
                        for metric, result in results.items()
                    ]
                )
            )
        case 3:
            _print_results_summary(results)
        case 4:
            for key, result in results.items():
                print(key)
                _print_results_summary(result)
        case _:
            raise ValueError(
                "Results dict passed to 'print_results_summary' function has too many layers"
            )


def get_results_path(config, ablation_tag: str | None = None):
    sampling_column = config.model["undersampler"].sampling_column
    results_path = os.path.join(config.paths["output_path"], sampling_column)
    if ablation_tag:
        results_path = os.path.join(results_path, f"ablation_{ablation_tag}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger.info("Results path: %s", results_path)
    return results_path


def get_figures_path(config, ablation_tag: str | None = None):
    sampling_column = config.model["undersampler"].sampling_column
    figures_path = os.path.join(config.paths["figures_path"], sampling_column)
    if ablation_tag:
        figures_path = os.path.join(figures_path, f"ablation_{ablation_tag}")
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    logger.info("Figures path: %s", figures_path)
    return figures_path


def filter_features_by_groups(
    list_of_features: list[str],
    allowed_prefixes: list[str],
) -> list[str]:
    """Keep only features whose base variable name matches an allowed prefix."""
    return [
        f for f in list_of_features
        if any(f == p or f.startswith(p + "_") for p in allowed_prefixes)
    ]


def _unwrap_to_pipeline(estimator):
    """Extract the inner Pipeline from any meta-estimator wrapper.

    Handles all four CV cases:
    - GridSearchThresholdCV with param_grid  → best_estimator_ (Pipeline from GridSearchCV)
    - GridSearchThresholdCV without param_grid → estimator_ (Pipeline, no calibration in nested CV)
    - GridSearchCV (threshold_tuning=False)  → best_estimator_ (Pipeline)
    - Plain Pipeline                          → returned as-is
    Also handles GroupCalibratedClassifierCV inside estimator_ when present.
    """
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_
    if hasattr(estimator, "estimator_"):
        inner = estimator.estimator_
        if hasattr(inner, "calibrated_classifiers_"):
            return inner.calibrated_classifiers_[0].estimator
        return inner
    return estimator


def _make_preprocessor(pipeline):
    """Return an imblearn Pipeline containing all steps except 'model'."""
    pipeline = _unwrap_to_pipeline(pipeline)
    return Pipeline([(name, step) for name, step in pipeline.steps if name != "model"])


def _detect_model_class(pipeline) -> str:
    """Return 'tabpfn', 'tabicl', or 'sklearn' based on the fitted inner estimator."""
    pipeline = _unwrap_to_pipeline(pipeline)
    try:
        inner = pipeline["model"].estimator_
        # Unwrap CalibratedClassifierCV — its calibrated_classifiers_ hold the real model
        if hasattr(inner, "calibrated_classifiers_"):
            inner = inner.calibrated_classifiers_[0].estimator
        cls_name = type(inner).__name__.lower()
    except (AttributeError, KeyError, TypeError):
        return "sklearn"
    if "tabpfn" in cls_name:
        return "tabpfn"
    if "tabicl" in cls_name:
        return "tabicl"
    return "sklearn"


def _save_fold_models(fold_pipelines: list, folder: str, model_class: str) -> None:
    """Save per-fold model files to *folder* using the native format for the model class."""
    os.makedirs(folder, exist_ok=True)
    for i, fold_model in enumerate(fold_pipelines):
        pipeline = _unwrap_to_pipeline(fold_model)
        wrapper = pipeline["model"]
        inner = wrapper.estimator_

        if model_class == "tabpfn":
            save_fitted_tabpfn_model(inner, os.path.join(folder, f"fold_{i}.tabpfn_fit"))
        elif model_class == "tabicl":
            inner.save(
                os.path.join(folder, f"fold_{i}.pkl"),
                save_model_weights=False,   # if False, reload from checkpoint on load
                save_training_data=True,    # needed: X_train_pp cannot be reconstructed from KV cache alone
                save_kv_cache=True,         # also save KV cache if already built
            )

        with open(os.path.join(folder, f"fold_{i}_meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "batch_size": wrapper.batch_size,
                    "device":     wrapper.device,
                    "classes_":   wrapper.classes_,
                },
                f,
            )
        logger.info("Saved fold %d model (%s) to %s", i, model_class, folder)


def load_fold_models(pkl_path: str, saved: dict, device: str | None = None) -> list:
    """Return a list of fitted Pipeline objects for inference.

    For sklearn models the list is read directly from *saved* (already in
    memory).  For TabPFN / TabICL the per-fold native files are loaded from
    the dedicated folder that mirrors the pkl name.

    Parameters
    ----------
    pkl_path : str
        Absolute path to the ``*_sklearn*.pkl`` file.
    saved : dict
        Already-loaded pickle dict (avoids re-reading the file).
    device : str, optional
        Target device override, e.g. ``"cuda:0"`` or ``"cpu"``.
        Defaults to the device stored at save time.
    """
    model_class = saved.get("model_class", "sklearn")

    if model_class == "sklearn":
        return saved["fit_models"]

    folder = pkl_path[: -len(".pkl")]
    preprocessors = saved["preprocessors"]
    fold_pipelines = []

    for i, preprocessor in enumerate(preprocessors):
        meta_path = os.path.join(folder, f"fold_{i}_meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        target_device = device or meta["device"]

        if model_class == "tabpfn":
            tabpfn_device = "cpu" if target_device == "cpu" else f"cuda:{target_device}"
            inner = load_fitted_tabpfn_model(
                os.path.join(folder, f"fold_{i}.tabpfn_fit"),
                device=tabpfn_device,
            )
        elif model_class == "tabicl":
            inner = TabICLClassifier.load(os.path.join(folder, f"fold_{i}.pkl"))
        else:
            raise ValueError(f"Unknown model_class '{model_class}'")

        wrapper = ParallelEstimatorWrapper(
            estimator=None,
            batch_size=meta["batch_size"],
            device=target_device,
            verbose=False,
        )
        wrapper.estimator_ = inner
        wrapper.is_fitted_ = True
        wrapper.classes_ = meta["classes_"]

        fold_pipelines.append(Pipeline(preprocessor.steps + [("model", wrapper)]))
        logger.info("Loaded fold %d model (%s) from %s", i, model_class, folder)

    return fold_pipelines


def save_results(
    config,
    results,
    results_predict,
    fit_models,
    best_params,
    best_threshold,
    model,
    file_appendix,
    file_base_name="_sklearn",
    results_hero_ml=None,
    results_hero_score=None,
    hero_fold_counts=None,
    ablation_tag: str | None = None,
):
    results_path = get_results_path(config, ablation_tag)
    logger.info("Saving results for model '%s' to %s", model, results_path)

    fold_pipelines = fit_models[model]
    model_class = _detect_model_class(fold_pipelines[0])

    pkl_path = os.path.join(results_path, f"{model}{file_base_name}{file_appendix}.pkl")

    save_dict = {
        "results":         results[model],
        "results_predict": results_predict[model],
        "best_params":     best_params[model],
        "best_threshold":  best_threshold[model],
        "model_class":     model_class,
    }

    if results_hero_ml:
        save_dict["results_hero_ml"]    = results_hero_ml
    if results_hero_score:
        save_dict["results_hero_score"] = results_hero_score
    if hero_fold_counts:
        save_dict["hero_fold_counts"]   = hero_fold_counts

    if model_class == "sklearn":
        save_dict["fit_models"] = fold_pipelines
    else:
        # Preprocessor pipeline (all steps except "model") is pickle-safe
        save_dict["preprocessors"] = [
            _make_preprocessor(p) for p in fold_pipelines
        ]
        _save_fold_models(fold_pipelines, pkl_path[: -len(".pkl")], model_class)

    with open(pkl_path, "wb") as f:
        pickle.dump(save_dict, f)


def load_results(
    config,
    file_appendix="",
    file_base_name="_sklearn",
):
    results = {}

    results_path = get_results_path(config)
    for model in config.model["models"].keys():
        logger.info("Loading results for model: %s", model)
        with open(
            os.path.join(results_path, f"{model}{file_base_name}{file_appendix}.pkl"),
            "rb",
        ) as f:
            results[model] = pickle.load(f)

        print_results_summary(results[model])

    return results


def get_result_dict(sklearn_dict, verbose=True):
    results_sklearn_model = {}
    for metric, v in sklearn_dict.items():
        if verbose:
            logger.debug("%s: %s", metric, v)
        results_sklearn_model_metric = {}
        results_sklearn_model_metric["values"] = v
        results_sklearn_model_metric["mean"] = np.mean(v)
        results_sklearn_model_metric["std"] = np.std(v)
        results_sklearn_model[metric] = results_sklearn_model_metric
    return results_sklearn_model
    

_VARIABLES = [
    "sf_ratio", "SpO2", "bodyT", "weight", "FiO2", "SBP", "DBP",
    "HR", "RR", "MAP", "PEEP", "ventilation", "invasive", "VIS",
]  # longest/multi-word first to avoid prefix collisions

_DEMOG = {"sex", "ga_weeks"}

_VAR_LABELS: dict[str, str] = {
    "sf_ratio":    "SF Ratio",
    "SpO2":        "SpO2",
    "bodyT":       "Body Temp.",
    "weight":      "Body Weight",
    "FiO2":        "FiO2",
    "SBP":         "SBP",
    "DBP":         "DBP",
    "HR":          "HR",
    "RR":          "RR",
    "MAP":         "MAP",
    "PEEP":        "PEEP",
    "ventilation": "Mech. Ventilation",
    "invasive":    "Invasive Ventilation",
    "VIS":         "VIS",
    # demog
    "ga_weeks":    "Gestational Age",
    "sex":         "Sex",
}

_STAT_LABELS: dict[str, str] = {
    "median":                   "Median",
    "mean":                     "Mean",
    "std":                      "SD",
    "skew":                     "Skewness",
    "ApEn":                     "ApEn",
    "SampEn":                   "SampEn",
    "FuzzEn":                   "FuzzEn",
    "slope":                    "Trend",
    "nb_drops_corrected":       "# Drops",
    "nb_spikes_corrected":      "# Spikes",
    "nb_outliers_corrected":    "# Outliers",
    "nb_drops_corrected_ext":   "# Drops (ext)",
    "nb_spikes_corrected_ext":  "# Spikes (ext)",
    "nb_outliers_corrected_ext":"# Outliers (ext)",
}

_UPSTREAM_LABELS: dict[str, str] = {
    "avg_last6h":       "Mean last 6h",
    "avg_last12h":      "Mean last 12h",
    "avg_last24h":      "Mean last 24h",
    "avg_last48h":      "Mean last 48h",
    "avg_upstream":     "Mean last 72h",
    "avg_last6hto12h":  "Mean last 6–12h",
    "avg_last12hto24h": "Mean last 12–24h",
    "avg_last24hto48h": "Mean last 24–48h",
    "sd_last6h":        "SD last 6h",
    "sd_last12h":       "SD last 12h",
    "sd_last24h":       "SD last 24h",
    "sd_last48h":       "SD last 48h",
}

_UPSTREAMS = [
    "avg_last6hto12h", "avg_last12hto24h", "avg_last24hto48h",
    "avg_upstream", "avg_last48h", "avg_last24h", "avg_last12h", "avg_last6h",
    "sd_last48h", "sd_last24h", "sd_last12h", "sd_last6h",
]  # longer tokens first

_SLOPE_RE = re.compile(r"^slope_\d+$")
_SLOPE_FORMAT_RE = re.compile(r"^(.+)_slope_(\d+)$")


def parse_feature_name(name: str) -> dict:
    """Parse a feature name into its components.

    Handles three patterns:
    - ``VARIABLE_STAT``          → feature_type "signal"
    - ``VARIABLE_STAT_UPSTREAM`` → feature_type "signal_upstream"
    - ``DEMOG``                  → feature_type "demog"

    Returns
    -------
    dict with keys: variable, stat, upstream, feature_type
    """
    if name in _DEMOG:
        return {"variable": None, "stat": name, "upstream": None, "feature_type": "demog"}

    upstream = None
    remainder = name
    for up in _UPSTREAMS:
        suffix = "_" + up
        if name.endswith(suffix):
            upstream = up
            remainder = name[: -len(suffix)]
            break

    variable = None
    for var in _VARIABLES:
        prefix = var + "_"
        if remainder.startswith(prefix):
            variable = var
            stat_raw = remainder[len(prefix):]
            stat = "slope" if _SLOPE_RE.match(stat_raw) else stat_raw
            feature_type = "signal_upstream" if upstream else "signal"
            return {"variable": variable, "stat": stat, "upstream": upstream, "feature_type": feature_type}

    return {"variable": None, "stat": remainder, "upstream": upstream, "feature_type": "other"}


def format_feature_name(name: str) -> str:
    """Convert a raw feature column name to a human-readable label.

    Examples
    --------
    ``HR_median_avg_last24h``     → ``"HR Median [Mean last 24h]"``
    ``SpO2_ApEn_sd_last12h``      → ``"SpO2 ApEn [SD last 12h]"``
    ``HR_std``                    → ``"HR SD"``
    ``ga_weeks``                  → ``"Gestational Age"``
    """
    # Slope/trend features: {var}_slope_{hours}  e.g. HR_slope_24
    m = _SLOPE_FORMAT_RE.match(name)
    if m:
        var_label = _VAR_LABELS.get(m.group(1), m.group(1))
        return f"{var_label} Trend last {m.group(2)}h"

    parsed = parse_feature_name(name)

    if parsed["feature_type"] == "demog":
        stat = parsed["stat"]  # stat holds the raw column name for demog features
        return _VAR_LABELS.get(stat, stat.replace("_", " ").title())

    var      = parsed["variable"] or ""
    stat     = parsed["stat"]     or ""
    upstream = parsed["upstream"]

    var_label  = _VAR_LABELS.get(var, var)
    stat_label = _STAT_LABELS.get(stat, stat.replace("_", " ").title())
    up_label   = _UPSTREAM_LABELS.get(upstream, upstream.replace("_", " ")) if upstream else ""

    if up_label:
        return f"{var_label} {stat_label} [{up_label}]"
    return f"{var_label} {stat_label}".strip()


def format_variable_name(name: str) -> str:
    """Map a raw variable name (e.g. ``"bodyT"``) to its display label."""
    return _VAR_LABELS.get(name, name)


def build_scorers(scoring_dict):
    scorers = {}
    for name, metric in scoring_dict.items():
        if isinstance(metric, str):
            scorers[name] = get_scorer(metric)
        else:
            scorers[name] = metric  
    return scorers