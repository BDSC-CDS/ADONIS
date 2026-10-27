"""Post-hoc SHAP explanations for ADONIS-ML fold models.

Dispatches to the most appropriate SHAP algorithm per model type:
  - TreeExplainer   : xgb, rf  — exact, fast (seconds)
  - LinearExplainer : lr        — exact, fast
  - PermutationExplainer: everything else (svm, knn, naive, tabpfn, tabicl)
    — model-agnostic; uses the full outer pipeline predict_proba so calibration
      and any feature-selection steps are respected.

Note: TreeExplainer / LinearExplainer explain the raw (uncalibrated) base model
when calibration is enabled.  PermutationExplainer always explains the final
predict_proba output, including calibration.
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd
import shap

from adonis_ml.machine_learning.calibration import DecisionFunctionWrapper

logger = logging.getLogger(__name__)

_TREE_MODELS = frozenset({"xgb", "rf", "lgbm"})
_LINEAR_MODELS = frozenset({"lr"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_to_pipeline(outer_estimator):
    """Peel GridSearchThresholdCV / GroupCalibratedClassifierCV to reach the Pipeline.

    Handles the three cases that appear in fit_models_sklearn:
      1. GridSearchThresholdCV → Pipeline
      2. GridSearchThresholdCV → GroupCalibratedClassifierCV → Pipeline
         (accessed via calibrated_classifiers_[0].estimator)
      3. Plain Pipeline (TabPFN/TabICL reconstructed by load_fold_models)
    """
    est = outer_estimator
    for _ in range(4):
        if hasattr(est, "steps"):
            return est
        if hasattr(est, "calibrated_classifiers_"):
            # sklearn CalibratedClassifierCV: take fitted pipeline from fold 0
            return est.calibrated_classifiers_[0].estimator
        if hasattr(est, "estimator_"):
            est = est.estimator_
        elif hasattr(est, "estimator"):
            est = est.estimator
        else:
            break
    raise RuntimeError(
        f"Could not unwrap an imblearn/sklearn Pipeline from {type(outer_estimator)}"
    )


def _preprocess_X(pipeline, X_feat_df: pd.DataFrame) -> np.ndarray:
    """Apply selector → [imputer] → [scaler] steps, skipping undersampler (step 0) and model (last).

    imblearn Pipeline skips sampler steps automatically during predict, but here
    we iterate manually so we can hand the preprocessed array directly to a
    tree/linear SHAP explainer without going through the ParallelEstimatorWrapper.
    """
    X = X_feat_df
    for _, step in list(pipeline.steps)[1:-1]:
        X = step.transform(X)
    return np.asarray(X)


def _get_base_model(pipeline):
    """Extract the actual fitted estimator from ParallelEstimatorWrapper inside the pipeline."""
    wrapper = pipeline.named_steps["model"]
    # Unwrap ParallelEstimatorWrapper (TabPFN/TabICL use estimator_, others use estimator)
    model = wrapper.estimator_ if hasattr(wrapper, "estimator_") else wrapper.estimator
    # Unwrap DecisionFunctionWrapper inserted for calibration-friendly decision scores
    if isinstance(model, DecisionFunctionWrapper):
        return model.estimator_ if hasattr(model, "estimator_") else model.estimator
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_shap_fold0(
    outer_estimator,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_type: str,
    feature_names: list[str],
    n_background: int = 100,
    n_explain: int | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    """Compute SHAP values for a single fold-0 model on a given dataset.

    Parameters
    ----------
    outer_estimator
        Fitted estimator for fold 0 — either a GridSearchThresholdCV (sklearn
        models) or a reconstructed imblearn Pipeline (TabPFN / TabICL).
    X_background
        Full self.X DataFrame.  Used to build the SHAP background / masker.
    X_explain
        Samples to explain (e.g. fold-0 test set, or full external set).
    model_type
        Key from PIPELINE_CONFIGS, e.g. "xgb", "tabpfn".
    feature_names
        self.list_of_features — the columns the model was trained on.
    n_background
        Number of background samples drawn from X_background.
    n_explain
        If not None, randomly subsample X_explain to this many rows.
        Useful for slow PermutationExplainer models.

    Returns
    -------
    dict with keys:
        shap_values   : ndarray (n_explain, n_features)
        expected_value: float
        X_display     : ndarray (n_explain, n_features) — feature values for beeswarm
        feature_names : list[str]
    """
    if n_explain is not None and n_explain < len(X_explain):
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(X_explain), n_explain, replace=False)
        X_explain = X_explain.iloc[idx]
        logger.info("Subsampled X_explain to %d rows for model '%s'", n_explain, model_type)

    X_bg_feat = X_background[feature_names]
    X_exp_feat = X_explain[feature_names]
    bg_sample = shap.sample(X_bg_feat, min(n_background, len(X_bg_feat)))

    # ------------------------------------------------------------------
    # TreeExplainer — XGBoost, RandomForest
    # ------------------------------------------------------------------
    if model_type in _TREE_MODELS:
        pipeline = _unwrap_to_pipeline(outer_estimator)
        bg_proc = _preprocess_X(pipeline, bg_sample)
        exp_proc = _preprocess_X(pipeline, X_exp_feat)

        model = _get_base_model(pipeline)
        if model_type == "lgbm":
            # Hand the raw Booster to shap and skip the interventional background:
            # with boost_from_average=True on undersampled data, the interventional
            # baseline drifts from the booster's own init_score and the additivity
            # check fails. tree_path_dependent uses sample counts stored in the
            # trees — no background needed.
            explainer = shap.TreeExplainer(model.booster_)
        else:
            explainer = shap.TreeExplainer(model, bg_proc)
        sv = explainer.shap_values(exp_proc)
        ev = explainer.expected_value

        if model_type != "lgbm":
            # Binary classifiers (e.g. RandomForest) return per-class outputs:
            # shap_values as list [c0, c1] or 3D (n, feat, 2); expected_value as length-2.
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[..., 1]
            if hasattr(ev, "__len__"):
                ev = ev[1]

        logger.info(
            "TreeExplainer done for '%s': shap_values shape %s", model_type, sv.shape
        )
        return {
            "shap_values": sv,
            "expected_value": float(ev),
            "X_display": exp_proc,
            "X_raw": X_exp_feat.to_numpy(),
            "feature_names": feature_names,
        }

    # ------------------------------------------------------------------
    # LinearExplainer — Logistic Regression (no feature selection)
    # ------------------------------------------------------------------
    if model_type in _LINEAR_MODELS:
        pipeline = _unwrap_to_pipeline(outer_estimator)
        bg_proc = _preprocess_X(pipeline, bg_sample)
        exp_proc = _preprocess_X(pipeline, X_exp_feat)

        model = _get_base_model(pipeline)
        explainer = shap.LinearExplainer(model, bg_proc)
        sv = explainer.shap_values(exp_proc)
        ev = explainer.expected_value
        expected_value = float(ev[0] if hasattr(ev, "__len__") else ev)

        logger.info(
            "LinearExplainer done for '%s': shap_values shape %s", model_type, sv.shape
        )
        return {
            "shap_values": sv,
            "expected_value": expected_value,
            "X_display": exp_proc,
            "X_raw": X_exp_feat.to_numpy(),
            "feature_names": feature_names,
        }

    # ------------------------------------------------------------------
    # PermutationExplainer — all other models
    # Works with calibrated pipelines and feature-selection steps because
    # it treats the full pipeline as a black box.
    # imblearn Pipeline automatically skips the undersampler during predict.
    # ------------------------------------------------------------------
    if model_type not in _TREE_MODELS | _LINEAR_MODELS:
        logger.warning(
            "PermutationExplainer for '%s' can be slow with many features (%d). "
            "Use --shap-n-explain to limit the number of explained samples.",
            model_type,
            len(feature_names),
        )

    def _predict(X_np: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X_np, columns=feature_names)
        return outer_estimator.predict_proba(df)[:, 1]

    explainer = shap.PermutationExplainer(_predict, bg_sample.values)
    max_evals = max(500, 2 * len(feature_names) + 1)
    X_exp_arr = X_exp_feat.values
    n_rows = X_exp_arr.shape[0]
    n_feat = len(feature_names)

    # Resume from a prior partial run if available
    shap_values = np.full((n_rows, n_feat), np.nan, dtype=float)
    base_values = np.full(n_rows, np.nan, dtype=float)
    start = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as fh:
            ckpt = pickle.load(fh)
        if ckpt.get("n_rows") == n_rows and ckpt.get("n_feat") == n_feat:
            shap_values = ckpt["shap_values"]
            base_values = ckpt["base_values"]
            start = int(ckpt["next_idx"])
            logger.info(
                "Resuming SHAP for '%s' from row %d / %d (checkpoint: %s)",
                model_type, start, n_rows, checkpoint_path,
            )
        else:
            logger.warning(
                "Checkpoint shape mismatch for '%s' (expected %dx%d, got %dx%d) — recomputing.",
                model_type, n_rows, n_feat, ckpt.get("n_rows"), ckpt.get("n_feat"),
            )

    for i in range(start, n_rows):
        sv_row = explainer(X_exp_arr[i:i + 1], max_evals=max_evals)
        shap_values[i] = np.asarray(sv_row.values).reshape(n_feat)
        base_values[i] = float(np.asarray(sv_row.base_values).ravel()[0])
        logger.info("SHAP '%s' row %d / %d done", model_type, i + 1, n_rows)
        if checkpoint_path:
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump({
                    "n_rows": n_rows,
                    "n_feat": n_feat,
                    "shap_values": shap_values,
                    "base_values": base_values,
                    "next_idx": i + 1,
                }, fh)
            os.replace(tmp, checkpoint_path)

    logger.info(
        "PermutationExplainer done for '%s': shap_values shape %s",
        model_type,
        shap_values.shape,
    )
    return {
        "shap_values": shap_values,
        "expected_value": float(np.nanmean(base_values)),
        "X_display": X_exp_arr,
        "X_raw": X_exp_feat.to_numpy(),
        "feature_names": feature_names,
    }
