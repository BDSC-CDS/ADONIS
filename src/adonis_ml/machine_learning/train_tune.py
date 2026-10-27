import logging
import os
import pickle
import warnings
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns 

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.metrics import get_scorer, roc_curve, precision_recall_curve

from adonis_ml.machine_learning.splitting import GroupKFoldFractional
from adonis_ml.machine_learning.utils import (
    get_result_dict,
    get_results_path,
    get_figures_path,
    print_results_summary,
    save_results,
)
from adonis_ml.utils import AdonisConfig
from adonis_ml.machine_learning.utils import build_scorers
from adonis_ml.machine_learning.scoring import get_custom_scorer
from adonis_ml.machine_learning.scoring import compute_y_pred_window
from adonis_ml.machine_learning.dataloading import get_sklearn_inputs, get_df_hero, get_subgroups_from_df, get_weights_from_df
from adonis_ml.machine_learning.tuning import GridSearchThresholdCV
from adonis_ml.machine_learning.calibration import DecisionFunctionWrapper, GroupCalibratedClassifierCV, apply_prevalence_shift

# Models whose predict_proba is a sigmoid/softmax output but which don't
# natively expose decision_function. Wrapping them lets Platt calibration
# fit on the pre-sigmoid margin instead of a bounded probability.
_MODELS_NEEDING_DF_WRAPPER = frozenset({"xgb", "lgbm", "tabpfn", "tabicl", "knn", "rf"})
from adonis_ml.machine_learning.parallel import ParallelEstimatorWrapper
from adonis_ml.machine_learning.cross_validation import cross_validate

from adonis_ml.machine_learning.hero import HeroPredictor
from adonis_ml.machine_learning.plotting import save_subgroups_markdown

logger = logging.getLogger(__name__)


_FOREWARNING_TIME = 12


def _last_streak_fw_time(t_arr: np.ndarray, yp_arr: np.ndarray, forewarning_time: int = _FOREWARNING_TIME) -> float:
    """Return forewarning time (hours) based on the last uninterrupted positive streak.

    Walks backward from the window closest to onset (largest t < 0) through the
    available windows only; missing hours are skipped (HeRO coverage can be
    sparse). The streak extends as long as consecutive *available* predictions
    are positive. Returns abs(streak_start), or NaN if the first available
    window is not positive or no windows exist in the forewarning range.
    """
    fw_mask = (t_arr >= -forewarning_time) & (t_arr < 0) & np.isfinite(t_arr)
    t_fw = t_arr[fw_mask]
    yp_fw = yp_arr[fw_mask]
    if len(t_fw) == 0:
        return float("nan")

    # Sort descending: closest to onset first
    order = np.argsort(t_fw)[::-1]
    t_fw = t_fw[order]
    yp_fw = yp_fw[order]

    streak_start = None
    for i in range(len(t_fw)):
        if yp_fw[i] != 1:
            break
        streak_start = t_fw[i]

    return float(-streak_start) if streak_start is not None else float("nan")


def _decrement_window_counter(window_id: str) -> str:
    prefix, counter = window_id.rsplit("_", 1)
    return f"{prefix}_{int(counter) - 1}"


def _get_calibration_method(config):
    cal_config = getattr(config, "calibration", None)
    if cal_config and cal_config.get("enabled", False):
        return cal_config.get("method", "sigmoid")
    return None


PIPELINE_CONFIGS = {
    "lr": {
        "scaling": True,
        "imputation": True,
        "feature_selection": False,
    },
    "svm": {
        "scaling": True,
        "imputation": True,
        "feature_selection": True,
    },
    "knn": {
        "scaling": True,
        "imputation": True,
        "feature_selection": True,
    },
    "naive": {
        "scaling": True,
        "imputation": True,
        "feature_selection": True,
    },
    "rf": {
        "scaling": True,
        "imputation": True,
        "feature_selection": False,
    },
    "xgb": {
        "scaling": True,
        "imputation": True,
        "feature_selection": False,
    },
    "lgbm": {
        "scaling": True,
        "imputation": True,
        "feature_selection": False,
    },
    "tabpfn": {
        "scaling": False,
        "imputation": False,
        "feature_selection": False,
    },
    "tabicl": {
        "scaling": False,
        "imputation": False,
        "feature_selection": False,
    },
}


class AdonisTrainerTuner:
    """
    This class implements sklearn-based functions for model training, tuning and evaluation through (nested) cross-validation.

    The class accepts data (pandas Dataframes) and model candidates (sklearn Estimators) as input,
    together with a config (dict from json file) containing metadata about splits, variables, scaling + imputation etc.
    """

    def __init__(
        self,
        df,
        demog,
        config,
        estimator_pipeline_model_key="model",
        ablation_prefixes: list[str] | None = None,
        ablation_tag: str | None = None,
    ):
        """
        Arguments:
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        list_of_features : list of shape (n_features,)
            Lists the names of all features to be used during training.
            Must be a subset of the columns in X.

        groups : array-like of shape (n_samples,)
            Lists the group for each sample. This is used for grouped splitting
            during cross validation.

        config : adonis_ml.utils.AdonisConfig
            Configuration class containing the configuration for training, validation etc..
            Created from a dictionary/json file.

        estimator_pipeline_model_key : str, optional, default: "model",
            Key used to store the machine learning model in the estimator pipeline, together
            with imputer and scaler (as specified in config).

        ablation_prefixes : list of str, optional
            If given, only features whose name starts with one of these prefixes are used.
        """
        X, y, groups, list_of_features, sampling_column, eclipse_mask = get_sklearn_inputs(config, df)
        if ablation_prefixes is not None:
            from adonis_ml.machine_learning.utils import filter_features_by_groups
            list_of_features = filter_features_by_groups(list_of_features, ablation_prefixes)
            logger.info("Ablation: keeping %d / %d features matching prefixes %s",
                        len(list_of_features), X.shape[1], ablation_prefixes)
        subgroups = get_subgroups_from_df(config, df, demog)
        weights = get_weights_from_df(config, df, demog)
        
        self.df = df
        self.demog = demog
        self.X = X
        self.y = y
        n_samples, n_features = self.X.shape

        assert len(list_of_features) <= n_features
        assert len(groups) == n_samples
        assert isinstance(config, AdonisConfig)

        self.list_of_features = list_of_features
        self.sampling_column = sampling_column
        self.groups = groups
        self.config = config
        self.subgroups = subgroups
        self.sample_weight = weights
        self.patient_ids = self.df["stay_id"].values
        self.window_ids = self.df["window_id"].values
        self.eclipse_mask = eclipse_mask

        self.estimator_pipeline_model_key = estimator_pipeline_model_key
        self.ablation_tag = ablation_tag
        self.hero_max = self._load_hero_max(config)

        # Custom scores (pAUC, target_recall, min_recall)
        params = {
            "window_ids": self.window_ids,
            "max_fpr": self.config.params["max_fpr"],
            "target_recall": self.config.params["min_recall"],
            "min_recall": self.config.params["min_recall"],
            "penalty": self.config.params["penalty"],
        }
        self.metrics = self.config.metrics
        for metric_name, metric_id in self.metrics.items():
            self.metrics[metric_name] = get_custom_scorer(metric_id, **params)

        self.selection_score_grid_search = get_custom_scorer(self.config.selection_score_grid_search, **params)
        self.selection_score_threshold_tuning = get_custom_scorer(self.config.selection_score_threshold_tuning, **params)


    def _load_hero_max(self, config) -> np.ndarray | None:
        """Merge hero_max into the main df on (stay_id, time_difference_hours).

        Returns a float array of length n_samples, positionally aligned with
        self.X. Rows without a hero measurement are NaN.
        """
        try:
            df_hero = get_df_hero(config)
            df_temp = self.df[["stay_id", "time_difference_hours"]].reset_index(drop=True)
            df_merged = df_temp.merge(
                df_hero[["stay_id", "time_difference_hours", "hero_max"]],
                on=["stay_id", "time_difference_hours"],
                how="left",
            )
            n_with_hero = df_merged["hero_max"].notna().sum()
            logger.info(
                "Hero data loaded: %d / %d rows have hero_max", n_with_hero, len(df_merged)
            )
            return df_merged["hero_max"].values.astype(float)
        except Exception as exc:
            logger.warning("Hero data unavailable — hero evaluation will be skipped: %s", exc)
            return None

    def _get_threshold_grid(self):
        """Return the threshold grid for TunedThresholdClassifierCV.

        If ``threshold_max`` is set in config.params, returns
        np.linspace(0, threshold_max, threshold_n); otherwise returns
        threshold_n as an int so sklearn uses its default linspace(0, 1, n).
        """
        threshold_max = self.config.params.get("threshold_max")
        threshold_n   = self.config.params.get("threshold_n", 1000)
        if threshold_max is not None:
            logger.info(
                "Threshold grid: linspace(0, %.4f, %d)", threshold_max, threshold_n
            )
            return np.linspace(0, threshold_max, threshold_n)
        return threshold_n

    def run_evaluation_hero(
        self,
        model_type: str,
        results_predict: dict,
        exclude_eclipse: bool = False,
    ):
        """Compute ML-model and HeroPredictor metrics on the hero sub-cohort per fold.

        For each outer fold, selects the test-set rows that have a hero_max value,
        then evaluates:
          - The trained ML model (using pre-computed y_prob / y_pred)
          - HeroPredictor fitted on that fold's hero subset

        Parameters
        ----------
        model_type : str
        results_predict : dict  — results_predict_sklearn[model_type]
        exclude_eclipse : bool  — if True, additionally removes rows with
            time_difference_hours_sepsis in [-72, -12) before evaluation.

        Returns
        -------
        results_ml : dict  {metric: {"values", "mean", "std"}}  — ML on hero subset
        results_score : dict                                     — HeroPredictor on same subset
        fold_counts : list[dict]  — [{"fold", "n_total", "n_cases", "n_controls"}, ...]
        """
        if self.hero_max is None:
            return {}, {}, []

        from sklearn.metrics import get_scorer as _get_scorer
        scoring = {
            k: (_get_scorer(v) if isinstance(v, str) else v)
            for k, v in self.metrics.items()
        }

        time_diff_sepsis = (
            self.df["time_difference_hours_sepsis"].values
            if "time_difference_hours_sepsis" in self.df.columns
            else None
        )

        ml_vals = {m: [] for m in scoring}
        hp_vals = {m: [] for m in scoring}
        fold_counts = []

        for fold, fold_data in results_predict["test"].items():
            test_idx      = fold_data["test_idx"]
            y_true        = np.asarray(fold_data["y_true"])
            y_prob        = np.asarray(fold_data["y_prob"])
            y_pred        = np.asarray(fold_data["y_pred"])
            window_ids    = self.window_ids[test_idx]
            hero_max_test = self.hero_max[test_idx]
            hero_mask     = ~np.isnan(hero_max_test)
            stay_ids_test = self.patient_ids[test_idx]

            if exclude_eclipse and time_diff_sepsis is not None:
                t = time_diff_sepsis[test_idx]
                eclipse_mask = (t >= -72) & (t < -12)
                hero_mask = hero_mask & ~eclipse_mask

            n_total    = int(hero_mask.sum())
            n_cases    = int(y_true[hero_mask].sum())
            n_controls = n_total - n_cases

            n_patients_total    = int(np.unique(stay_ids_test[hero_mask]).shape[0])
            n_patients_cases    = int(np.unique(stay_ids_test[hero_mask & (y_true == 1)]).shape[0])
            n_patients_controls = int(np.unique(stay_ids_test[hero_mask & (y_true == 0)]).shape[0])

            logger.info(
                "Fold %d [%s] — hero sub-cohort: %d windows (%d cases, %d controls) | "
                "%d patients (%d cases, %d controls)",
                fold, model_type, n_total, n_cases, n_controls,
                n_patients_total, n_patients_cases, n_patients_controls,
            )
            fold_counts.append({
                "fold": fold,
                "n_total": n_total, "n_cases": n_cases, "n_controls": n_controls,
                "n_patients": n_patients_total,
                "n_patients_cases": n_patients_cases,
                "n_patients_controls": n_patients_controls,
            })

            if n_total < 10 or n_cases == 0:
                logger.warning(
                    "Fold %d [%s]: too few hero patients (%d) or no cases — skipping metrics",
                    fold, model_type, n_total,
                )
                continue

            y_true_h  = y_true[hero_mask]
            y_prob_h  = y_prob[hero_mask]
            y_pred_h  = y_pred[hero_mask]
            wids_h    = window_ids[hero_mask]

            # --- ML model metrics on hero subset ---
            for key, scorer in scoring.items():
                try:
                    if key == "recall_conservative":
                        val = scorer._score_func(y_true_h, y_pred_h, wids_h)
                    elif scorer._response_method == "predict":
                        val = scorer._score_func(y_true_h, y_pred_h, **scorer._kwargs)
                    else:
                        val = scorer._score_func(y_true_h, y_prob_h, **scorer._kwargs)
                    ml_vals[key].append(val)
                except Exception as exc:
                    logger.warning("Hero ML metric '%s' fold %d failed: %s", key, fold, exc)

            # --- HeroPredictor metrics on hero subset ---
            X_hero    = pd.DataFrame({"hero_max": hero_max_test[hero_mask]})
            hp        = HeroPredictor(use_max=True, fixed_prevalence=0.005)
            hp.fit(X_hero, y_true_h)
            hero_prob = hp.predict_proba(X_hero)[:, 1]
            hero_pred = hp.predict(X_hero)

            for key, scorer in scoring.items():
                try:
                    if key == "recall_conservative":
                        val = scorer._score_func(y_true_h, hero_pred, wids_h)
                    elif scorer._response_method == "predict":
                        val = scorer._score_func(y_true_h, hero_pred, **scorer._kwargs)
                    else:
                        val = scorer._score_func(y_true_h, hero_prob, **scorer._kwargs)
                    hp_vals[key].append(val)
                except Exception as exc:
                    logger.warning("Hero score metric '%s' fold %d failed: %s", key, fold, exc)

        results_ml    = get_result_dict(
            {m: np.array(v) for m, v in ml_vals.items() if v}, verbose=False
        )
        results_score = get_result_dict(
            {m: np.array(v) for m, v in hp_vals.items() if v}, verbose=False
        )
        return results_ml, results_score, fold_counts

    def get_hero_predict_dict(
        self,
        results_predict_sklearn: dict,
        apply_prior_shift: bool = False,
    ) -> dict:
        """Build a results_predict dict restricted to the hero sub-cohort.

        Returns a nested dict in the same format as results_predict_sklearn, usable
        directly by run_evaluation_curve(). Adds a 'hero_predictor' entry whose
        y_prob is derived from hero_max OR values using the hero cohort prevalence.

        Parameters
        ----------
        results_predict_sklearn : dict  — {model_type: {"test": {fold: fold_data}}}
        apply_prior_shift : bool  — if True, apply Saerens prior-shift correction to
            ML model probabilities (calibrated at whole-dataset prevalence) so they
            reflect the HeRO cohort prevalence. hero_predictor always uses the exact
            HeRO prevalence for OR→probability conversion.

        Returns
        -------
        dict  — same nested format, filtered to hero sub-cohort rows
        """
        if self.hero_max is None:
            return {}

        first_model = next(iter(results_predict_sklearn))

        # Prevalence of the whole dataset (training calibration reference)
        prev_main = float(self.y.mean())

        # Prevalence of the HeRO sub-cohort (computed from actual hero labels)
        hero_y_true_all = []
        for fold_data in results_predict_sklearn[first_model]["test"].values():
            test_idx = fold_data["test_idx"]
            hero_mask = ~np.isnan(self.hero_max[test_idx])
            y_true_h = np.asarray(fold_data["y_true"])[hero_mask]
            if y_true_h.size >= 10 and y_true_h.sum() > 0:
                hero_y_true_all.append(y_true_h)
        prev_hero = float(np.concatenate(hero_y_true_all).mean()) if hero_y_true_all else prev_main
        logger.info(
            "HeRO prevalence: %.4f  |  whole-dataset prevalence: %.4f", prev_hero, prev_main
        )

        results_hero: dict = {}

        # ML models: filter to hero sub-cohort rows per fold, optionally apply prior-shift
        for model_type, model_data in results_predict_sklearn.items():
            results_hero[model_type] = {"test": {}}
            for fold, fold_data in model_data["test"].items():
                test_idx = fold_data["test_idx"]
                hero_max_test = self.hero_max[test_idx]
                hero_mask = ~np.isnan(hero_max_test)

                n_total = int(hero_mask.sum())
                n_cases = int(np.asarray(fold_data["y_true"])[hero_mask].sum())
                if n_total < 10 or n_cases == 0:
                    continue

                y_prob = np.asarray(fold_data["y_prob"])[hero_mask]
                if apply_prior_shift:
                    y_prob = apply_prevalence_shift(y_prob, prev_main, prev_hero)

                results_hero[model_type]["test"][fold] = {
                    "y_true": np.asarray(fold_data["y_true"])[hero_mask],
                    "y_prob": y_prob,
                    "test_idx": test_idx[hero_mask],
                }

        # hero_predictor: convert hero_max OR to probability using exact HeRO prevalence
        results_hero["hero_predictor"] = {"test": {}}
        for fold, fold_data in results_predict_sklearn[first_model]["test"].items():
            test_idx = fold_data["test_idx"]
            hero_max_test = self.hero_max[test_idx]
            hero_mask = ~np.isnan(hero_max_test)

            n_total = int(hero_mask.sum())
            n_cases = int(np.asarray(fold_data["y_true"])[hero_mask].sum())
            if n_total < 10 or n_cases == 0:
                continue

            or_values = hero_max_test[hero_mask].astype(float)
            y_prob_hero = np.clip(
                (prev_hero * or_values) / (1.0 + prev_hero * (or_values - 1.0)),
                0.0, 1.0,
            )
            results_hero["hero_predictor"]["test"][fold] = {
                "y_true": np.asarray(fold_data["y_true"])[hero_mask],
                "y_prob": y_prob_hero,
                "test_idx": test_idx[hero_mask],
            }

        return results_hero

    def get_estimator_pipeline(self, model, model_type):
        """
        Wrap model (sklearn Estimator) in a Pipeline that
        - Selects and orders the correct columns, as expected by `list_of_features`.
        - Imputes missing data with the imputer specified in the config
        - Scales with the scaler specified in the config
        - Finally applies the model.

        This is useful to ensure that the model always sees consistently ordered, imputed
        and scaled data.

        This pipeline can also be used to save and deploy pre-fit models, ensuring the same
        data pipeline as during training.

        Arguments:
        ----------
        model : sklearn Estimator

        Returns:
        --------
        sklearn Pipeline as described above

        """
        req = PIPELINE_CONFIGS.get(
            model_type,
            {
                "scaling": True,
                "imputation": True,
                "feature_selection": True,
            },
        )

        steps = [
            (
                "undersampler",
                deepcopy(self.config.model["undersampler"]),
            ),
            (
                "selector",
                ColumnTransformer(
                    [
                        (
                            "selector",
                            "passthrough",
                            self.list_of_features,
                        )
                    ],
                    remainder="drop",
                ),
            ),
        ]

        # Imputation
        if req["imputation"]:
            steps.append(
                (
                    "imputer",
                    deepcopy(self.config.model["imputer"]),
                )
            )

        # Scaling
        if req["scaling"]:
            steps.append(
                (
                    "scaler", 
                    deepcopy(self.config.model["scaler"]).set_fit_request(sample_weight=False)
                )
            )

        # Feature selection
        if req["feature_selection"]:
            steps.append(
                (
                    "feature_selector",
                    deepcopy(self.config.model["feature_selector"]),
                )
            )

        # Final estimator
        batch_size = self.config.params["batch_size"]
        if model_type == "tabpfn":
            batch_size = min(4096, batch_size)
        elif model_type == "tabicl":
            batch_size = min(1024, batch_size)
        base_model = deepcopy(model)
        if model_type in _MODELS_NEEDING_DF_WRAPPER:
            base_model = DecisionFunctionWrapper(base_model)
        steps.append(
            (
                self.estimator_pipeline_model_key,
                ParallelEstimatorWrapper(
                    base_model,
                    batch_size,
                    "gpu" if model_type in ["xgb", "tabpfn", "tabicl"] else "cpu",
                    verbose=False,
                ),
            )
        )

        return Pipeline(steps)


    def refit_and_save_model(
        self,
        model,
        model_type,
        save_model=True,
    ):
        """
        Refit and safe a selected model on the whole training data set (X, y), without
        cross validation. This yields a final model but no performance estimate (unlike
        cross validation, which yields a performance estimate but no final model).

        This is useful to fit and deploy a final model, i.e. after tuning and model selection.

        Arguments:
        ----------
        model : sklearn Estimator
            The model to be refitted

        model_type : str
            The model name, used for saving the final model

        save_model : bool, default: True
            Whether to save the final model

        """
        logger.info("Refitting model '%s' on full dataset", model_type)
        estimator = self.get_estimator_pipeline(model, model_type)
        try:
            param_grid = self.config.model["param_grid"][model_type]
            param_grid_model = {}
            for k, v in param_grid.items():
                param_grid_model[
                    f"{self.estimator_pipeline_model_key}__estimator__{k}"
                ] = v
            param_grid = param_grid_model
        except:
            param_grid = None
        params = {}
        params["groups"] = self.groups
        # TODO: Fix it
        # params["window_ids"] = self.window_ids
        # params["window_ids"] = None

        if self.config.params["threshold_tuning"]:
            nested_estimator = GridSearchThresholdCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=self.config.cross_validation["inner_cv"],
                selection_score_grid_search=self.selection_score_grid_search,
                selection_score_threshold_tuning=self.selection_score_threshold_tuning,
                thresholds=self._get_threshold_grid(),
                refit=True,
                batch_size=self.config.params["batch_size"],
                n_jobs=self.config.params["n_jobs"],
                return_train_score=False,
                scoring_metrics=None,
                calibration_method=_get_calibration_method(self.config),
            )
        elif param_grid is not None:
            nested_estimator = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=self.config.cross_validation["inner_cv"],
                scoring=self.selection_score_grid_search,
                refit=True
            )
        else:
            nested_estimator = clone(estimator)
            params = {}

        nested_estimator.fit(self.X, self.y, **params)
        check_is_fitted(nested_estimator)

        if save_model:
            logger.info(
                "Saving refitted model '%s' to %s",
                model_type,
                os.path.join(get_results_path(self.config), f"{model_type}_sklearn_refitted.pkl"),
            )
            with open(
                os.path.join(
                    get_results_path(self.config), f"{model_type}_sklearn_refitted.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "model": nested_estimator,
                        "list_of_features": self.list_of_features,
                    },
                    f,
                )

        return nested_estimator

    def nested_cross_validation_sklearn(
        self,
        model,
        model_type,
        param_grid=None,
        return_estimator=True,
    ):
        """
        Perform (nested) cross validation on a specific model. This yields a performance estimate
        of the model on the outer splits (`self.config.cross_validation["outer_cv"]`) of the data.

        If a grid of hyperparameters is provided, nested cross validation is applied, using the
        inner splits (`self.config.cross_validation["inner_cv"]`) of the data and sklearn's `GridSearchCV`, to tune
        hyperparameters before computing the performance estimate. Hyperparameter tuning and
        model selection is based on `self.config.selection_score`.

        All scores in `self.config.metrics` are computed. If doing hyperparameter

        Arguments:
        ----------
        model : sklearn Estimator
            The model to be refitted

        param_grid : dict or list of dictionaries (as in `GridSearchCV`)
            Dictionary with parameter names (`str`) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

        return_estimator : bool, default: True
            Whether to return the fitted and tuned estimator.

        """
        estimator = self.get_estimator_pipeline(model, model_type)

        params = {
            "groups": self.groups,
            "window_ids": self.window_ids,
            "sample_weight": self.sample_weight,
            "eclipse_mask": self.eclipse_mask,
        }

        n_outer = self.config.cross_validation["outer_cv"].get_n_splits()
        logger.info(
            "Starting nested CV for model '%s' with %d outer folds",
            model_type,
            n_outer,
        )

        if param_grid is None:
            warnings.warn(
                "Received no parameter grid! Will call non-nested CV instead...",
            )
            if self.config.params["threshold_tuning"]:
                nested_estimator = GridSearchThresholdCV(
                    estimator=estimator,
                    cv=self.config.cross_validation["inner_cv"],
                    selection_score_threshold_tuning=self.selection_score_threshold_tuning,
                    thresholds=self._get_threshold_grid(),
                    refit=True,
                    batch_size=self.config.params["batch_size"],
                    n_jobs=self.config.params["n_jobs"],
                    return_train_score=self.config.params["return_train_score"],
                    scoring_metrics=self.metrics,
                    calibration_method=_get_calibration_method(self.config),
                )
            else:
                nested_estimator = clone(estimator)
                params = None
            return cross_validate(
                nested_estimator,
                self.X,
                y=self.y,
                groups=self.groups,
                scoring=self.metrics,
                cv=self.config.cross_validation["outer_cv"],
                window_ids=self.window_ids,
                eclipse_mask=self.eclipse_mask,
                params=params,
                return_estimator=return_estimator
            )

        if self.config.params["threshold_tuning"]:
            nested_estimator = GridSearchThresholdCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=self.config.cross_validation["inner_cv"],
                selection_score_grid_search=self.selection_score_grid_search,
                selection_score_threshold_tuning=self.selection_score_threshold_tuning,
                thresholds=self._get_threshold_grid(),
                refit=True,
                batch_size=self.config.params["batch_size"],
                n_jobs=self.config.params["n_jobs"],
                return_train_score=self.config.params["return_train_score"],
                scoring_metrics=self.metrics,
                calibration_method=_get_calibration_method(self.config),
            )
        else:
            nested_estimator = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=self.config.cross_validation["inner_cv"],
                scoring=self.selection_score_grid_search,
                refit=True
            )
        return cross_validate(
            nested_estimator,
            self.X,
            y=self.y,
            groups=self.groups,
            scoring=self.metrics,
            cv=self.config.cross_validation["outer_cv"],
            window_ids=self.window_ids,
            eclipse_mask=self.eclipse_mask,
            params=params,
            return_estimator=return_estimator,
        )

    def run_nested_cross_validation_sklearn(
        self,
        selected_models=None,
        file_appendix="",
        do_save_results=True,
    ):
        """
        Perform (nested) cross validation on the models provided in `self.config.model["models"]`.

        For details see `self.nested_cross_validation_sklearn`.

        Arguments:
        ----------

        selected_models : list of str, default: None
            List to subselect the model provided in `self.config.model["models"]`. If `None` the whole
            list in `self.config.model["models"]` will be used.

        file_appendix : str, default: ""
            Optional appendix to file name under which the results will be saved.

        do_save_results : bool, default: True
            Whether to save the results
        """
        if selected_models is None:
            selected_models = self.config.model["models"].keys()
        else:
            assert np.array(
                [m in self.config.model["models"] for m in selected_models]
            ).all(), "one of the selected models was not in config"
        results_sklearn = {model_type: {key: {}} for model_type in self.config.model["models"].keys() for key in ["test", "vali"]}
        results_predict_sklearn = {model_type: {key: {}} for model_type in self.config.model["models"].keys() for key in ["test", "vali"]}
        fit_models_sklearn = {}
        best_params_sklearn = {}
        best_threshold_sklearn = {}
        for model_type, model in self.config.model["models"].items():
            if model_type in selected_models:
                logger.info("Running nested CV for model: %s", model_type)
                try:
                    param_grid_model_ = self.config.model["param_grid"][model_type]
                    logger.info("Parameter grid found for model '%s':", model_type)
                    logger.debug("  raw param grid: %s", param_grid_model_)

                    param_grid_model = {}
                    for k, v in param_grid_model_.items():
                        param_grid_model[
                            f"{self.estimator_pipeline_model_key }__estimator__{k}"
                        ] = v
                    logger.debug(
                        "Prepending pipeline key '%s' to param grid keys",
                        self.estimator_pipeline_model_key,
                    )
                    logger.debug("  expanded param grid: %s", param_grid_model)
                except:
                    warnings.warn(f"No param grid found for model {model_type}...")
                    param_grid_model = None

                sklearn_dict = self.nested_cross_validation_sklearn(
                    model,
                    model_type,
                    param_grid=param_grid_model,
                )
                fit_models = sklearn_dict.pop("estimator")
                results_predict = sklearn_dict.pop("results_predict")

                # Results (validation set)
                if (
                    self.config.params["return_train_score"] &
                    (
                        (param_grid_model is not None) |
                        (self.config.params["threshold_tuning"])
                    )
                ):
                    sklearn_dict_mean_inner = {metric.replace("test","vali"): [] for metric in sklearn_dict.keys()}
                    results_predict_inner = {}
                    for fold, estimator in enumerate(fit_models):
                        sklearn_dict_inner = estimator.cv_results_
                        results_predict_inner[fold] = sklearn_dict_inner.pop("results_predict")
                        results_sklearn_model = get_result_dict(sklearn_dict_inner, verbose=False)

                        for metric in sklearn_dict.keys():    
                            sklearn_dict_mean_inner[metric.replace("test","vali")].append(results_sklearn_model[metric]["mean"])

                    results_sklearn_model_inner = get_result_dict(sklearn_dict_mean_inner, verbose=False)
                    print_results_summary(results_sklearn_model_inner)

                    results_sklearn[model_type]["vali"] = results_sklearn_model_inner
                    results_predict_sklearn[model_type]["vali"] = results_predict_inner
                
                # Results (test set)
                results_sklearn_model = get_result_dict(sklearn_dict, verbose=False)
                print_results_summary(results_sklearn_model)

                results_sklearn[model_type]["test"] = results_sklearn_model
                logger.info(
                    "Model '%s' CV complete — test scores: %s",
                    model_type,
                    {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in results_sklearn_model.items()},
                )
                results_predict_sklearn[model_type]["test"] = results_predict

                if param_grid_model is None:
                    best_params_sklearn[model_type] = None
                else:
                    best_params_sklearn[model_type] = [estimator.best_params_ for estimator in fit_models]

                fit_models_sklearn[model_type] = list(fit_models)
                if self.config.params["threshold_tuning"]:
                    best_threshold_sklearn[model_type] = [estimator.best_threshold_ for estimator in fit_models]
                else:
                    best_threshold_sklearn[model_type] = None

                # Hero cohort evaluation (inside CV test set, patients with a hero_max value)
                results_hero_ml, results_hero_score, hero_fold_counts = self.run_evaluation_hero(
                    model_type, results_predict_sklearn[model_type]
                )

                if do_save_results:
                    save_results(
                        self.config,
                        results_sklearn,
                        results_predict_sklearn,
                        fit_models_sklearn,
                        best_params_sklearn,
                        best_threshold_sklearn,
                        model_type,
                        file_appendix,
                        results_hero_ml=results_hero_ml,
                        results_hero_score=results_hero_score,
                        hero_fold_counts=hero_fold_counts,
                        ablation_tag=self.ablation_tag,
                    )

        return fit_models_sklearn, results_sklearn, results_predict_sklearn, best_params_sklearn, best_threshold_sklearn

    def run_external_validation(
        self,
        fit_models_sklearn: dict,
        best_threshold_sklearn: dict,
        file_appendix: str = "_external",
        do_save_results: bool = True,
    ) -> tuple[dict, dict]:
        """Apply pre-trained fold models to the external dataset held in self.X / self.y.

        Each of the N outer-fold models is applied to ALL external samples, producing
        N prediction sets.  Metrics are aggregated as mean ± std across folds, preserving
        the same dict structure as run_nested_cross_validation_sklearn so that
        run_evaluation_subgroups, run_evaluation_curve, and all plotting functions work
        without modification.

        Parameters
        ----------
        fit_models_sklearn : dict
            {model_type: [estimator_fold0, ..., estimator_foldN-1]}
            Loaded from the ``fit_models`` key of a ``*_sklearn.pkl`` file.
        best_threshold_sklearn : dict
            {model_type: [threshold_fold0, ...] or None}
            Loaded from the ``best_threshold`` key of the same pickle.
        file_appendix : str
            Suffix appended to the output pickle name, e.g. ``"_external_USZ"``.
        do_save_results : bool
            Whether to persist results to disk.

        Returns
        -------
        results_sklearn : dict
            {model_type: {"test": {metric: {"values", "mean", "std"}}}}
        results_predict : dict
            {model_type: {"test": {fold: {"test_idx", "y_true", "y_prob", "y_pred"}}}}
        """
        n_samples = len(self.X)
        test_idx = np.arange(n_samples)

        # Normalize: convert any string scorer names to scorer objects (mirrors cross_validate)
        scorers = {k: (get_scorer(v) if isinstance(v, str) else v) for k, v in self.metrics.items()}

        results_sklearn = {
            model_type: {"test": {}, "vali": {}}
            for model_type in fit_models_sklearn
        }
        results_predict = {
            model_type: {"test": {}, "vali": {}}
            for model_type in fit_models_sklearn
        }

        for model_type, fold_estimators in fit_models_sklearn.items():
            thresholds = best_threshold_sklearn.get(model_type)
            fold_scores = {f"test_{k}": [] for k in scorers}
            fold_scores_ga = {f"test_{k}_training_windows": [] for k in scorers}

            logger.info(
                "External validation — model '%s', %d fold estimators, %d samples",
                model_type, len(fold_estimators), n_samples,
            )

            window_ids_test = self.window_ids[test_idx]
            ga_mask = (self.X["train_knn_0.1"] == 1).values

            for fold, estimator in enumerate(fold_estimators):
                y_prob = estimator.predict_proba(self.X)[:, 1]

                if thresholds is not None:
                    y_pred = (y_prob >= thresholds[fold]).astype(int)
                else:
                    y_pred = estimator.predict(self.X)

                results_predict[model_type]["test"][fold] = {
                    "test_idx": test_idx,
                    "y_true": self.y,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                }

                # Metrics — all samples
                for key, scorer in scorers.items():
                    if key == "recall_conservative":
                        score = scorer._score_func(self.y, y_pred, window_ids_test)
                    elif scorer._response_method == "predict":
                        score = scorer._score_func(self.y, y_pred, **scorer._kwargs)
                    else:
                        score = scorer._score_func(self.y, y_prob, **scorer._kwargs)
                    fold_scores[f"test_{key}"].append(score)

                # Metrics — training windows (train_knn_0.1 == 1)
                y_true_tw = self.y[ga_mask]
                y_pred_tw = y_pred[ga_mask]
                y_prob_tw = y_prob[ga_mask]
                window_ids_tw = window_ids_test[ga_mask]
                for key, scorer in scorers.items():
                    if key == "recall_conservative":
                        score_tw = scorer._score_func(y_true_tw, y_pred_tw, window_ids_tw)
                    elif scorer._response_method == "predict":
                        score_tw = scorer._score_func(y_true_tw, y_pred_tw, **scorer._kwargs)
                    else:
                        score_tw = scorer._score_func(y_true_tw, y_prob_tw, **scorer._kwargs)
                    fold_scores_ga[f"test_{key}_training_windows"].append(score_tw)

                logger.info(
                    "  fold %d/%d — roc_auc=%.4f",
                    fold + 1, len(fold_estimators),
                    fold_scores["test_roc_auc"][-1] if "test_roc_auc" in fold_scores else float("nan"),
                )

            sklearn_dict = {
                k: np.array(v)
                for k, v in {**fold_scores, **fold_scores_ga}.items()
            }
            results_sklearn_model = get_result_dict(sklearn_dict, verbose=False)
            results_sklearn[model_type]["test"] = results_sklearn_model

            logger.info(
                "External validation '%s' complete — %s",
                model_type,
                {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in results_sklearn_model.items()},
            )
            print_results_summary(results_sklearn_model)

            if do_save_results:
                save_results(
                    self.config,
                    results_sklearn,
                    results_predict,
                    {model_type: fold_estimators},
                    {model_type: None},
                    {model_type: thresholds},
                    model_type,
                    file_appendix,
                    ablation_tag=self.ablation_tag,
                )

        return results_sklearn, results_predict

    def run_external_cv_validation(
        self,
        fit_models_sklearn: dict,
        best_threshold_sklearn: dict,
        file_appendix: str = "_external",
        do_save_results: bool = True,
    ) -> tuple[dict, dict]:
        """Split external dataset into K stratified folds and evaluate all fold models on each.

        For each (model_fold, test_fold) pair all N × K individual scores are collected
        and aggregated in a single step at the end.  With N=5 fold models and K=2 test
        splits this yields 10 values per metric before computing mean ± std.

        results_predict stores one entry per (model_fold, test_fold) pair under the flat
        key ``test_fold_idx * n_models + fold_idx``, so downstream curve and subgroup
        functions receive N × K individual predictions and iterate them uniformly.

        Parameters
        ----------
        fit_models_sklearn : dict
            {model_type: [estimator_fold0, ..., estimator_foldN-1]}
        best_threshold_sklearn : dict
            {model_type: [threshold_fold0, ...] or None}
        file_appendix : str
            Suffix appended to the output pickle name, e.g. ``"_external_USZ_testcv"``.
        do_save_results : bool
            Whether to persist results to disk.

        Returns
        -------
        results_sklearn : dict
            {model_type: {"test": {metric: {"values": array(N*K,), "mean", "std"}}}}
        results_predict : dict
            {model_type: {"test": {flat_key: {"test_idx", "y_true", "y_prob", "y_pred"}}}}
            Each entry is an individual (model_fold, test_fold) prediction.
        """
        test_cv = self.config.cross_validation["test_cv"]
        scorers = {k: (get_scorer(v) if isinstance(v, str) else v) for k, v in self.metrics.items()}
        ga_mask_full = (self.X["train_knn_0.1"] == 1).values

        results_sklearn = {
            model_type: {"test": {}, "vali": {}}
            for model_type in fit_models_sklearn
        }
        results_predict = {
            model_type: {"test": {}, "vali": {}}
            for model_type in fit_models_sklearn
        }

        for model_type, fold_estimators in fit_models_sklearn.items():
            thresholds = best_threshold_sklearn.get(model_type)
            n_models = len(fold_estimators)
            test_fold_scores = {f"test_{k}": [] for k in scorers}
            test_fold_scores_eclipse = {f"test_{k}_eclipse": [] for k in scorers}
            test_fold_scores_ga = {f"test_{k}_training_windows": [] for k in scorers}

            logger.info(
                "External CV validation — model '%s', %d fold estimators, test_cv n_splits=%d",
                model_type, n_models, test_cv.n_splits,
            )

            splits = list(test_cv.split(self.X, self.y, groups=self.groups))

            for test_fold_idx, (_, test_idx) in enumerate(splits):
                X_test = self.X.iloc[test_idx]
                y_test = self.y.iloc[test_idx]
                window_ids_test = self.window_ids[test_idx]
                ga_mask_test = ga_mask_full[test_idx]

                # Eclipse mask for this test fold
                non_eclipse = ~self.eclipse_mask[test_idx]

                # Collect per-model predictions on this test fold
                y_probs = []
                y_preds = []
                per_model_scores = {f"test_{k}": [] for k in scorers}
                per_model_scores_eclipse = {f"test_{k}_eclipse": [] for k in scorers}
                per_model_scores_ga = {f"test_{k}_training_windows": [] for k in scorers}

                for fold, estimator in enumerate(fold_estimators):
                    y_prob = estimator.predict_proba(X_test)[:, 1]
                    y_probs.append(y_prob)

                    if thresholds is not None:
                        y_pred = (y_prob >= thresholds[fold]).astype(int)
                    else:
                        y_pred = estimator.predict(X_test)
                    y_preds.append(y_pred)

                    # Metrics — eclipse excluded (main result, mirrors internal CV default)
                    y_test_ne = np.asarray(y_test)[non_eclipse]
                    y_pred_ne = y_pred[non_eclipse]
                    y_prob_ne = y_prob[non_eclipse]
                    window_ids_ne = window_ids_test[non_eclipse]
                    for key, scorer in scorers.items():
                        if key == "recall_conservative":
                            score = scorer._score_func(y_test_ne, y_pred_ne, window_ids_ne)
                        elif scorer._response_method == "predict":
                            score = scorer._score_func(y_test_ne, y_pred_ne, **scorer._kwargs)
                        else:
                            score = scorer._score_func(y_test_ne, y_prob_ne, **scorer._kwargs)
                        per_model_scores[f"test_{key}"].append(score)

                    # Metrics — eclipse included
                    for key, scorer in scorers.items():
                        if key == "recall_conservative":
                            score_ecl = scorer._score_func(y_test, y_pred, window_ids_test)
                        elif scorer._response_method == "predict":
                            score_ecl = scorer._score_func(y_test, y_pred, **scorer._kwargs)
                        else:
                            score_ecl = scorer._score_func(y_test, y_prob, **scorer._kwargs)
                        per_model_scores_eclipse[f"test_{key}_eclipse"].append(score_ecl)

                    # Metrics — training windows (train_knn_0.1 == 1)
                    y_true_tw = y_test[ga_mask_test]
                    y_pred_tw = y_pred[ga_mask_test]
                    y_prob_tw = y_prob[ga_mask_test]
                    window_ids_tw = window_ids_test[ga_mask_test]
                    for key, scorer in scorers.items():
                        if key == "recall_conservative":
                            score_tw = scorer._score_func(y_true_tw, y_pred_tw, window_ids_tw)
                        elif scorer._response_method == "predict":
                            score_tw = scorer._score_func(y_true_tw, y_pred_tw, **scorer._kwargs)
                        else:
                            score_tw = scorer._score_func(y_true_tw, y_prob_tw, **scorer._kwargs)
                        per_model_scores_ga[f"test_{key}_training_windows"].append(score_tw)

                # Accumulate all N individual scores (flat, no per-fold averaging)
                for key in test_fold_scores:
                    test_fold_scores[key].extend(per_model_scores[key])
                for key in test_fold_scores_eclipse:
                    test_fold_scores_eclipse[key].extend(per_model_scores_eclipse[key])
                for key in test_fold_scores_ga:
                    test_fold_scores_ga[key].extend(per_model_scores_ga[key])

                # Store per-(model_fold, test_fold) predictions for downstream use
                for fold_idx, (y_prob_f, y_pred_f) in enumerate(zip(y_probs, y_preds)):
                    flat_key = test_fold_idx * n_models + fold_idx
                    results_predict[model_type]["test"][flat_key] = {
                        "test_idx": test_idx,
                        "y_true": y_test,
                        "y_prob": y_prob_f,
                        "y_pred": y_pred_f,
                    }

                logger.info(
                    "  test fold %d/%d — roc_auc=%.4f (avg over %d models)",
                    test_fold_idx + 1, test_cv.n_splits,
                    np.mean(test_fold_scores["test_roc_auc"][-n_models:]) if "test_roc_auc" in test_fold_scores else float("nan"),
                    n_models,
                )

            sklearn_dict = {
                k: np.array(v)
                for k, v in {**test_fold_scores, **test_fold_scores_eclipse, **test_fold_scores_ga}.items()
            }
            results_sklearn_model = get_result_dict(sklearn_dict, verbose=False)
            results_sklearn[model_type]["test"] = results_sklearn_model

            logger.info(
                "External CV validation '%s' complete — %s",
                model_type,
                {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in results_sklearn_model.items()},
            )
            print_results_summary(results_sklearn_model)

            if do_save_results:
                save_results(
                    self.config,
                    results_sklearn,
                    results_predict,
                    {model_type: fold_estimators},
                    {model_type: None},
                    {model_type: thresholds},
                    model_type,
                    file_appendix,
                    ablation_tag=self.ablation_tag,
                )

        return results_sklearn, results_predict

    def run_evaluation_subgroups(
        self, 
        results_predict_sklearn
    ):
        scoring = self.metrics
        subgroups = self.subgroups
        window_ids = self.window_ids
        patient_ids = self.patient_ids
        config = self.config

        # Prepare scorers
        if scoring is None:
            scoring = {"score": get_scorer("accuracy")}
        elif isinstance(scoring, str):
            scoring = {"score": get_scorer(scoring)}
        elif callable(scoring):
            scoring = {"score": scoring}
        elif isinstance(scoring, dict):
            scoring = {k: (get_scorer(v) if isinstance(v, str) else v) for k, v in scoring.items()}
        else:
            raise ValueError("Invalid 'scoring' type")

        # Reconstruct y_all for the full dataset from fold test predictions so we can
        # compute global unique patient counts (not inflated by window-level splits).
        first_model_type = list(results_predict_sklearn.keys())[0]
        y_all = np.full(len(patient_ids), np.nan)
        for fold_data in results_predict_sklearn[first_model_type]["test"].values():
            idx = fold_data["test_idx"]
            y_all[idx] = np.where(np.isnan(y_all[idx]), fold_data["y_true"], y_all[idx])

        # Patients with at least one y_true==1 row are "sepsis patients"; the rest are pure controls.
        # Used to avoid counting sepsis patients' pre-onset (y_true==0) rows as controls.
        sepsis_pids = np.unique(patient_ids[~np.isnan(y_all) & (y_all == 1)])
        is_sepsis_pid_global = np.isin(patient_ids, sepsis_pids)  # one bool per dataset row

        # Global unique patient counts per subgroup column × category (same for all models/folds)
        subgroup_global_counts: dict[str, dict] = {}
        for subgroup_col in subgroups.columns:
            subgroup_global_counts[subgroup_col] = {}
            controls_in_all = config.subgroups[subgroup_col].get("controls_in_all_groups", False)
            for cat in subgroups[subgroup_col].cat.categories:
                # n_sepsis: only sepsis patients in THIS specific subgroup (not expanded with controls)
                mask_g = np.asarray(subgroups[subgroup_col] == cat)
                valid = ~np.isnan(y_all) & mask_g
                n_sep = int(np.unique(patient_ids[valid & is_sepsis_pid_global]).shape[0])
                if controls_in_all:
                    # All pure controls are in every category — count them globally
                    n_ctrl = int(np.unique(patient_ids[~np.isnan(y_all) & ~is_sepsis_pid_global]).shape[0])
                else:
                    n_ctrl = int(np.unique(patient_ids[valid & ~is_sepsis_pid_global]).shape[0])
                subgroup_global_counts[subgroup_col][cat] = {
                    "n_sepsis_total":   n_sep,
                    "n_controls_total": n_ctrl,
                }

        # Per-test-row boolean: is this row's patient a sepsis patient (globally)?
        is_sepsis_pid = np.isin(patient_ids, sepsis_pids)  # aligned with full dataset

        # Storage
        results_subgroups = {model_type: {} for model_type in results_predict_sklearn.keys()}

        for model_type, results_predict in results_predict_sklearn.items():
            subgroup_scores = {key: [] for key in subgroups.columns}
            for fold, results_predict_inner in results_predict["test"].items():
                test_idx = results_predict_inner["test_idx"]
                y_true = results_predict_inner["y_true"]
                y_prob = results_predict_inner["y_prob"]
                y_pred = results_predict_inner["y_pred"]

                subgroups_test = subgroups.iloc[test_idx]
                window_ids_test = window_ids[test_idx]
                patient_ids_test = patient_ids[test_idx]
                is_sepsis_test   = is_sepsis_pid[test_idx]   # True for sepsis patients' rows

                # Compute test metrics by subpopulation
                for subgroup_col in subgroups.columns:
                    sub_results = []
                    subgroup_values = subgroups[subgroup_col].cat.categories
                    controls_in_all = config.subgroups[subgroup_col].get("controls_in_all_groups", False)

                    for subgroup_value in subgroup_values:
                        mask = subgroups_test[subgroup_col] == subgroup_value
                        if controls_in_all:
                            # Include target sepsis patients (full stay) + pure controls.
                            # Exclude pre-onset rows of sepsis patients in OTHER subgroups.
                            subgroup_membership = np.asarray(mask)  # positional, aligned with test_idx
                            new_mask_arr = (subgroup_membership & is_sepsis_test) | ~is_sepsis_test
                            mask = pd.Series(new_mask_arr, index=subgroups_test.index)
                        subgroup_y_true = y_true[mask]
                        subgroup_y_pred = y_pred[mask]
                        subgroup_y_prob = y_prob[mask]
                        subgroup_window_ids = window_ids_test[mask]

                        if subgroup_y_true.empty:
                            continue  # skip if no samples in subgroup for this fold

                        subgroup_entry = {"fold": fold, "subgroup": subgroup_value}
                        for key, scorer in scoring.items():
                            if key == "recall_conservative":
                                subgroup_entry[key] = scorer._score_func(subgroup_y_true, subgroup_y_pred, subgroup_window_ids)
                            else:
                                if scorer._response_method == "predict":
                                    subgroup_entry[key] = scorer._score_func(subgroup_y_true, subgroup_y_pred, **scorer._kwargs)
                                else:
                                    subgroup_entry[key] = scorer._score_func(subgroup_y_true, subgroup_y_prob, **scorer._kwargs)

                        mask_arr = np.asarray(mask)
                        subgroup_entry["n"]          = int(np.unique(patient_ids_test[mask_arr]).shape[0])
                        subgroup_entry["n_sepsis"]   = int(np.unique(patient_ids_test[mask_arr & is_sepsis_test]).shape[0])
                        subgroup_entry["n_controls"] = int(np.unique(patient_ids_test[mask_arr & ~is_sepsis_test]).shape[0])

                        sub_results.append(subgroup_entry)

                    subgroup_scores[subgroup_col].extend(sub_results)

            for subgroup_col in subgroups.columns:
                subgroup_scores[subgroup_col] = pd.DataFrame(subgroup_scores[subgroup_col])

            for subgroup_col in subgroups.columns:
                logger.info(
                    "Subgroup '%s' results shape: %s for model '%s'",
                    subgroup_col,
                    subgroup_scores[subgroup_col].shape,
                    model_type,
                )
            # Attach global counts (same value for all folds) so parquet carries them
            for subgroup_col in subgroups.columns:
                df_sub = subgroup_scores[subgroup_col]
                if df_sub.empty:
                    continue
                df_sub["n_sepsis_total"] = df_sub["subgroup"].map(
                    {cat: v["n_sepsis_total"] for cat, v in subgroup_global_counts[subgroup_col].items()}
                )
                df_sub["n_controls_total"] = df_sub["subgroup"].map(
                    {cat: v["n_controls_total"] for cat, v in subgroup_global_counts[subgroup_col].items()}
                )
                subgroup_scores[subgroup_col] = df_sub

            results_subgroups[model_type] = subgroup_scores

        save_subgroups_markdown(self.config, results_subgroups)
        return results_subgroups


    def run_forewarning_time(self, results_predict_sklearn: dict, forewarning_time: int = _FOREWARNING_TIME) -> None:
        """Save a threshold-swept PPV vs forewarning-time curve per model.

        For each model we pool OOF predictions across folds, then for each
        unique observed probability τ compute:
          - PPV(τ) over all windows (sepsis + controls): TP/(TP+FP) using y_prob ≥ τ.
          - fw_time(τ): coverage-weighted mean over all sepsis patients of the
            last-streak forewarning time within [-forewarning_time, 0) applied
            to y_prob ≥ τ. Non-alarm patients contribute 0.

        Output: forewarning_curve_{forewarning_time}h[_hero].parquet per model.
        Schema: threshold | ppv | fw_time | n_alarm | n_sepsis
        """
        if "time_difference_hours_sepsis" not in self.df.columns:
            logger.warning("time_difference_hours_sepsis not in df; skipping forewarning curve.")
            return

        fw_tag = f"{forewarning_time}h"
        time_diff_all = self.df["time_difference_hours_sepsis"].values
        patient_ids = self.patient_ids
        results_path = get_results_path(self.config)

        # Main-cohort curves (all OOF windows)
        for model_type, model_data in results_predict_sklearn.items():
            curve = self._compute_forewarning_curve(
                model_data["test"], time_diff_all, patient_ids, forewarning_time,
            )
            if curve is None:
                continue
            model_dir = os.path.join(results_path, model_type)
            os.makedirs(model_dir, exist_ok=True)
            out_path = os.path.join(model_dir, f"forewarning_curve_{fw_tag}.parquet")
            curve.to_parquet(out_path, index=False)
            logger.info("Saved forewarning curve → %s  (%d thresholds)", out_path, len(curve))

        # HeRO sub-cohort curves for every ML model + hero_predictor
        if self.hero_max is not None:
            results_predict_hero = self.get_hero_predict_dict(results_predict_sklearn)
            for model_type, model_data in results_predict_hero.items():
                curve = self._compute_forewarning_curve(
                    model_data["test"], time_diff_all, patient_ids, forewarning_time,
                )
                if curve is None:
                    continue
                model_dir = os.path.join(results_path, model_type)
                os.makedirs(model_dir, exist_ok=True)
                out_path = os.path.join(model_dir, f"forewarning_curve_{fw_tag}_hero.parquet")
                curve.to_parquet(out_path, index=False)
                logger.info("Saved HeRO forewarning curve → %s  (%d thresholds)", out_path, len(curve))

    @staticmethod
    def _compute_forewarning_curve(
        folds: dict,
        time_diff_all: np.ndarray,
        patient_ids: np.ndarray,
        forewarning_time: int,
    ) -> pd.DataFrame | None:
        """Return a threshold | ppv | fw_time | n_alarm | n_sepsis DataFrame.

        `folds` is a dict {fold_idx: {"y_true", "y_prob", "test_idx"}}.
        """
        if not folds:
            return None

        y_true_parts, y_prob_parts, t_parts, pid_parts = [], [], [], []
        for fold_data in folds.values():
            test_idx = fold_data["test_idx"]
            y_true_parts.append(np.asarray(fold_data["y_true"], dtype=float))
            y_prob_parts.append(np.asarray(fold_data["y_prob"], dtype=float))
            t_parts.append(time_diff_all[test_idx])
            pid_parts.append(patient_ids[test_idx])

        y_true = np.concatenate(y_true_parts)
        y_prob = np.concatenate(y_prob_parts)
        t_pool = np.concatenate(t_parts)
        pid_pool = np.concatenate(pid_parts)

        # Fixed grid of 1000 thresholds between observed min/max probabilities.
        # Finer than needed for PPV (which has a closed-form cumulative walk)
        # but keeps the fw_time loop bounded at O(N_thresholds × N_sepsis × W).
        n_thresholds = 1000
        thresholds = np.linspace(float(y_prob.min()), float(y_prob.max()), n_thresholds)

        # PPV via a cumulative TP/FP walk + searchsorted. y_prob sorted ascending;
        # for threshold τ, positives are indices where y_prob >= τ, i.e. the
        # suffix starting at searchsorted(sorted, τ, side="left").
        order_asc = np.argsort(y_prob)
        y_prob_asc = y_prob[order_asc]
        y_true_asc = y_true[order_asc]
        # Cumulative counts from the TOP (highest probs) walking down:
        tp_from_top = np.cumsum(y_true_asc[::-1])[::-1]
        fp_from_top = np.cumsum((1.0 - y_true_asc)[::-1])[::-1]
        # Pad a 0 at the end so searchsorted hits can land past the array.
        tp_from_top = np.r_[tp_from_top, 0.0]
        fp_from_top = np.r_[fp_from_top, 0.0]
        pos = np.searchsorted(y_prob_asc, thresholds, side="left")
        tp = tp_from_top[pos]
        fp = fp_from_top[pos]
        denom = tp + fp
        ppv = np.where(denom > 0, tp / np.maximum(denom, 1.0), np.nan)

        # Identify sepsis patients: any pid with at least one y_true == 1 window
        # (labelling horizon is e.g. 12 h; patients still have label-0 windows
        # further pre-onset which we want to include in the fw_time streak).
        sepsis_pids = np.unique(pid_pool[y_true == 1])
        n_sepsis = int(len(sepsis_pids))
        if n_sepsis == 0:
            return None

        # For sepsis patients, keep every pre-onset window in [-forewarning_time, 0)
        # regardless of label. forewarning_time is the fw-time lookback (e.g. 72 h),
        # not the labelling horizon.
        pre_onset_mask = np.isfinite(t_pool) & (t_pool >= -forewarning_time) & (t_pool < 0)
        is_sepsis_window = np.isin(pid_pool, sepsis_pids)
        patient_window_mask = pre_onset_mask & is_sepsis_window

        # Pre-slice each patient's pre-onset windows once.
        per_patient: list[tuple[np.ndarray, np.ndarray]] = []
        for pid in sepsis_pids:
            m = patient_window_mask & (pid_pool == pid)
            per_patient.append((t_pool[m], y_prob[m]))

        fw_time_col = np.empty(len(thresholds), dtype=float)
        n_alarm_col = np.empty(len(thresholds), dtype=int)
        for i, tau in enumerate(thresholds):
            total, n_alarm = 0.0, 0
            for t_p, p_p in per_patient:
                fw = _last_streak_fw_time(t_p, (p_p >= tau).astype(int), forewarning_time)
                if np.isfinite(fw):
                    total += fw
                    n_alarm += 1
            fw_time_col[i] = total / n_sepsis
            n_alarm_col[i] = n_alarm

        # Suppress thresholds where no sepsis patient alarms: the window-level
        # PPV at those τ reduces to a 1/0 coin flip on a handful of firings and
        # isn't meaningful alongside a fw_time of 0.
        ppv = np.where(n_alarm_col > 0, ppv, np.nan)

        prevalence = float(np.mean(y_true)) if len(y_true) else float("nan")

        return pd.DataFrame({
            "threshold":  thresholds,
            "ppv":        ppv,
            "fw_time":    fw_time_col,
            "n_alarm":    n_alarm_col,
            "n_sepsis":   n_sepsis,
            "prevalence": prevalence,
        })


    def run_evaluation_curve(
        self,
        results_predict_sklearn,
        max_threshold: float = 0.05,  # kept for API compat; no longer used for DCA grid
    ):
        del max_threshold  # DCA is now evaluated at unique observed probabilities

        results_roc_curve = {model_type: {} for model_type in results_predict_sklearn.keys()}
        results_precision_recall_curve = {model_type: {} for model_type in results_predict_sklearn.keys()}
        results_dca_curve = {model_type: {} for model_type in results_predict_sklearn.keys()}

        for model_type, results_predict in results_predict_sklearn.items():
            test_roc_curve = []
            test_precision_recall_curve = []

            for fold, results_predict_inner in results_predict["test"].items():
                y_true = results_predict_inner["y_true"]
                y_prob = results_predict_inner["y_prob"]

                # ROC (per-fold, kept as-is)
                roc_entry = {}
                roc_entry["fpr"], roc_entry["tpr"], _ = roc_curve(y_true, y_prob)
                roc_entry = pd.DataFrame(roc_entry)
                roc_entry["fold"] = fold
                test_roc_curve.append(roc_entry)

                # PRC (per-fold, kept as-is)
                prc_entry = {}
                prc_entry["precision"], prc_entry["recall"], _ = precision_recall_curve(y_true, y_prob)
                prc_entry = pd.DataFrame(prc_entry)
                prc_entry["fold"] = fold
                test_precision_recall_curve.append(prc_entry)

            # DCA (pooled OOF, one curve per model, evaluated at unique probs)
            y_true_pool = np.concatenate([
                np.asarray(fd["y_true"], dtype=float)
                for fd in results_predict["test"].values()
            ])
            y_prob_pool = np.concatenate([
                np.asarray(fd["y_prob"], dtype=float)
                for fd in results_predict["test"].values()
            ])

            # Sort by probability descending; cumulative TP / FP walk down the ranked list.
            order = np.argsort(-y_prob_pool)
            y_true_desc = y_true_pool[order]
            y_prob_desc = y_prob_pool[order]
            tp_cum = np.cumsum(y_true_desc)
            fp_cum = np.cumsum(1.0 - y_true_desc)

            # Collapse ties: keep the last position of each distinct probability so
            # the count at that threshold includes all tied predictions.
            distinct_mask = np.r_[np.diff(y_prob_desc) != 0, True]
            thresholds = y_prob_desc[distinct_mask]
            tp = tp_cum[distinct_mask]
            fp = fp_cum[distinct_mask]

            N = len(y_true_pool)
            prevalence = y_true_pool.mean()
            weight = np.where(thresholds < 1.0, thresholds / (1.0 - thresholds), np.nan)
            nb_model = tp / N - weight * fp / N
            nb_treat_all = prevalence - weight * (1.0 - prevalence)

            dca_df = pd.DataFrame({
                "threshold":   thresholds,
                "net_benefit": nb_model,
                "treat_all":   nb_treat_all,
                "prevalence":  prevalence,
            }).sort_values("threshold").reset_index(drop=True)

            results_roc_curve[model_type] = pd.concat(test_roc_curve, axis=0, ignore_index=True)
            results_precision_recall_curve[model_type] = pd.concat(test_precision_recall_curve, axis=0, ignore_index=True)
            results_dca_curve[model_type] = dca_df

        return results_roc_curve, results_precision_recall_curve, results_dca_curve

    def run_shap_analysis(
        self,
        fit_models_sklearn: dict,
        results_predict: dict,
        n_background: int = 100,
        n_explain: int | None = None,
        source: str = "internal",
    ) -> dict:
        """Compute SHAP values using the fold-0 model for each model type.

        Parameters
        ----------
        fit_models_sklearn
            {model_type: [estimator_fold0, ..., estimator_foldN-1]}
        results_predict
            {model_type: {"test": {fold: {"test_idx", "y_true", ...}}}}
            Used only when source="internal" to locate fold-0 test indices.
        n_background
            Background sample size passed to the SHAP explainer.
        n_explain
            Cap on the number of explained samples.  None = explain all.
            Set this for PermutationExplainer models (tabpfn, tabicl, svm, …)
            to keep runtime manageable.
        source
            "internal" — explain fold-0 test samples from nested CV.
            "external" — explain all samples in self.X (external validation).

        Returns
        -------
        dict[model_type -> shap_result_dict]
        """
        from adonis_ml.machine_learning.explainability import compute_shap_fold0

        results_shap = {}
        results_path = get_results_path(self.config)

        for model_type, fold_estimators in fit_models_sklearn.items():
            outer_estimator = fold_estimators[0]

            if source == "internal":
                test_idx = results_predict[model_type]["test"][0]["test_idx"]
                X_explain = self.X.iloc[test_idx]
            else:
                X_explain = self.X

            logger.info(
                "SHAP — model '%s', %d explain samples (cap=%s), %d background samples",
                model_type,
                len(X_explain),
                n_explain,
                n_background,
            )

            checkpoint_path = os.path.join(results_path, f"{model_type}_shap_partial.pkl")
            shap_result = compute_shap_fold0(
                outer_estimator,
                X_background=self.X,
                X_explain=X_explain,
                model_type=model_type,
                feature_names=self.list_of_features,
                n_background=n_background,
                n_explain=n_explain,
                checkpoint_path=checkpoint_path,
            )
            shap_result["model_type"] = model_type

            pkl_path = os.path.join(results_path, f"{model_type}_shap.pkl")
            with open(pkl_path, "wb") as fh:
                pickle.dump(shap_result, fh)
            logger.info("Saved SHAP results → %s", pkl_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            results_shap[model_type] = shap_result

        return results_shap

    def compute_and_save_metrics_per_window_hour(
        self,
        model_type: str,
        results_predict: dict,
    ) -> None:
        """Compute classification metrics per relative window hour and save as parquet.

        For each fold in results_predict["test"], assembles the test-set DataFrame,
        applies the in-memory window patch, adds relative_hour (1–12 = −12 to −1 h
        before sepsis), and computes metrics at each hour across all samples (controls
        + eclipse + sepsis).  Results are saved to:
            {results_path}/{model_type}/metrics_over_time.parquet

        Parquet columns: fold, relative_hour, <metric1>, <metric2>, ...
        """
        scorers = build_scorers(self.config.metrics)

        rows = []
        folds_data = results_predict.get("test", results_predict)

        for fold, fold_data in folds_data.items():
            test_idx = fold_data["test_idx"]
            y_prob   = np.asarray(fold_data["y_prob"])
            y_true   = np.asarray(fold_data["y_true"])
            y_pred   = np.asarray(fold_data["y_pred"])

            df_sub = self.df.iloc[test_idx].copy()
            fold_df = pd.DataFrame({
                "window_id":             self.window_ids[test_idx],
                "time_difference_hours": df_sub["time_difference_hours"].values,
                "time_diff_sepsis":      df_sub["time_difference_hours_sepsis"].values
                                         if "time_difference_hours_sepsis" in df_sub.columns
                                         else np.full(len(test_idx), np.nan),
                "y_prob": y_prob,
                "y_true": y_true,
                "y_pred": y_pred,
            })

            # ── In-memory patch ─────────────────────────────────────────────
            # TODO: remove this patch once dataset is re-extracted from the pipeline
            fold_df = fold_df[fold_df["time_diff_sepsis"] != 0].copy()
            t_col = fold_df["time_diff_sepsis"]
            boundary = t_col.notna() & (t_col < 0) & (t_col % _FOREWARNING_TIME == 0)
            fold_df.loc[boundary, "window_id"] = (
                fold_df.loc[boundary, "window_id"].map(_decrement_window_counter)
            )
            # ────────────────────────────────────────────────────────────────

            fold_df["relative_hour"] = (
                fold_df.groupby("window_id")["time_difference_hours"]
                .rank(method="first")
                .astype(int)
            )

            for hour, grp in fold_df.groupby("relative_hour"):
                yt = grp["y_true"].values
                yp = grp["y_prob"].values
                yd = grp["y_pred"].values

                if len(np.unique(yt)) < 2:
                    continue

                row: dict = {"fold": fold, "relative_hour": int(hour)}
                for metric_name, scorer in scorers.items():
                    try:
                        if scorer._response_method == "predict":
                            row[metric_name] = scorer._score_func(yt, yd, **scorer._kwargs)
                        else:
                            row[metric_name] = scorer._score_func(yt, yp, **scorer._kwargs)
                    except Exception as exc:
                        logger.warning(
                            "Metric '%s' at hour %d fold %d failed: %s",
                            metric_name, hour, fold, exc,
                        )
                        row[metric_name] = np.nan
                rows.append(row)

        if not rows:
            logger.warning("compute_and_save_metrics_per_window_hour: no rows generated for %s", model_type)
            return

        df_out = pd.DataFrame(rows).sort_values(["fold", "relative_hour"])
        results_path = get_results_path(self.config)
        model_path   = os.path.join(results_path, model_type)
        os.makedirs(model_path, exist_ok=True)
        pq_path = os.path.join(model_path, "metrics_over_time.parquet")
        df_out.to_parquet(pq_path, index=False)
        logger.info("Saved metrics_over_time parquet: %s (%d rows)", pq_path, len(df_out))


class AdonisTrainerTunerHeRO(AdonisTrainerTuner):
    def __init__(
        self,
        df,
        demog,
        config,
        estimator_pipeline_model_key="model",
        ablation_prefixes: list[str] | None = None,
        ablation_tag: str | None = None,
    ):
        super().__init__(
            df,
            demog,
            config,
            estimator_pipeline_model_key,
            ablation_prefixes,
            ablation_tag,
        )

        self.df_hero = get_df_hero(config)

        train_idx = (~df["stay_id"].isin(self.df_hero["stay_id"].unique()))
        self.X = self.X[train_idx].reset_index(drop=True)
        self.y = self.y[train_idx].reset_index(drop=True)
        self.groups = self.groups[train_idx]
        self.sample_weight = self.sample_weight[train_idx]
        self.window_ids = self.window_ids[train_idx]

        self.scorers = build_scorers(self.config.metrics)

        self.hero_idx = ~self.df_hero["hero"].isna()
        self.X_ext = self.df_hero.loc[self.hero_idx, self.list_of_features + ["hero","hero_max"]].reset_index(drop=True)
        self.y_ext = self.df_hero.loc[self.hero_idx, "sepsis_time"].reset_index(drop=True)


    def run_evaluation_with_hero(self, estimators):
        # Fit on the HeRO evaluation set so proba_baseline matches the evaluation cohort.
        # This keeps OR scales consistent between HeRO ORs and ML model ORs.
        estimators["hero"] = HeroPredictor(use_max=False).fit(self.X_ext, self.y_ext)

        # Add HeRO max estimator
        # estimators["hero_max"] = HeroPredictor(use_max=True).fit(self.X_ext, self.y_ext)

        results_sklearn, results_prob, results_or, results_metrics = self.compare_with_hero(estimators)
        return results_sklearn, results_prob, results_or, results_metrics
        

    def compare_with_hero(
            self, 
            estimators, 
            do_save_results=True,
            file_base_name="sklearn"
        ):
        """
        Compare different models against the HeRO baseline.
        Saves evaluation DataFrames for each model and returns results.
        """
        import warnings
        warnings.filterwarnings('ignore')

        results_sklearn = {}
        results_prob = [] 
        results_metrics = []

        for model_type, estimator in estimators.items():
            logger.info("Evaluating HeRO comparison for model: %s", model_type)
            results_sklearn_model, results_sklearn_prob, results_sklearn_metrics = self._evaluate_model(model_type, estimator)

            print_results_summary(results_sklearn_model)
            logger.info(
                "Model '%s' HeRO eval — avg_fw_time: %s, avg_fa_rate: %s",
                model_type,
                results_sklearn_model.get("avg_fw_time"),
                results_sklearn_model.get("avg_fa_rate"),
            )
            results_sklearn[model_type] = results_sklearn_model

            results_prob.append(results_sklearn_prob)
            results_metrics.append(results_sklearn_metrics)
        
        results_prob = pd.concat(results_prob, ignore_index=True)
        results_metrics = pd.concat(results_metrics, ignore_index=True)

        # Build OR DataFrame (convert probability to OR)
        hero_predictor = estimators.get("hero")
        results_or = results_prob[["stay_id", "time_difference_hours_sepsis", "model", "y_true"]].copy()
        results_or["y_or"] = np.nan

        for model_type in results_or["model"].unique():
            mask = results_or["model"] == model_type
            if model_type in ("hero", "hero_max"):
                col = "hero_max" if model_type == "hero_max" else "hero"
                results_or.loc[mask, "y_or"] = self.X_ext[col].values
            elif hero_predictor is not None:
                results_or.loc[mask, "y_or"] = hero_predictor.predict_or(
                    results_prob.loc[mask, "y_proba"].values
                )

        if do_save_results:
            results_path = get_results_path(self.config)

            with open(
                os.path.join(results_path, f"{file_base_name}_hero.pkl"),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "results": results_sklearn,
                        "results_prob": results_prob,
                        "results_or": results_or,
                        "results_metrics": results_metrics,
                        "fit_models": estimators
                    },
                    f)

        return results_sklearn, results_prob, results_or, results_metrics
    

    def _evaluate_model(self, model_type, estimator):
        """
        Evaluate one model and return (metrics, df_eval).
        """
        # Predictions
        y_pred, y_prob = self._predict_model(model_type, estimator)

        # Build evaluation DataFrame
        results_sklearn_prob = (
            self.df_hero.loc[self.hero_idx, ["stay_id", "time_difference_hours_sepsis"]]
            .reset_index(drop=True)
            .copy()
        )
        results_sklearn_prob["model"] = model_type
        results_sklearn_prob["y_true"] = self.y_ext
        results_sklearn_prob["y_pred"] = y_pred
        results_sklearn_prob["y_proba"] = y_prob

        # Compute metrics
        results_sklearn_model = {}
        for metric_name, scorer in self.scorers.items():
            if scorer._response_method == "predict":
                results_sklearn_model[metric_name] = scorer._score_func(self.y_ext, y_pred, **scorer._kwargs)
            else:
                results_sklearn_model[metric_name] = scorer._score_func(self.y_ext, y_prob, **scorer._kwargs)

        # Custom metrics
        results_sklearn_model["avg_fw_time"] = self.forewarning_time_score(results_sklearn_prob, model_type)
        results_sklearn_model["avg_fa_rate"] = self.false_alarm_count(results_sklearn_prob)

        results_sklearn_metrics = self.compute_metrics_over_time(model_type, y_pred, y_prob)

        return results_sklearn_model, results_sklearn_prob, results_sklearn_metrics
    

    def _get_X_input(self, model_type):
        """
        Select appropriate input features depending on model type.
        """
        if model_type == "hero":
            return self.X_ext.loc[:, "hero":"hero"]
        elif model_type == "hero_max":
            return self.X_ext.loc[:, "hero_max":"hero_max"]
        return self.X_ext
    

    def _predict_model(self, model_type, estimator):
        """
        Run predict and predict_proba for the given model.
        """
        X_input = self._get_X_input(model_type)
        y_pred = estimator.predict(X_input)
        y_proba = estimator.predict_proba(X_input)[:, 1]
        return y_pred, y_proba


    def _calibrate_model(self, estimator):
        """
        Calibrate probabilities using inner CV.
        """
        fit_params = {"groups": self.groups}
        try:
            calibrated_estimator = GroupCalibratedClassifierCV(
                estimator=clone(estimator.estimator_),
                method="sigmoid",
                cv=self.config.cross_validation["inner_cv"],
            )
        except:
            calibrated_estimator = GroupCalibratedClassifierCV(
                estimator=clone(estimator),
                method="sigmoid",
                cv=self.config.cross_validation["inner_cv"],
            )
        calibrated_estimator.fit(self.X, self.y, None, **fit_params)
        check_is_fitted(calibrated_estimator)
        
        return calibrated_estimator.predict_proba(self.X_ext)[:, 1]


    def forewarning_time_score(self, df_eval, model_type, plot=False):
        fw_times = []
        for _, group in df_eval.groupby("stay_id"):
            group = group.sort_values("time_difference_hours_sepsis", ascending=False)

            # Identify group type (sepsis vs control)
            y_label = group["y_true"].max()  # 1 for sepsis, 0 for control

            # Find first correct prediction
            correct = group[(group["y_pred"] == 1)]
            if not correct.empty:
                fw_time = correct["time_difference_hours_sepsis"].min()
                if y_label == 1:
                    fw_times.append({"group": "sepsis", "fw_time": fw_time})
                else:
                    fw_times.append({"group": "control", "fw_time": fw_time})
        
        fw_times = pd.DataFrame(fw_times)
        results = fw_times.groupby('group')["fw_time"].agg(mean='mean', std='std')

        if plot:
            sns.set_theme(style="whitegrid", font_scale=1)
            plt.figure(figsize=(7, 4), dpi=300)

            colors=["k", "r"]
            bins = np.linspace(-47, 0, 48)  # 1-hour bins

            sns.histplot(
                data=fw_times,
                x="fw_time",
                hue="group",
                bins=bins, 
                stat='density', 
                legend=False,
                edgecolor="black", alpha=0.6
            )
            
            for i, group in enumerate(["sepsis", "control"]):
                plt.axvline(
                    results.loc[group, "mean"], 
                    color=colors[i], 
                    linestyle="--", 
                    linewidth=1.6,
                    label=f"{group} (mean = {results.loc[group, 'mean']:.1f} ± {results.loc[group, 'std']:.1f} h)"
                )

            plt.xlim(-47, 0)
            plt.xticks(np.arange(-48, 1, 6))
            plt.xlabel("Time to CRASH (hours)")
            plt.ylabel("Frequency")
            plt.legend(frameon=False)
            plt.tight_layout()

            out_path = os.path.join(get_figures_path(self.config), f"hero_forewarning_time_{model_type}.png")

            plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
            plt.close()  
        
        return results.loc["sepsis", "mean"]


    def false_alarm_count(self, df_eval):
        fa_per_patient = []
        for _, group in df_eval.groupby("stay_id"):
            fa = group[(group["y_true"] == 0) & (group["y_pred"] == 1)]
            if not fa.empty:
                fa_per_patient.append(fa.shape[0])
            else:
                fa_per_patient.append(0)

        avg_fa_patient = np.mean(fa_per_patient)
        std_fa_patient = np.std(fa_per_patient)
        
        return avg_fa_patient


    def compute_metrics_over_time(self, model_type, y_pred, y_prob):
        results = []
        for time, group in self.df_hero.loc[self.hero_idx,:].reset_index(drop=True).groupby("time_difference_hours_sepsis"):
            row = {
                "model": model_type,
                "time_difference_hours_sepsis": time
            }
            for metric_name, scorer in self.scorers.items():
                if scorer._response_method == "predict":
                    row[metric_name] = scorer._score_func(self.y_ext.iloc[group.index], y_pred[group.index], **scorer._kwargs)
                else:
                    row[metric_name] = scorer._score_func(self.y_ext.iloc[group.index], y_prob[group.index], **scorer._kwargs)

            results.append(row)

        return pd.DataFrame(results).sort_values(["model","time_difference_hours_sepsis"])


class AdonisTrainerFractional(AdonisTrainerTuner):
    """Nested CV run K times over K disjoint training-group subsets, with aggregation.

    Each of the K subsets is a disjoint chunk of the training groups. Within each
    outer CV fold, only groups belonging to the current subset are used for training;
    the test set is untouched. Metrics are aggregated across the K runs to estimate
    data-variability uncertainty.
    """

    def run_nested_cv_fractional(
        self,
        n_subsets: int,
        selected_models=None,
        base_ablation_tag: str | None = None,
        do_save_results: bool = True,
    ):
        assert n_subsets >= 2
        original_outer_cv = self.config.cross_validation["outer_cv"]
        original_tag = self.ablation_tag
        n_splits = original_outer_cv.n_splits
        seed = self.config.params.get("random_seed", 0)

        per_subset_results: dict[int, dict] = {}

        try:
            for k in range(n_subsets):
                self.config.cross_validation["outer_cv"] = GroupKFoldFractional(
                    n_splits=n_splits,
                    n_subsets=n_subsets,
                    subset_idx=k,
                    random_state=seed,
                )
                self.ablation_tag = (
                    f"{base_ablation_tag}/subset_{k}" if base_ablation_tag
                    else f"subset_{k}"
                )
                logger.info("=== Fractional subset %d/%d — tag=%s ===",
                            k + 1, n_subsets, self.ablation_tag)

                _, results_sklearn, _, _, _ = self.run_nested_cross_validation_sklearn(
                    selected_models=selected_models,
                    do_save_results=do_save_results,
                )
                per_subset_results[k] = results_sklearn
        finally:
            self.config.cross_validation["outer_cv"] = original_outer_cv
            self.ablation_tag = original_tag

        aggregated = self._aggregate_across_subsets(per_subset_results, n_subsets)

        if do_save_results:
            self._save_fractional_aggregate(
                aggregated=aggregated,
                per_subset_results=per_subset_results,
                n_subsets=n_subsets,
                base_ablation_tag=base_ablation_tag,
            )

        return per_subset_results, aggregated

    def _aggregate_across_subsets(self, per_subset_results, n_subsets):
        """Two aggregations per model & metric:
        - by_fold  (primary): for each outer fold, mean across K subsets →
          mean/std across the n_folds per-fold means. std is directly
          comparable to --do-cv (both are fold-to-fold variability over the
          same GroupKFold test folds).
        - by_subset: for each subset, per-subset mean over its folds →
          mean/std across the K subset-means. Captures patient-mix variance.
        """
        models = list(per_subset_results[0].keys())
        metrics = list(per_subset_results[0][models[0]]["test"].keys())
        agg = {}
        for model in models:
            agg[model] = {}
            for metric in metrics:
                # (K subsets, n_folds) matrix of per-fold values
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

    def _save_fractional_aggregate(self, aggregated, per_subset_results,
                                    n_subsets, base_ablation_tag):
        results_path = get_results_path(self.config, base_ablation_tag)
        for model, model_agg in aggregated.items():
            payload = {
                "n_subsets": n_subsets,
                "per_subset_metrics": {
                    k: per_subset_results[k][model]["test"] for k in range(n_subsets)
                },
                "aggregated": model_agg,
            }
            path = os.path.join(results_path, f"{model}_sklearn_aggregated.pkl")
            with open(path, "wb") as fh:
                pickle.dump(payload, fh)
            logger.info("Saved aggregate for '%s' → %s", model, path)
