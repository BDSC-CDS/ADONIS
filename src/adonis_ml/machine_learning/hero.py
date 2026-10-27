import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from adonis_ml.machine_learning.utils import get_result_dict, print_results_summary, build_scorers


class HeroPredictor(ClassifierMixin, BaseEstimator):
    """
    Class for generating sepsis onset prediction as done by the HeRO score.

    The (continuous) hero score is thresholded by a fixed threshold to produce binary predictions.

    Probability measures are obtained by converting the hero score (odds-ratio) into a probability.
    Note that this probability is NOT calibrated.
    """

    def __init__(self, threshold=2.0, use_max=False, fixed_prevalence=None):
        self.threshold = threshold
        self.use_max = use_max
        self.fixed_prevalence = fixed_prevalence

    def fit(self, X, y):
        column_name = "hero_max" if self.use_max else "hero"
        X = pd.DataFrame(np.ones([y.shape[0], 1]), columns=[column_name])
        X, y = validate_data(self, X, y)

        if self.fixed_prevalence is not None:
            self.proba_baseline = np.array([self.fixed_prevalence])
        else:
            unique, counts = np.unique(y, return_counts=True)
            self.proba_baseline = counts[unique==1] / len(y)

        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        pred = (X > self.threshold).astype(int)
        return pred

    def predict_or(self, proba):
        """Convert predicted probability to odds ratio relative to baseline.

        Inverse of the Bayesian OR to probability formula used in predict_proba:
            OR = (P / (1-P)) / (baseline / (1-baseline))
        """
        check_is_fitted(self)
        proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
        odds_model = proba / (1 - proba)
        odds_baseline = float(self.proba_baseline) / (1 - float(self.proba_baseline))
        return odds_model / odds_baseline

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        pred_proba_positive_class = ((self.proba_baseline * X) / (1 + self.proba_baseline * (X - 1)))
        pred_proba_positive_class = np.clip(pred_proba_positive_class, 0, 1)
        pred_proba_negative_class = 1.0 - pred_proba_positive_class
        pred_proba = np.concatenate(
            [pred_proba_negative_class, pred_proba_positive_class],
            axis=-1,
        )
        assert (0.0 <= pred_proba).all() and (pred_proba <= 1.0).all()
        return pred_proba


def run_hero(
    df,
    config,
    return_estimator=True,
    threshold=2.0,
):
    from sklearn.model_selection import StratifiedGroupKFold

    # TODO implement (threshold) tuning option with nested CV
    estimator = HeroPredictor(threshold=threshold)

    hero_indices = ~df["hero_median"].isna()
    df_hero = df[hero_indices].copy()
    X = df_hero["hero_median"].values.reshape(-1, 1)
    y = df_hero["sepsis_time"].values
    groups = df_hero["ref_septic_patient"].values
    print(
        f"Class distribution for Hero scoring (1: case, 0: control): "
        f"{df_hero['sepsis_time'].value_counts()}"
    )
    sklearn_dict = cross_validate(
        estimator,
        X,
        y=y,
        groups=groups,
        scoring=config.metrics,
        cv=config.cross_validation["outer_cv"],  # StratifiedGroupKFold(n_splits=5), #
        return_estimator=return_estimator,
    )
    fit_model = sklearn_dict.pop("estimator")
    results_sklearn_model = get_result_dict(sklearn_dict)
    print_results_summary(results_sklearn_model)


def aggregate_hero(config, df_results_prob):
    df_results_agg = {}
    models = df_results_prob["model"].unique()

    for model in models:
        df = df_results_prob[df_results_prob["model"]==model]
        
        y_true_all = []
        y_pred_all = []

        for stay_id, df_stay in df.groupby("stay_id"):
            # At least 3 consecutive probabilities >= 0.6
            mask = (
                df_stay['y_pred']
                .eq(1)
                .rolling(3)
                .sum()
                .ge(3)
            )
            
            # Store prediction
            y_pred_stay = int(mask.any())
            y_true_stay = df_stay['y_true'].iloc[0]

            y_pred_all.append(y_pred_stay)
            y_true_all.append(y_true_stay)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        # Compute metrics
        scorers = build_scorers(config.metrics)

        df_results_model = {}
        for metric_name, scorer in scorers.items():
            df_results_model[metric_name] = scorer._score_func(y_true_all, y_pred_all, **scorer._kwargs)

        df_results_agg[model] = df_results_model

    return df_results_agg
