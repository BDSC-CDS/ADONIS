import numpy as np
import torch
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

import sklearn
sklearn.set_config(enable_metadata_routing=True)


class GroupCalibratedClassifierCV(CalibratedClassifierCV):
    """CalibratedClassifierCV that accepts group-aware CV splitters.

    Accepts either a splitter object or a pre-computed list of (train, test)
    index pairs as `cv`.
    """

    def __init__(
        self,
        estimator,
        method="sigmoid",
        cv=None,
        n_jobs=1,
        ensemble=False,
    ):
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            ensemble=ensemble,
        )
        self.cv = cv
        self.set_fit_request(groups=True)

    def fit(self, X, y, sample_weight=None, groups=None):
        if hasattr(self.cv, "split"):
            self.cv = list(self.cv.split(X, y, groups))
        # else already a pre-computed list of (train, test) tuples
        super().fit(X, y, sample_weight)
        self.free_gpu_memory()
        return self

    def free_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# ============================================================
# Calibration curve utilities 
# ============================================================

def apply_prevalence_shift(y_prob: np.ndarray, prev_source: float, prev_target: float) -> np.ndarray:
    """Saerens–Latinne–Decaestecker (2002) prior-shift correction.

    Adjusts probabilities calibrated at *prev_source* to reflect a population
    with prevalence *prev_target*, assuming the class-conditional distributions
    are unchanged (label-shift / prior-shift assumption).
    """
    r = (prev_target * (1.0 - prev_source)) / (prev_source * (1.0 - prev_target))
    return r * y_prob / (r * y_prob + (1.0 - y_prob))


def wilson_interval(successes, total, alpha=0.05):
    """Wilson confidence interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan

    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z**2 / total

    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0, centre - margin), min(1, centre + margin)


def calibration_errors(prob_true, prob_pred, counts, epsilon=1e-15):
    """Return (ece, relative_ece, symmetric_relative_ece), weighted by bin count."""
    weights = counts / np.sum(counts)
    absolute_error = np.abs(prob_true - prob_pred)

    ece = np.sum(weights * absolute_error)

    relative_error = absolute_error / np.maximum(prob_pred, epsilon)
    relative_ece = np.sum(weights * relative_error)

    symmetric_relative_error = (
        2.0 * absolute_error / np.maximum(prob_true + prob_pred, epsilon)
    )
    symmetric_relative_ece = np.sum(weights * symmetric_relative_error)

    return ece, relative_ece, symmetric_relative_ece


def calibration_curve_custom(y_true, probabilities, bin_edges):
    """Calibration curve for arbitrary bin edges.

    Returns (prob_true, prob_pred, counts, positives), each a 1-D array
    over the non-empty bins in order.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_indices = np.digitize(probabilities, bin_edges, right=True)
    bin_indices[probabilities == bin_edges[-1]] = len(bin_edges) - 1

    prob_true, prob_pred, counts, positives = [], [], [], []
    for i in range(1, len(bin_edges)):
        mask = bin_indices == i
        n = int(np.sum(mask))
        if n == 0:
            continue
        k = int(np.sum(y_true[mask]))
        prob_true.append(k / n)
        prob_pred.append(float(np.mean(probabilities[mask])))
        counts.append(n)
        positives.append(k)

    return (
        np.asarray(prob_true),
        np.asarray(prob_pred),
        np.asarray(counts),
        np.asarray(positives),
    )


def _bin_edges_quantile(probabilities, n_bins):
    quantiles = np.quantile(probabilities, np.linspace(0, 1, n_bins + 1))
    return np.unique(quantiles)


def _bin_edges_linear(probabilities, n_bins, thresholds, threshold_region_factor):
    min_prob = np.min(probabilities)
    if thresholds is not None:
        thr_med = np.median(thresholds)
        upper = thr_med * threshold_region_factor
        return np.linspace(min_prob, upper, n_bins)

    max_prob = np.max(probabilities)
    upper = np.quantile(probabilities, 0.999)
    return np.concatenate(
        [np.linspace(min_prob, upper, n_bins), [max_prob + 1e-12]]
    )


def compute_calibration_curve(
    y_true,
    probabilities,
    n_bins=20,
    strategy="quantile",
    thresholds=None,
    threshold_region_factor=5,
    alpha=0.05,
):
    """Compute a binned calibration curve with Wilson intervals + scalar errors.

    Returns a dict with keys:
        prob_true, prob_pred       : per-bin observed frequency / mean predicted prob
        wilson_lower, wilson_upper : Wilson interval on prob_true, at level (1 - alpha)
        counts, positives          : per-bin sample and positive counts
        error_scores               : {brier, ce, ece, relative_ece, symmetric_relative_ece}
    """
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    if strategy == "quantile":
        bin_edges = _bin_edges_quantile(probabilities, n_bins)
    elif strategy == "linear":
        bin_edges = _bin_edges_linear(
            probabilities, n_bins, thresholds, threshold_region_factor
        )
    else:
        raise ValueError("strategy must be 'quantile' or 'linear'")

    prob_true, prob_pred, counts, positives = calibration_curve_custom(
        y_true, probabilities, bin_edges
    )

    lower, upper = [], []
    for k, n in zip(positives, counts):
        lo, hi = wilson_interval(k, n, alpha=alpha)
        lower.append(lo)
        upper.append(hi)

    brier = brier_score_loss(y_true, probabilities)
    probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    ce = log_loss(y_true, probabilities_clipped, labels=[0, 1])
    ece, relative_ece, symmetric_relative_ece = calibration_errors(
        prob_true, prob_pred, counts
    )

    return {
        "prob_true":    prob_true,
        "prob_pred":    prob_pred,
        "wilson_lower": np.asarray(lower),
        "wilson_upper": np.asarray(upper),
        "counts":       counts,
        "positives":    positives,
        "error_scores": {
            "brier":                  float(brier),
            "ce":                     float(ce),
            "ece":                    float(ece),
            "relative_ece":           float(relative_ece),
            "symmetric_relative_ece": float(symmetric_relative_ece),
        },
    }


class _FrozenCalibrator:
    """Non-sklearn wrapper so clone() cannot re-initialise the fitted calibrator."""

    def __init__(self, calibrator):
        self._calibrator = calibrator

    def predict(self, X):
        return self._calibrator.predict(X)


class FixedCalibratorClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a base estimator with a pre-fitted calibrator that is never refitted.

    During fit() only the base estimator is (re)fitted; the calibrator is kept
    fixed. This makes the object safe to clone: sklearn's clone() passes
    constructor params by reference, so the fitted calibrator is preserved
    across clones without being re-initialised.
    """

    def __init__(self, estimator, calibrator):
        self.estimator = estimator    # unfitted base (clone-safe constructor param)
        self.calibrator = calibrator  # pre-fitted, shared reference through clone()

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        # Mirror sklearn's _get_prediction_method dispatch: use decision_function
        # when available so the frozen calibrator receives the same scale it was
        # fit on. Otherwise fall back to predict_proba[:, 1].
        if hasattr(self.estimator_, "decision_function"):
            raw = self.estimator_.decision_function(X)
        else:
            raw = self.estimator_.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def free_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# ============================================================
# Decision-function wrapper
# ============================================================

class DecisionFunctionWrapper(ClassifierMixin, BaseEstimator):
    """Expose a raw pre-sigmoid score via .decision_function for models that
    would otherwise force CalibratedClassifierCV to fall back to predict_proba.

    Dispatch by estimator type:
      - XGBClassifier    → predict(output_margin=True)
      - LGBMClassifier   → predict(raw_score=True)
      - TabPFNClassifier → predict_logits(X)[:, 1] - predict_logits(X)[:, 0]
      - otherwise (MLP, TabICL, RF, kNN, ...) → logit(clip(predict_proba[:, 1]))

    For MLP / TabPFN / TabICL the recovered score is exactly the pre-sigmoid
    margin (or its softmax equivalent). For RF / kNN there is no true logit —
    predict_proba is a vote fraction — but the logit transform still gives
    Platt a well-behaved unbounded input (fits a power transform on the odds)
    instead of forcing sigmoid(a·p + b) on a bounded input.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kw):
        self.estimator_ = clone(self.estimator).fit(X, y, **kw)
        self.classes_ = self.estimator_.classes_
        return self

    def get_params(self, deep=True):
        # Expose inner estimator's params transparently (without the
        # 'estimator__' prefix) so existing param grids keep working
        # after this wrapper is inserted into the pipeline.
        params = super().get_params(deep=deep)
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params.setdefault(k, v)
        return params

    def set_params(self, **params):
        if not params:
            return self
        own, forward = {}, {}
        for k, v in params.items():
            if k == "estimator" or k.startswith("estimator__"):
                own[k] = v
            else:
                forward[k] = v
        if own:
            super().set_params(**own)
        if forward:
            self.estimator.set_params(**forward)
        return self

    def decision_function(self, X):
        est = self.estimator_

        try:
            from xgboost import XGBClassifier
            if isinstance(est, XGBClassifier):
                return est.predict(X, output_margin=True)
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier
            if isinstance(est, LGBMClassifier):
                return est.predict(X, raw_score=True)
        except ImportError:
            pass

        try:
            from tabpfn import TabPFNClassifier
            if isinstance(est, TabPFNClassifier):
                logits = est.predict_logits(X)
                return logits[:, 1] - logits[:, 0]
        except ImportError:
            pass

        p = np.clip(est.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)
