import logging

import sklearn
sklearn.set_config(enable_metadata_routing=True)

from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, FixedThresholdClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from adonis_ml.machine_learning.cross_validation import cross_validate
from adonis_ml.machine_learning.calibration import GroupCalibratedClassifierCV, FixedCalibratorClassifier, _FrozenCalibrator

import torch

logger = logging.getLogger(__name__)


class GridSearchThresholdCV(TunedThresholdClassifierCV):
    """Three-step tuning pipeline: hyperparameter search → calibration → threshold tuning.

    Step 1 — GridSearchCV (refit=False): finds best hyperparameters without refitting.
    Step 2 — GroupCalibratedClassifierCV (ensemble=False): fits a single calibrator on
              out-of-fold predictions using the same CV splits. The calibrator is fixed
              after this step.
    Step 3 — TunedThresholdClassifierCV: tunes the decision threshold using
              FixedCalibratorClassifier, which refits only the base model per fold while
              keeping the calibrator frozen. Final refit also refits only the base model.

    All three steps share the same pre-computed CV splits (from inner_cv).
    """

    def __init__(
        self,
        estimator,
        param_grid=None,
        cv=None,
        selection_score_grid_search=None,
        selection_score_threshold_tuning="f1",
        thresholds=1000,
        refit=True,
        n_jobs=1,
        batch_size=-1,
        return_train_score=False,
        scoring_metrics=None,
        calibration_method=None,
    ):
        super().__init__(
            estimator,
            cv=cv,
            scoring=selection_score_threshold_tuning,
            thresholds=thresholds,
            refit=refit,
            n_jobs=n_jobs,
        )

        self.param_grid = param_grid
        self.selection_score_grid_search = selection_score_grid_search
        self.selection_score_threshold_tuning = selection_score_threshold_tuning
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.return_train_score = return_train_score
        self.scoring_metrics = scoring_metrics
        self.calibration_method = calibration_method

        if self.return_train_score:
            self.cv_train_score = cv

    def fit(self, X, y, **params):
        groups = params.get("groups")
        sample_weight = params.get("sample_weight")
        window_ids = params.get("window_ids")

        # ── Step 0: pre-compute shared CV splits ──────────────────────────────
        cv_splits = list(self.cv.split(X, y, groups))

        # ── Step 1: hyperparameter tuning (no refit) ──────────────────────────
        best_estimator = clone(self.estimator)

        if self.param_grid is not None:
            grid_search = GridSearchCV(
                estimator=clone(self.estimator),
                param_grid=self.param_grid,
                cv=cv_splits,
                scoring=self.selection_score_grid_search,
                refit=False,
                n_jobs=self.n_jobs,
            )
            grid_search.fit(X, y, sample_weight=sample_weight)
            check_is_fitted(grid_search)
            self.best_params_ = grid_search.best_params_
            best_estimator = clone(self.estimator).set_params(**self.best_params_)
            logger.info(
                "GridSearchCV best params: %s  score: %.4f",
                self.best_params_,
                grid_search.best_score_,
            )

        # Store uncalibrated base for return_train_score path
        base_estimator = clone(best_estimator)

        # ── Step 2: calibration ───────────────────────────────────────────────
        calibrator = None
        if self.calibration_method is not None:
            logger.info(
                "Fitting calibrator (method='%s') on %d splits",
                self.calibration_method,
                len(cv_splits),
            )
            calibrated = GroupCalibratedClassifierCV(
                estimator=clone(best_estimator),
                method=self.calibration_method,
                cv=cv_splits,
                ensemble=False,
            )
            calibrated.fit(X, y, groups=groups, sample_weight=sample_weight)
            calibrator = _FrozenCalibrator(calibrated.calibrated_classifiers_[0].calibrators[0])
            logger.info("Calibrator fitted: %s", calibrator)

        # ── Step 3: threshold tuning ──────────────────────────────────────────
        if calibrator is not None:
            threshold_estimator = FixedCalibratorClassifier(
                estimator=clone(best_estimator),
                calibrator=calibrator,
            )
        else:
            threshold_estimator = clone(best_estimator)

        self.estimator = threshold_estimator
        self.cv = cv_splits  # already pre-computed — no groups needed by TunedThresholdClassifierCV

        threshold_params = {}
        if sample_weight is not None:
            threshold_params["sample_weight"] = sample_weight
        if window_ids is not None:
            threshold_params["window_ids"] = window_ids

        super().fit(X, y, **threshold_params)
        logger.info("Best threshold: %.4f", self.best_threshold_)

        self.free_gpu_memory()

        # ── return_train_score (uncalibrated base, no groups routing needed) ──
        if self.return_train_score:
            nested_estimator = FixedThresholdClassifier(
                base_estimator,
                threshold=self.best_threshold_,
            )
            sklearn_dict = cross_validate(
                nested_estimator,
                X,
                y=y,
                groups=groups,
                scoring=self.scoring_metrics,
                cv=self.cv_train_score,
                window_ids=window_ids,
                params=None,
                return_estimator=False,
            )
            self.cv_results_ = sklearn_dict
            self.free_gpu_memory()

        return self

    def free_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
