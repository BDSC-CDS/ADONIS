import logging
from time import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import get_scorer

logger = logging.getLogger(__name__)

def cross_validate(
    estimator,
    X,
    y=None,
    groups=None,
    scoring=None,
    cv=None,
    window_ids=None,
    eclipse_mask=None,
    params=None,
    return_estimator=False,
):
    """
    Custom cross-validation that mimics sklearn.cross_validate(),
    but computes subgroup metrics efficiently by reusing predictions.

    Parameters
    ----------
    estimator : sklearn estimator
        Model implementing fit/predict.
    scoring : str or callable or dict, optional
        Scorer(s) following sklearn API.
    cv : int, default=5
        Number of folds for StratifiedKFold.
    subgroup_col : str, default="sex"
        Column in X used for subgroup evaluation.
    return_estimator : bool, default=False
        Whether to return fitted estimators.

    Returns
    -------
    results : dict
        Same format as sklearn.cross_validate(), plus 'subgroup_results' DataFrame.
    """

    params = {} if params is None else params

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

    # Storage
    n_splits = cv.get_n_splits()
    results_predict = {fold: {} for fold in range(n_splits)}
    test_scores = {key: np.zeros(n_splits) for key in scoring.keys()}
    test_scores_eclipse = {key: np.zeros(n_splits) for key in scoring.keys()}
    test_scores_training_windows = {key: np.zeros(n_splits) for key in scoring.keys()}
    fit_times = np.zeros(n_splits)
    score_times = np.zeros(n_splits)
    estimators = []

    # CV loop
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        window_ids_test = window_ids[test_idx]

        routed_params = {}
        if bool(params):
            groups_train = params["groups"][train_idx]
            sample_weight_train = params["sample_weight"][train_idx]
            window_ids_train = params["window_ids"][train_idx]

            routed_params["groups"] = groups_train
            routed_params["window_ids"] = window_ids_train
            routed_params["sample_weight"] = sample_weight_train

        logger.info("Fold %d/%d — fit start", fold + 1, n_splits)
        start_fit = time()
        est = clone(estimator)
        est.fit(X_train, y_train, **routed_params)
        fit_times[fold] = time() - start_fit
        logger.info("Fold %d/%d — fit time: %.1fs", fold + 1, n_splits, fit_times[fold])

        start_score = time()
        
        y_prob = est.predict_proba(X_test)[:, 1]
        y_pred = est.predict(X_test)

        # Save predictions
        results_predict[fold]["train_idx"] = train_idx
        results_predict[fold]["test_idx"] = test_idx
        results_predict[fold]["y_true"] = y_test
        results_predict[fold]["y_prob"] = y_prob
        results_predict[fold]["y_pred"] = y_pred

        # Compute metrics excluding eclipse windows (main results)
        if eclipse_mask is not None:
            non_eclipse = ~eclipse_mask[test_idx]
            y_test_ne = np.asarray(y_test)[non_eclipse]
            y_pred_ne = y_pred[non_eclipse]
            y_prob_ne = y_prob[non_eclipse]
            window_ids_test_ne = window_ids_test[non_eclipse]
        else:
            y_test_ne, y_pred_ne, y_prob_ne, window_ids_test_ne = y_test, y_pred, y_prob, window_ids_test

        for key, scorer in scoring.items():
            if key == "recall_conservative":
                test_scores[key][fold] = scorer._score_func(y_test_ne, y_pred_ne, window_ids_test_ne)
            else:
                if scorer._response_method == "predict":
                    test_scores[key][fold] = scorer._score_func(y_test_ne, y_pred_ne, **scorer._kwargs)
                else:
                    test_scores[key][fold] = scorer._score_func(y_test_ne, y_prob_ne, **scorer._kwargs)

        # Compute metrics including eclipse windows
        for key, scorer in scoring.items():
            if key == "recall_conservative":
                test_scores_eclipse[key][fold] = scorer._score_func(y_test, y_pred, window_ids_test)
            else:
                if scorer._response_method == "predict":
                    test_scores_eclipse[key][fold] = scorer._score_func(y_test, y_pred, **scorer._kwargs)
                else:
                    test_scores_eclipse[key][fold] = scorer._score_func(y_test, y_prob, **scorer._kwargs)

        # Compute metrics restricted to the training windows (train_knn_0.1 == 1)
        mask = X_test["train_knn_0.1"] == 1
        y_test_ga = y_test[mask]
        y_pred_ga = y_pred[mask]
        y_prob_ga = y_prob[mask]
        window_ids_test_ga = window_ids_test[mask]

        for key, scorer in scoring.items():
            if key == "recall_conservative":
                test_scores_training_windows[key][fold] = scorer._score_func(y_test_ga, y_pred_ga, window_ids_test_ga) 
            else:
                if scorer._response_method == "predict":
                    test_scores_training_windows[key][fold] = scorer._score_func(y_test_ga, y_pred_ga, **scorer._kwargs)
                else:
                    test_scores_training_windows[key][fold] = scorer._score_func(y_test_ga, y_prob_ga, **scorer._kwargs)

        score_times[fold] = time() - start_score
        logger.info(
            "Fold %d/%d — scores: %s",
            fold + 1,
            n_splits,
            {k: f"{v[fold]:.4f}" for k, v in test_scores.items()},
        )

        if return_estimator:
            estimators.append(est)

    # Prepare output dictionary (same as sklearn.cross_validate)
    results = {}
    for k, v in test_scores.items():
        results[f"test_{k}"] = v
    for k, v in test_scores_eclipse.items():
        results[f"test_{k}_eclipse"] = v
    for k, v in test_scores_training_windows.items():
        results[f"test_{k}_training_windows"] = v
    results["results_predict"] = results_predict
    results["fit_time"] = fit_times
    results["score_time"] = score_times

    if return_estimator:
        results["estimator"] = estimators

    return results