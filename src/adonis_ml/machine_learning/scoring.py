import pandas as pd
import numpy as np

from sklearn.metrics import make_scorer, recall_score, precision_score, roc_auc_score

def compute_y_pred_window(y_pred, window_ids):
    """
    Compute binary window-level predictions.
    y_pred_window = 1 if any prediction in the window is 1, else 0.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted labels (0 or 1) for each time step.
    window_ids : array-like of shape (n_samples,)
        Window identifier for each time step (same size as y_pred).

    Returns
    -------
    y_pred_window : pandas.Series
        Window-level predictions indexed by unique window_id.
    """
    y_pred = np.asarray(y_pred)
    window_ids = np.asarray(window_ids)

    # Create a DataFrame for grouping
    df = pd.DataFrame({
        "window_id": window_ids,
        "y_pred": y_pred
    })

    # Aggregate: 1 if any prediction in the window == 1
    y_pred_window = (
        df.groupby("window_id")["y_pred"]
        .transform("max")
    )

    return y_pred_window.values


def recall_conservative_score(y_true, y_pred, window_ids, sample_weight=None):
    y_pred_window = compute_y_pred_window(y_pred, window_ids)
    return recall_score(y_true, y_pred_window, sample_weight=sample_weight)
    

def recall_distance_score(y_true, y_pred, window_ids=None, sample_weight=None, target_recall=0.6):
    """
    Compute negative absolute distance between recall and target recall.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    target_recall : float, optional
        Desired recall value to compare against.

    Returns
    -------
    score : float
        Negative absolute distance from target recall.
        (Higher is better for model selection.)
    """
    if window_ids is None:
        recall = recall_score(y_true, y_pred, sample_weight=sample_weight)
    else:
        recall = recall_conservative_score(y_true, y_pred, window_ids=window_ids, sample_weight=sample_weight)
    distance = np.abs(recall - target_recall)
    return -distance


def precision_with_recall_constraint_score(y_true, y_pred, window_ids=None, sample_weight=None, min_recall=0.6, penalty=-1.0):
    """
    Return precision only if recall >= min_recall; otherwise return penalty.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    min_recall : float
        Minimum acceptable recall threshold.
    penalty : float
        Value to return if recall < min_recall (default = -1.0).

    Returns
    -------
    score : float
        Precision if recall >= min_recall, else penalty.
    """
    if window_ids is None:
        recall = recall_score(y_true, y_pred, sample_weight=sample_weight)
    else:
        recall = recall_conservative_score(y_true, y_pred, window_ids=window_ids, sample_weight=sample_weight)
    if recall < min_recall:
        return penalty
    return precision_score(y_true, y_pred, sample_weight=sample_weight)


def get_custom_scorer(scorer="recall_distance_scorer", **params):
    if scorer == "p_roc_auc":
        max_fpr = params["max_fpr"]
        return make_scorer(
            roc_auc_score,
            response_method="predict",
            greater_is_better=True,
            max_fpr=max_fpr
        )
    elif scorer == "roc_auc_weighted":
        return make_scorer(
            roc_auc_score,
            response_method="predict",
            greater_is_better=True,
        ).set_score_request(sample_weight=True)
    elif scorer == "specificity":
        return make_scorer(
            recall_score,
            response_method="predict",
            greater_is_better=True,
            average="binary",
            pos_label=0
        )
    elif scorer == "recall_conservative":
        return make_scorer(
            recall_conservative_score,
            response_method="predict",
            greater_is_better=True
        ).set_score_request(window_ids=True)
    elif scorer == "recall_distance_score":
        target_recall = params["target_recall"]
        return make_scorer(
            recall_distance_score,
            response_method="predict",
            greater_is_better=True,
            target_recall=target_recall
        ).set_score_request(window_ids=True, sample_weight=True)
    elif scorer == "precision_with_recall_constraint_score":
        min_recall = params["min_recall"]
        penalty = params["penalty"]
        return make_scorer(
            precision_with_recall_constraint_score,
            response_method="predict",
            greater_is_better=True,
            min_recall=min_recall,
            penalty=penalty
        ).set_score_request(window_ids=True, sample_weight=True)
    else:
        return scorer