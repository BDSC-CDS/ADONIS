"""Upstream rolling-window feature computation.

Two variants:
  combine_features_upstream  — batch path (groupby stay_id). Used by CHUV / _run_sequential.
  combine_upstream_patient   — single-patient path (no groupby). Used by Phase-2 workers.
"""
from __future__ import annotations
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _build_feature_list(list_of_variables: dict, features_to_extract: dict) -> list[str]:
    """Return the flat list of variable columns to roll over."""
    features = list(list_of_variables["vitals"])
    for feature_name, feature_info in features_to_extract.items():
        if feature_info.get("upstream"):
            features.extend(list_of_variables.get(feature_name, []))
            if feature_name == "spikes_drops":
                features.extend(list_of_variables.get(f"{feature_name}_extreme", []))
    return features


def combine_features_upstream(
    df: pd.DataFrame,
    list_of_variables: dict,
    upstream_period: int,
    features_to_extract: dict,
) -> pd.DataFrame:
    """Batch variant — keeps groupby('stay_id'). Logic unchanged from original pipeline."""
    list_of_features = _build_feature_list(list_of_variables, features_to_extract)

    for variable in list_of_features:
        new_cols = {}
        grouped = df.groupby("stay_id")[variable]

        new_cols[variable + "_avg_upstream"] = (
            grouped.rolling(window=upstream_period, min_periods=upstream_period // 2)
            .mean()
            .reset_index(0, drop=True)
        )

        for h in [48, 24, 12, 6]:
            new_cols[f"{variable}_avg_last{h}h"] = (
                grouped.rolling(window=h, min_periods=h // 2)
                .mean()
                .reset_index(0, drop=True)
            )

        for h in [6, 12, 24]:
            new_cols[f"{variable}_avg_last{h}hto{int(2 * h)}h"] = (
                grouped.shift(h)
                .rolling(window=h, min_periods=h // 2)
                .mean()
                .reset_index(0, drop=True)
            )

        for h in [48, 24, 12, 6]:
            new_cols[f"{variable}_sd_last{h}h"] = (
                grouped.rolling(window=h, min_periods=h // 2)
                .std()
                .reset_index(0, drop=True)
            )

        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df


def combine_upstream_patient(
    df: pd.DataFrame,
    list_of_variables: dict,
    upstream_period: int,
    features_to_extract: dict,
) -> pd.DataFrame:
    """Single-patient variant — no groupby on stay_id.

    Applies rolling windows to all feature columns simultaneously per window size,
    which is faster than column-by-column rolling.
    """
    list_of_features = _build_feature_list(list_of_variables, features_to_extract)

    # Filter to columns that actually exist in df
    list_of_features = [f for f in list_of_features if f in df.columns]
    if not list_of_features:
        return df

    logger.debug("Upstream rolling over %d feature columns", len(list_of_features))

    # Full upstream average
    avg_up = (
        df[list_of_features]
        .rolling(upstream_period, min_periods=upstream_period // 2)
        .mean()
    )
    avg_up.columns = [f"{v}_avg_upstream" for v in list_of_features]
    df = pd.concat([df, avg_up], axis=1)

    # Rolling average and std per window — one call per window across all columns
    for h in [48, 24, 12, 6]:
        avg = df[list_of_features].rolling(h, min_periods=h // 2).mean()
        avg.columns = [f"{v}_avg_last{h}h" for v in list_of_features]

        std = df[list_of_features].rolling(h, min_periods=h // 2).std()
        std.columns = [f"{v}_sd_last{h}h" for v in list_of_features]

        df = pd.concat([df, avg, std], axis=1)

    # Shifted averages
    for h in [6, 12, 24]:
        shifted_avg = (
            df[list_of_features].shift(h).rolling(h, min_periods=h // 2).mean()
        )
        shifted_avg.columns = [f"{v}_avg_last{h}hto{2 * h}h" for v in list_of_features]
        df = pd.concat([df, shifted_avg], axis=1)

    return df
