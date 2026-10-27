"""Phase-1 and Phase-2 worker functions for the patient-level parallel pipeline.

Both functions are module-level so loky can serialize them cleanly.

Phase 1 — populate_patient_cache:
    Reads all variable parquet files concurrently (ThreadPoolExecutor),
    applies the full processing pipeline, writes a fully-processed
    per-patient cache file. I/O bound — run with n_io_jobs (16-32).

Phase 2 — compute_patient_features:
    Reads the single cached file, aggregates to hourly, computes all
    features and upstream rolling windows. CPU bound — run with n_jobs (64).
"""
from __future__ import annotations
import logging
import os
import sys
import time as _time
import warnings

import pandas as pd

from adonis_ml.data_processing.data_loader import aggregate_hourly_patient
from adonis_ml.data_processing.add_features import FeatureExtractor
from adonis_ml.data_processing.upstream_features import combine_upstream_patient


def _setup_worker_logger(log_file: str | None = None) -> None:
    """Configure logging in the worker process.

    Never writes to stdout — tqdm owns stdout and interleaved log lines
    corrupt the progress bar.  If a log_file is provided, appends INFO+
    records to it (safe on Linux via O_APPEND).  If no log_file is given,
    WARNING+ records go to stderr so real errors are still visible.
    """
    fmt = "%(asctime)s [PID:%(process)d] %(name)s - %(levelname)s - %(message)s"
    root = logging.getLogger()
    if root.handlers:
        return

    # Suppress pandas catastrophic-cancellation warning (expected for constant hourly signals)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*catastrophic cancellation.*")

    root.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
    else:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(sh)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def populate_patient_cache(patient_id: str, extractor, log_file: str | None = None) -> bool:
    """Phase-1 worker.

    Delegates entirely to Extractor.extract_single_to_cache which handles
    concurrent variable reads, merging, full processing pipeline, and writing
    the cache file.  Returns True on success, False if the patient is skipped.
    """
    _setup_worker_logger(log_file)
    try:
        return extractor.extract_single_to_cache(patient_id)
    except Exception as exc:
        logger.error(
            "Unhandled error caching patient %s: %s",
            patient_id, exc, exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def compute_patient_features(
    patient_id: str,
    cache_path: str,
    variables_to_extract: dict,
    features_config: dict,
    list_of_variables: dict,
    demog: pd.DataFrame,
    upstream_period: int,
    config,
    log_file: str | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Phase-2 worker.

    Reads the fully-processed per-patient cache file produced by Phase 1,
    then runs hourly aggregation, feature extraction, and upstream rolling
    windows.  Pure CPU — no Extractor, no DB connections, no large preloaded
    data passed to this worker.
    """
    _setup_worker_logger(log_file)
    t0 = _time.perf_counter()

    if not os.path.exists(cache_path):
        logger.warning("Cache file not found for patient %s — skipping", patient_id)
        return None, {}

    df_min = pd.read_parquet(cache_path)
    if df_min.empty:
        logger.warning("Empty cache file for patient %s — skipping", patient_id)
        return None, {}

    # Forward-fill delta columns — direct ffill (single patient, no groupby)
    for group_name, variables in variables_to_extract.items():
        if group_name not in ("vitals", "observations"):
            continue
        for var_name, var_info in variables.items():
            if var_info["extract"]:
                col = f"{var_name}_delta"
                if col in df_min.columns:
                    df_min[col] = df_min[col].ffill()

    logger.info(
        "Patient %s: read cache — %d minute rows", patient_id, len(df_min)
    )

    # Hourly aggregation — single groupby agg, no stay_id grouping
    df_hour = aggregate_hourly_patient(df_min, variables_to_extract)
    logger.info(
        "Patient %s: aggregated to %d hourly rows", patient_id, len(df_hour)
    )

    # Feature extraction — single-patient variants (no groupby on stay_id)
    feature_extractor = FeatureExtractor(config, demog, list_of_variables)
    df, list_of_vars_out = feature_extractor.extract(df_min, df_hour)
    logger.debug(
        "Patient %s: extracted feature groups %s",
        patient_id, list(list_of_vars_out.keys()),
    )

    # Upstream rolling windows — vectorized, no groupby
    df = combine_upstream_patient(df, list_of_vars_out, upstream_period, features_config)

    elapsed = _time.perf_counter() - t0
    logger.info(
        "Patient %s: computed features — %d rows, %.2fs",
        patient_id, len(df), elapsed,
    )
    return df, list_of_vars_out
