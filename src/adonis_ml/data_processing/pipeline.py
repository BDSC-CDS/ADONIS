"""Data-processing pipeline for the ADONIS-ML project.

Architecture overview
---------------------
  The pipeline runs in three ordered Steps (matching the README):

    Step 1 — I/O-bound cache population (n_io_jobs workers, default 16)
      Each loky worker calls extractor.extract_single_to_cache(patient_id),
      which reads all variable parquet files concurrently via an inner
      ThreadPoolExecutor, applies the full per-patient processing pipeline
      (demographics, ventilation, nSOFA/VIS/sf-ratio, delta signals, time
      differences), and writes a single fully-processed parquet cache file.
      Because the Extractor carries all preloaded data in memory, workers
      are capped at n_io_jobs to avoid excessive memory duplication.

    Step 2 — CPU-bound feature computation (n_jobs workers, default 64)
      Workers are pure CPU: they read the Step-1 cache file, run hourly
      aggregation, extract all feature groups (spikes/drops, trends, std,
      kurtosis/skewness, entropy) using single-patient variants that avoid
      groupby overhead, then compute upstream rolling windows.  No DB
      connections or large preloaded data are passed to these workers.
      Patients are processed in chunks of `chunk_size` (default 200);
      a checkpoint is saved every `checkpoint_freq` chunks.

    Step 3 — Time-window selection + case/control matching (non-parallel)
      Per-hospital post-processing pass over the Step-2 output. Runs
      `DatasetSelector.select_dataset` (and optionally
      `select_nsofa_dataset` when `extract_nsofa` is set) to carve each
      patient's processed timeseries into the per-hospital modelling
      dataset consumed downstream by the ML pipeline.

Config keys (under experiment.parallel)
----------------------------------------
  n_jobs       CPU-bound Step-2 worker count   (default 64)
  n_io_jobs    I/O-bound Step-1 worker count   (default 16)
  chunk_size   Patients per Step-2 dispatch    (default 200)

Checkpoint / resume
-------------------
  If data_total.parquet already exists in the checkpoint directory and
  extract_data is False, the pipeline skips extraction (Steps 1 and 2)
  and goes straight to Step 3 (time-window selection).  Columns present
  in the parquet but disabled in the current config are dropped
  automatically.

"""
from __future__ import annotations
import json
import logging
import os
import pickle
import warnings

import pandas as pd
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

from adonis_ml.data_processing.add_features import FeatureExtractor
from adonis_ml.data_processing.data_loader import BioMedITDataLoader
from adonis_ml.data_processing.dataset_selector import (
    ContinuousTimeWindowSelector,
    ResampleTimeWindowSelector,
    SepsisRiskSelector,
)
from adonis_ml.data_processing.patient_processor import (
    compute_patient_features,
    populate_patient_cache,
)
from adonis_ml.data_processing.upstream_features import combine_features_upstream
from adonis_ml.utils import AdonisConfig, check_column_names_and_ranges

requests.packages.urllib3.disable_warnings()

logger = logging.getLogger(__name__)


class DataProcessingPipeline:
    """Orchestrates data extraction, feature computation, and time-window selection.

    Instantiate with an AdonisConfig, then call run().  The environment field
    in the config selects the execution mode automatically.
    """

    def __init__(
        self,
        config: AdonisConfig,
        debug_n_patients: int | None = None,
        debug_hospital: str | None = None,
    ):
        self.config = config
        self.debug_n_patients = debug_n_patients
        self.debug_hospital = debug_hospital
        self._parse_args()

        if self.environment == "BioMedIT":
            self.data_loader = BioMedITDataLoader(self.config)
        else:
            raise NotImplementedError(f"Unknown environment: {self.environment}")

        path_to_runtime_checks = os.path.join(
            self.path_to_config, "config_runtime_checks.json"
        )
        with open(path_to_runtime_checks) as f:
            self.config_runtime_checks = AdonisConfig(json.load(f))

        check_column_names_and_ranges(
            self.data_loader.get_demog(),
            self.config_runtime_checks.demographics["columns"],
        )

        # Build the base list-of-variables mapping used by FeatureExtractor.
        # Feature-specific sub-lists (spikes_drops, trends, …) are appended
        # during extraction and returned alongside the feature DataFrame.
        self.list_of_variables = {
            group_name: [] for group_name in self.variables_to_extract.keys()
        }
        self.list_of_variables["demographics"] = ["ga_weeks", "sex"]

        for group_name, variables in self.variables_to_extract.items():
            for variable_name, variable_info in variables.items():
                if variable_info["extract"]:
                    if group_name in ["vitals"]:
                        self.list_of_variables[group_name].append(
                            variable_name + "_median"
                        )
                    if group_name in ["observations", "labs"]:
                        self.list_of_variables[group_name].append(
                            variable_name + "_median"
                        )
                    if group_name in ["observations"]:
                        self.list_of_variables[group_name].append(
                            variable_name + "_diff"
                        )

        if self.resampling_strategy == "resampling":
            self.dataset_selector = ResampleTimeWindowSelector(
                self.config, self.data_loader.get_demog()
            )
        elif self.resampling_strategy == "continuous":
            self.dataset_selector = ContinuousTimeWindowSelector(
                self.config, self.data_loader.get_demog()
            )
        elif self.resampling_strategy == "sepsis":
            self.dataset_selector = SepsisRiskSelector(
                self.config, self.data_loader.get_demog()
            )
        else:
            raise NotImplementedError(
                f"Unknown resampling strategy: {self.resampling_strategy}"
            )

        logger.info(
            "DataProcessingPipeline ready — environment=%s, patients=%d",
            self.environment,
            len(self.data_loader.stay_ids),
        )

    def _parse_args(self):
        self.environment = self.config.environment

        if self.environment == "BioMedIT":
            user = os.getlogin()
            self.config.paths["checkpoint"] = self.config.paths["checkpoint"].replace(
                "$USER", user
            )

        self.project_name = self.config.experiment["tags"]["project"]
        self.experiment_type = self.config.experiment["type"]
        self.extract_data = self.config.experiment["extract_data"]
        self.extract_hero = self.config.experiment["extract_hero"]
        self.extract_nsofa = self.config.experiment["extract_nsofa"]
        self.resampling_strategy = self.config.experiment["time_window_selection"][
            "resampling_strategy"
        ]

        self.checkpoint_freq = self.config.experiment["logging"]["checkpoint_freq"]

        self.max_gestational_age = self.config.experiment["params"][
            "max_gestational_age"
        ]
        self.upstream_period = self.config.experiment["params"]["upstream_period"]

        self.hospitals = self.config.experiment["hospitals"]
        self.features_to_extract = self.config.experiment["features"]
        self.variables_to_extract = self.config.experiment["variables"]
        self.patient_selection = self.config.experiment["patient_selection"]

        self.path_to_checkpoint = self.config.paths["checkpoint"]
        self.path_to_config = self.config.paths["config"]

        parallel_cfg = self.config.experiment.get("parallel", {})
        self.n_jobs = parallel_cfg.get("n_jobs", 16)
        self.n_io_jobs = parallel_cfg.get("n_io_jobs", 16)
        self.chunk_size = parallel_cfg.get("chunk_size", 500)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run the full pipeline: extract → select time windows."""
        os.makedirs(self.path_to_checkpoint, exist_ok=True)
        vars_file = os.path.join(self.path_to_checkpoint, "list_variables.pkl")

        if self.environment == "BioMedIT":
            included_hospitals = [h for h, info in self.hospitals.items() if info["include"]]
            hospital_checkpoint_files = {
                h: os.path.join(self.path_to_checkpoint, f"data_total_{h}.parquet")
                for h in included_hospitals
            }
            all_exist = all(os.path.isfile(f) for f in hospital_checkpoint_files.values())

            if not all_exist or self.extract_data:
                _, list_of_variables = self._run_parallel()
                if list_of_variables:
                    logger.info("Saving list_of_variables")
                    with open(vars_file, "wb") as f:
                        pickle.dump(list_of_variables, f)
                else:
                    logger.info("No new patients processed — existing list_of_variables unchanged.")
            else:
                logger.info("All per-hospital checkpoints exist — skipping Step 2.")
                with open(vars_file, "rb") as f:
                    list_of_variables = pickle.load(f)

            # ===== Step 3 — time-window selection + case/control matching =====
            for hospital_name in included_hospitals:
                hospital_file = hospital_checkpoint_files[hospital_name]
                if not os.path.isfile(hospital_file):
                    logger.error(
                        "Hospital checkpoint not found for %s — skipping select_dataset.",
                        hospital_name,
                    )
                    continue
                logger.info("Loading data for hospital %s...", hospital_name)
                df_hospital = pd.read_parquet(hospital_file)
                if all_exist and not self.extract_data:
                    df_hospital, list_of_variables = self._validate_checkpoint(
                        df_hospital, list_of_variables
                    )

                if self.extract_nsofa:
                    logger.info("Running select_nsofa_dataset for %s...", hospital_name)
                    self.dataset_selector.select_nsofa_dataset(df_hospital, hospital_name)

                logger.info(
                    "Running select_dataset for %s (%d rows)...", hospital_name, len(df_hospital)
                )
                self.dataset_selector.select_dataset(df_hospital, hospital_name)
                
                del df_hospital

        else:
            checkpoint_file = os.path.join(self.path_to_checkpoint, "data_total.parquet")
            if not os.path.isfile(checkpoint_file) or self.extract_data:
                df_total, list_of_variables = self._run_sequential()
                logger.info("Saving final dataset — %d rows", len(df_total))
                df_total.to_parquet(checkpoint_file)
                if list_of_variables:
                    logger.info("Saving list_of_variables")
                    with open(vars_file, "wb") as f:
                        pickle.dump(list_of_variables, f)
            else:
                logger.info("Loading existing dataset from checkpoint")
                df_total = pd.read_parquet(checkpoint_file)
                with open(vars_file, "rb") as f:
                    list_of_variables = pickle.load(f)
                df_total, list_of_variables = self._validate_checkpoint(df_total, list_of_variables)

            # ===== Step 3 — time-window selection + case/control matching =====
            for hospital_name, hospital_info in self.hospitals.items():
                if hospital_info["include"]:
                    if self.extract_nsofa:
                        self.dataset_selector.select_nsofa_dataset(df_total, hospital_name)
                    self.dataset_selector.select_dataset(df_total, hospital_name)


    # ------------------------------------------------------------------
    # Sequential path
    # ------------------------------------------------------------------

    def _run_sequential(self) -> tuple[pd.DataFrame, dict]:
        """Sequential batch path.

        Iterates over the data loader's batches, extracts features, computes
        upstream rolling windows, and checkpoints every `checkpoint_freq` batches.
        """
        feature_extractor = FeatureExtractor(
            self.config, self.data_loader.get_demog(), self.list_of_variables
        )

        df_total = pd.DataFrame()
        list_of_variables: dict = {}
        checkpoint_file = os.path.join(self.path_to_checkpoint, "data_total.parquet")

        for i, (df_min, df_hour) in enumerate(
            tqdm(self.data_loader, desc="Batch processing")
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df, list_of_variables = feature_extractor.extract(df_min, df_hour)

            df = combine_features_upstream(
                df,
                list_of_variables,
                self.upstream_period,
                self.features_to_extract,
            )

            df_total = pd.concat([df_total, df])

            if i % self.checkpoint_freq == 0:
                logger.info("Checkpoint at batch %d", i)
                df_total.to_parquet(checkpoint_file)

        return df_total, list_of_variables

    # ------------------------------------------------------------------
    # Parallel path (BioMedIT)
    # ------------------------------------------------------------------

    def _run_parallel(self) -> tuple[pd.DataFrame, dict]:
        """Parallel path for BioMedIT (Steps 1 and 2).

        Step 1 — populate per-patient processed parquet cache.
            I/O-bound: n_io_jobs loky workers, each running
            Extractor.extract_single_to_cache() which uses an inner
            ThreadPoolExecutor to read all variable files concurrently.

        Step 2 — compute features per patient.
            CPU-bound: n_jobs loky workers reading from the Step-1 cache.
            Patients are dispatched in chunks of chunk_size; a checkpoint
            is saved every checkpoint_freq chunks.
        """
        patient_ids = list(self.data_loader.stay_ids)

        if self.debug_hospital is not None:
            demog = self.data_loader.get_demog()
            valid = set(demog.loc[demog["hospital"] == self.debug_hospital, "patient_id"])
            patient_ids = [pid for pid in patient_ids if pid in valid]
            logger.info("Debug: restricted to hospital %s — %d patients", self.debug_hospital, len(patient_ids))

        if self.debug_n_patients is not None:
            patient_ids = patient_ids[:self.debug_n_patients]
            logger.info("Debug: capped to first %d patients", len(patient_ids))

        n_patients = len(patient_ids)
        extractor = self.data_loader.extractor

        # Discover log file so workers can append to it too
        log_file = next(
            (
                h.baseFilename
                for h in logging.getLogger().handlers
                if isinstance(h, logging.FileHandler)
            ),
            None,
        )

        # ===== Step 1 — I/O-bound cache population =====
        # Check which patients already have a cache file in the main process —
        # no workers needed for a simple os.path.exists check.
        uncached = [pid for pid in patient_ids if not os.path.exists(extractor._processed_cache_path(pid))]
        n_already_cached = n_patients - len(uncached)
        logger.info(
            "Step 1 — %d/%d patients already cached, processing %d with %d I/O workers",
            n_already_cached, n_patients, len(uncached), self.n_io_jobs,
        )

        if uncached:
            cache_results = list(tqdm(
                Parallel(n_jobs=self.n_io_jobs, backend="loky", verbose=0, return_as="generator")(
                    delayed(populate_patient_cache)(pid, extractor, log_file)
                    for pid in uncached
                ),
                total=len(uncached),
                desc="Step 1 — caching",
            ))
            n_newly_cached = sum(bool(r) for r in cache_results)
            logger.info(
                "Step 1 complete — %d/%d newly cached, %d total",
                n_newly_cached, len(uncached), n_already_cached + n_newly_cached,
            )
        else:
            logger.info("Step 1 complete — all patients already cached, skipping.")

        # ===== Step 2 — CPU-bound feature computation =====
        import pyarrow as pa
        import pyarrow.parquet as pq

        demog = self.data_loader.get_demog()

        # Build stay_id → hospital mapping for per-hospital file routing
        sid_to_hospital: dict[str, str] = (
            demog.set_index("stay_id")["hospital"].to_dict()
            if "stay_id" in demog.columns and "hospital" in demog.columns
            else {}
        )
        all_hospitals: list[str] = sorted(set(sid_to_hospital.values()))

        # Per-hospital checkpoint files
        hospital_checkpoint_files: dict[str, str] = {
            h: os.path.join(self.path_to_checkpoint, f"data_total_{h}.parquet")
            for h in all_hospitals
        }

        # Resume: collect already-processed stay_ids across all hospital checkpoint files
        already_processed_sids: set = set()
        for h, hfile in hospital_checkpoint_files.items():
            if os.path.exists(hfile):
                try:
                    sids = set(
                        pd.read_parquet(hfile, columns=["stay_id"])["stay_id"].unique()
                    )
                    already_processed_sids |= sids
                    logger.info("Found existing checkpoint for %s: %d stay_ids.", h, len(sids))
                except Exception as exc:
                    logger.warning("Could not read checkpoint for %s (%s) — ignoring.", h, exc)

        pid_to_sid = (
            demog.set_index("patient_id")["stay_id"].to_dict()
            if "stay_id" in demog.columns else {}
        )
        patient_ids_to_process = [
            pid for pid in patient_ids
            if pid_to_sid.get(pid) not in already_processed_sids
        ]
        n_skipped = len(patient_ids) - len(patient_ids_to_process)
        if n_skipped:
            logger.info(
                "Step 2 — skipping %d already-processed patients, %d remaining.",
                n_skipped, len(patient_ids_to_process),
            )

        if not patient_ids_to_process:
            logger.info("Step 2 — all patients already in checkpoint, nothing to do.")
            return pd.DataFrame(), {}

        logger.info(
            "Step 2 — computing features with %d CPU workers (chunk_size=%d, per-hospital streaming)",
            self.n_jobs, self.chunk_size,
        )

        # Per-hospital tmp files and writers; atomic rename at the end.
        tmp_files: dict[str, str] = {h: f + ".tmp" for h, f in hospital_checkpoint_files.items()}
        for tmp in tmp_files.values():
            if os.path.exists(tmp):
                os.remove(tmp)
                logger.warning("Removed stale temp file: %s", tmp)

        writers: dict[str, pq.ParquetWriter | None] = {h: None for h in all_hospitals}
        total_rows: dict[str, int] = {h: 0 for h in all_hospitals}
        list_of_variables: dict = {}

        try:
            # Copy existing row groups into the respective tmp files when resuming
            if already_processed_sids:
                for h, hfile in hospital_checkpoint_files.items():
                    if os.path.exists(hfile):
                        logger.info("Copying existing checkpoint row groups for %s...", h)
                        src = pq.ParquetFile(hfile)
                        writers[h] = pq.ParquetWriter(tmp_files[h], src.schema_arrow)
                        for i in range(src.num_row_groups):
                            writers[h].write_table(src.read_row_group(i))
                        total_rows[h] = src.metadata.num_rows
                        logger.info(
                            "Copied %d row groups (%d rows) for %s.",
                            src.num_row_groups, total_rows[h], h,
                        )

            chunks = [
                patient_ids_to_process[i: i + self.chunk_size]
                for i in range(0, len(patient_ids_to_process), self.chunk_size)
            ]

            with tqdm(total=len(patient_ids_to_process), desc="Step 2 — features") as pbar:
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_frames: list[pd.DataFrame] = []
                    for df, vars_out in Parallel(
                        n_jobs=self.n_jobs, backend="loky", verbose=0,
                        return_as="generator_unordered",
                    )(
                        delayed(compute_patient_features)(
                            pid,
                            extractor._processed_cache_path(pid),
                            self.variables_to_extract,
                            self.features_to_extract,
                            self.list_of_variables,
                            demog,
                            self.upstream_period,
                            self.config,
                            log_file,
                        )
                        for pid in chunk
                    ):
                        if df is not None and not df.empty:
                            chunk_frames.append(df)
                            if not list_of_variables:
                                list_of_variables = vars_out
                        pbar.update(1)

                    if chunk_frames:
                        chunk_df = pd.concat(chunk_frames, ignore_index=True)
                        chunk_df["hospital"] = chunk_df["stay_id"].map(sid_to_hospital)

                        chunk_rows_by_hospital: dict[str, int] = {}
                        for hosp, grp in chunk_df.groupby("hospital", dropna=True):
                            if hosp not in writers:
                                logger.warning(
                                    "Unknown hospital '%s' — skipping %d rows.", hosp, len(grp)
                                )
                                continue
                            table = pa.Table.from_pandas(
                                grp.reset_index(drop=True), preserve_index=False
                            )
                            if writers[hosp] is None:
                                writers[hosp] = pq.ParquetWriter(tmp_files[hosp], table.schema)
                            writers[hosp].write_table(table)
                            total_rows[hosp] += len(grp)
                            chunk_rows_by_hospital[hosp] = len(grp)

                        logger.info(
                            "Chunk %d/%d written — %d rows total, per-hospital: %s",
                            chunk_idx + 1, len(chunks), len(chunk_df), chunk_rows_by_hospital,
                        )
        finally:
            for w in writers.values():
                if w is not None:
                    w.close()

        for h, tmp in tmp_files.items():
            if os.path.exists(tmp):
                os.replace(tmp, hospital_checkpoint_files[h])
                logger.info(
                    "Hospital %s: %d total rows → %s", h, total_rows[h], hospital_checkpoint_files[h]
                )

        logger.info("Step 2 complete — per-hospital row totals: %s", total_rows)
        return pd.DataFrame(), list_of_variables

    # ------------------------------------------------------------------
    # Checkpoint validation
    # ------------------------------------------------------------------

    def _validate_checkpoint(
        self,
        df_total: pd.DataFrame,
        list_of_variables: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """Validate a loaded checkpoint against the current config.

        Asserts that all enabled variables are present and drops columns
        for variables that are now disabled.  Also prunes list_of_variables
        to match.
        """
        columns_names = df_total.columns.values
        columns_to_drop = []

        for group_name, variables in self.variables_to_extract.items():
            for variable_name, variable_info in variables.items():
                if variable_info["extract"]:
                    matching = [c for c in columns_names if variable_name in c]
                    if not matching:
                        logger.warning(
                            "Variable '%s' is enabled in config but absent from the checkpoint "
                            "(dropped by score cleanup, or extraction failed).",
                            variable_name,
                        )
                else:
                    columns_to_drop.extend(
                        c for c in columns_names if variable_name in c
                    )
                    for group, feature_names in list_of_variables.items():
                        list_of_variables[group] = [
                            f for f in feature_names if variable_name not in f
                        ]

        df_total.drop(columns=columns_to_drop, inplace=True)
        columns_names = df_total.columns.values

        for group, feature_names in list_of_variables.items():
            for feature_name in feature_names:
                if feature_name not in columns_names:
                    logger.warning(
                        "Feature '%s' from list_variables.pkl not found in checkpoint "
                        "(dropped by score cleanup, or extraction failed).",
                        feature_name,
                    )

        return df_total, list_of_variables

