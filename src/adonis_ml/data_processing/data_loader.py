from __future__ import annotations
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from typing import Callable
import json

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from adonis_ml.data_processing.connector import GraphDBConnector
from adonis_ml.data_processing.extractor import Extractor
from adonis_ml.data_processing.demographics import demographics_dict
from adonis_ml.data_processing.utils import cached_extract, rename_hospitals, resolve_parquet_path

HOURLY_AGG_RULES = {
    "vitals":       ["median", "mean"],
    "observations": ["median", "diff"],
    "ventilations": ["median", "mean"],
    "incubation": ["median", "mean"],
}

DERIVED_SIGNAL_RULES = {
    "nSOFA":        {"agg": ["max"]},
    "VIS":          {"agg": ["max"]},
    "sf_ratio":     {"agg": ["median", "mean"]},
    "ventilation":  {"agg": ["max"]},
    "invasive":     {"agg": ["max"]},
}
 
PASSTHROUGH_COLUMNS = [
    "stay_id",
    "time_difference_minutes",
    "time_difference_hours",
    "time_difference_days",
    "ga_weeks",
    "sex",
]
 
BREAK_THRESHOLD = pd.Timedelta(hours=25)
MIN_ABX_DURATION = pd.Timedelta(days=5)


def aggregate_hourly_patient(
    df_min: pd.DataFrame, variables_to_extract: dict
) -> pd.DataFrame:
    """Single-patient hourly aggregation.

    One groupby split on time_difference_hours instead of N×M
    separate transform() calls. Used by Phase-2 workers — no groupby on stay_id.
    """
    # "diff" is not a valid pandas agg function (it's a transform); track diff
    # variables separately and compute them as max-min after the main agg.
    agg_dict: dict[str, list[str]] = {}
    diff_vars: list[str] = []

    for group_name, variables in variables_to_extract.items():
        agg_types = HOURLY_AGG_RULES.get(group_name, [])
        if not agg_types:
            continue
        for var_name, var_info in variables.items():
            if not var_info["extract"] or var_name not in df_min.columns:
                continue
            standard = [a for a in agg_types if a != "diff"]
            if standard:
                agg_dict.setdefault(var_name, []).extend(standard)
            if "diff" in agg_types and var_name not in diff_vars:
                diff_vars.append(var_name)

    for var_name, rules in DERIVED_SIGNAL_RULES.items():
        if var_name in df_min.columns:
            standard = [a for a in rules["agg"] if a != "diff"]
            if standard:
                agg_dict.setdefault(var_name, []).extend(standard)
            if "diff" in rules["agg"] and var_name not in diff_vars:
                diff_vars.append(var_name)

    grouped = df_min.groupby("time_difference_hours")

    if agg_dict:
        agg_result = grouped.agg(agg_dict)
        agg_result.columns = [f"{var}_{agg}" for var, agg in agg_result.columns]
        agg_result = agg_result.reset_index()
    else:
        agg_result = (
            df_min[["time_difference_hours"]].drop_duplicates().reset_index(drop=True)
        )

    # Compute diff columns as max-min per hour (matches original sequential pipeline)
    for var_name in diff_vars:
        diff_col = grouped[var_name].max() - grouped[var_name].min()
        diff_col.name = f"{var_name}_diff"
        agg_result = agg_result.merge(
            diff_col.reset_index(), on="time_difference_hours", how="left"
        )

    present_cols = [c for c in PASSTHROUGH_COLUMNS if c in df_min.columns]
    passthrough = (
        df_min[present_cols]
        .drop_duplicates(subset=["time_difference_hours"])
        .reset_index(drop=True)
    )
    return passthrough.merge(agg_result, on="time_difference_hours", how="left")


class AbstractDataLoader(ABC):
    """Iterable base class that loads demographics once, then yields
    time-series data in batches."""

    def __init__(self, config: dict):
        self.config = config
        self._parse_args()

        # Database connection
        self.conn = {}
        self._get_connection()

        # Load file-name index
        path = os.path.join(self.path_to_checkpoint, "file_names.parquet")
        self.df_file_names = cached_extract(
            path, self._load_filenames, force=self.extract_demographics
        )

        # Demographics extraction
        path = os.path.join(self.path_to_checkpoint, "demographics.parquet")
        self.demog = cached_extract(path, self._load_demog, force=self.extract_demographics)

        self.stay_ids = self.demog["stay_id"].values

    # ===================================================================
    # ITERATION PROTOCOL
    # ===================================================================

    def __len__(self) -> int:
        return (len(self.stay_ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._current_idx = 0  # Reset iteration
        return self

    def __next__(self):
        if self._current_idx >= len(self):
            raise StopIteration

        batch = self.stay_ids[
            self._current_idx
            * self.batch_size : (self._current_idx + 1)
            * self.batch_size
        ]
        self._current_idx += 1

        return self._load_data(batch)
    
    # ===================================================================
    # CONFIG PARSING
    # ===================================================================

    def _parse_args(self):
        self.extract_demographics = self.config.experiment["extract_demographics"]
        self.batch_size = self.config.experiment["params"]["batch_size"]
        self.max_gestational_age = self.config.experiment["params"]["max_gestational_age"]
        self.upstream_period = self.config.experiment["params"]["upstream_period"]
        self.min_length_stay_hours = self.config.experiment["params"]["min_length_stay_hours"]
        self.prediction_time = self.config.experiment["params"]["prediction_time"]
        self.supercontrols = self.config.experiment["params"]["supercontrols"]
        self.self_controls = self.config.experiment["params"]["self_controls"]
        self.nb_to_select = self.config.experiment["params"]["nb_to_select"]
        self.three_days = self.config.experiment["params"]["three_days"]
        self.days_to_exclude = self.config.experiment["params"]["days_to_exclude"]
        self.transition = self.config.experiment["params"]["transition"]
        self.hospitals = self.config.experiment["hospitals"]
        self.variables_to_extract = self.config.experiment["variables"]
        self.patients_to_select = self.config.experiment["patient_selection"]
        self.path_to_checkpoint = self.config.paths["checkpoint"]

    # ===================================================================
    # DEMOGRAPHICS: load + cache
    # ===================================================================

    def _load_demog(self) -> pd.DataFrame:
        path = os.path.join(self.path_to_checkpoint, "demographics_raw.parquet")
        demog_raw = cached_extract(path, self._load_demog_raw, force=self.extract_demographics)

        demog, excluded = self._apply_inclusion_criteria(demog_raw)

        # Save excluded
        path = os.path.join(self.path_to_checkpoint, "demographics_excluded.parquet")
        excluded.to_parquet(path)

        return demog
    
    def get_demog(self, hospital=None) -> pd.DataFrame:
        if hospital is None:
            return self.demog.copy()
        else:
            return self.demog[self.demog["hospital"] == hospital].copy()
        
    # ===================================================================
    # INCLUSION CRITERIA (static)
    # ===================================================================
    
    def _apply_inclusion_criteria(
        self, 
        demog: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all inclusion criteria sequentially, tracking per-criterion
        exclusion counts. Returns (included, excluded) DataFrames."""
 
        criteria = self._get_inclusion_criteria()
 
        included = pd.Series(True, index=demog.index)
        exclusion_log: dict[str, int] = {}
 
        for name, criterion_fn in criteria.items():
            mask = criterion_fn(demog)
            newly_excluded = included & ~mask
            exclusion_log[name] = int(newly_excluded.sum())
            included &= mask
 
        # Save exclusion log
        path = os.path.join(self.path_to_checkpoint, "exclusion_log.json")
        with open(path, "w") as f:
            json.dump(exclusion_log, f, indent=2)
 
        excluded = demog[~included].copy()
        demog = demog[included].copy()
        return demog, excluded
 
    def _get_inclusion_criteria(self) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
        """Return an ordered dict of {name: function} where each function
        takes a DataFrame and returns a boolean Series (True = keep).
        Subclasses can override to add domain-specific criteria."""
        return {
            "newborn_prediction": lambda d: d["incl_criteria_prediction"] == 1,
            "gestational_age": lambda d: d["ga_weeks"] < self.max_gestational_age,
        }
    
    def _load_filenames(self) -> None:
        pass
        
    # ===================================================================
    # ABSTRACT INTERFACE
    # ===================================================================

    @abstractmethod
    def _get_connection(self) -> None:
        pass

    @abstractmethod
    def _load_demog_raw(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _load_data(self, batch) -> pd.DataFrame:
        pass

    def _load_data_min(self, df, max_month=4):
        """
        This function loads data min-by-min by doing a preprocessing
        For example it helps for the calculation of the distances with add_drops_spikes() in add_features.py

        Args:
            - max_month (optionnal, int): if=2 then we keep the 2 first months
                                        if None and everything is kept
                                        if not specified max_month = 4 to align with max postnatal age of 44 weeks (in case of a 24 weeker)
            - false_control (bool): if True it drops patients with more than 5 days of abx

        Returns:
            - df (dataframe): useful vitals min-by-min for all patients

        """
        df.replace(np.inf, np.nan, inplace=True)
        df.replace(-np.inf, np.nan, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.merge(self.demog[["stay_id", "ga_weeks"]], on="stay_id", how="left")

        if max_month:
            # delete days after max months
            df = df.drop(
                index=df.loc[df["time_difference_hours"] >= max_month * 30 * 24].index
            )
            df.reset_index(drop=True, inplace=True)
            df = df.drop(
                index=df.loc[df["time_difference_hours"].isna()].index
            )  # added to compansate bug of 08_01, later solved

        # sort dataframe
        df = df.sort_values(by=["stay_id", "time_difference_minutes"])
        return df

    def _load_data_hour(self, df, max_month=None):
        """
        This function loads data hour-by-hour by doing a preprocessing
        For example it helps for the calculation of the distances with add_dist_mahalanobis() in add_features.py

        Args:
            - max_month (optionnal, int): if=2 then we keep the 2 first months
                                        if non specified then None and everything is kept
            - false_control (bool): if True it drops patients with more than 5 days of abx

        Returns:
            - df (dataframe): useful vitals hour-by-hour for all patients

        """
        df.replace(np.inf, np.nan, inplace=True)
        df.replace(-np.inf, np.nan, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.merge(
            self.demog[["stay_id", "ga_weeks", "sex"]], on="stay_id", how="left"
        )

        if max_month:
            # delete days after max_month months
            df = df.drop(
                index=df.loc[df["time_difference_hours"] >= max_month * 30 * 24].index
            )
            df.reset_index(drop=True, inplace=True)

        df = df.dropna(subset="time_difference_hours").copy()
        df["time_difference_hours"] = df["time_difference_hours"].astype(int)

        # Sort dataframe
        df = df.sort_values(by=["stay_id", "time_difference_hours"])

        return df



class BioMedITDataLoader(AbstractDataLoader):
    def __init__(self, config: dict):
        super().__init__(config)

        self.extractor = Extractor(self.config, self.conn, self.demog, self.df_file_names)

        # Load file-name index
        path = os.path.join(self.path_to_checkpoint, "hero.parquet")
        self.df_hero = cached_extract(
            path, self._load_hero, force=self.extract_demographics
        )

        # conn no longer needed — release the GraphDB socket
        self.conn = {}
        self.stay_ids = self.demog["patient_id"].unique()
        logger.info(
            "BioMedITDataLoader ready — %d unique patients", len(self.stay_ids)
        )

    def _parse_args(self):
        super()._parse_args()
        self.path_to_GraphDB_credentials = self.config.paths["GraphDB_credentials"]
        self.path_to_queries = self.config.paths["queries"]
        self.path_to_time_series = self.config.paths["time_series"]
        self.medications = self.config.experiment["medications"]

    def _get_connection(self):
        self.conn["GraphDB"] = GraphDBConnector(self.path_to_GraphDB_credentials)

    # ===================================================================
    # HeRO LOADER
    # ===================================================================

    def _load_hero(self):
        file_path = os.path.join(self.path_to_queries, "hero.txt")
        with open(file_path, "r", encoding="utf-8") as file:
            query = file.read()

        # Turn on inference
        self.conn["GraphDB"].conn.clearParameter("infer")
        self.conn["GraphDB"].conn.addParameter("infer", "false")
        try:
            df_hero = self.conn["GraphDB"].query(query)
        finally:
            # Turn on inference
            self.conn["GraphDB"].conn.clearParameter("infer")
            self.conn["GraphDB"].conn.addParameter("infer", "true")

        df_hero = df_hero.rename(columns={"value": "hero", "source": "source_hero"})

        # Keep only measurements from recognised source systems
        valid_sources = {"MV5_NAT", "internal register"}
        df_hero = df_hero[df_hero["source_hero"].isin(valid_sources)]

        df_hero["time"] = pd.to_datetime(df_hero["time"], utc=True)

        df_hero = df_hero.merge(
            self.demog[["stay_id", "adm_date"]], on="stay_id", how="inner"
        )

        df_hero["adm_date"] = pd.to_datetime(df_hero["adm_date"], format="mixed", utc=True)

        df_hero["time_difference_hours"] = (
            (df_hero["time"] - df_hero["adm_date"]).dt.total_seconds() / 3600
        ).astype(int)

        df_hero["hero"] = pd.to_numeric(df_hero["hero"], errors="coerce")
        df_hero = df_hero.dropna(subset=["hero"])

        # Aggregate to one row per (stay_id, hour): hero_max = max score in that hour
        df_hero = (
            df_hero.groupby(["stay_id", "time_difference_hours"], sort=False)
            .agg(hero_max=("hero", "max"), source_hero=("source_hero", "first"))
            .reset_index()
        )

        return (
            df_hero[["stay_id", "time_difference_hours", "hero_max", "source_hero"]]
            .sort_values(["stay_id", "time_difference_hours"])
            .reset_index(drop=True)
        )

    # ===================================================================
    # FILE LOADER
    # ===================================================================

    def _load_filenames(self):
        file_path = os.path.join(self.path_to_queries, "file_names.txt")
        with open(file_path, "r", encoding="utf-8") as file:
            query = file.read()

        # Turn on inference
        self.conn["GraphDB"].conn.clearParameter("infer")
        self.conn["GraphDB"].conn.addParameter("infer", "false")
        try:
            df_file_names = self.conn["GraphDB"].query(query)
        finally:
            # Turn on inference
            self.conn["GraphDB"].conn.clearParameter("infer")
            self.conn["GraphDB"].conn.addParameter("infer", "true")

        df_file_names = rename_hospitals(df_file_names, self.hospitals)
        df_file_names = df_file_names.dropna(subset="concept")
        df_file_names["concept"] = df_file_names["concept"].apply(lambda x: x.replace(" ", ""))
        df_file_names["label"] = pd.Series()

        expanded_rows = []
        for _, variables in self.variables_to_extract.items():
            for variable_name, variable in variables.items():
                matching_rows = df_file_names[
                    df_file_names["concept"] == variable["concept"]
                ].copy()
                matching_rows["label"] = variable_name
                expanded_rows.append(matching_rows)

        df_file_names = pd.concat(expanded_rows, ignore_index=True).copy()

        # Fix OxygenAdministration problem (CHUV)
        df_file_names.loc[
            df_file_names["file_name"].apply(lambda x: "OxygenAdministration" in x),
            "label",
        ] = "OxygenAdm"

        df_file_names = (
            df_file_names.dropna(subset="label")
            .pivot(index=["hospital", "patient_id"], columns="label", values="file_name")
            .reset_index()
        )

        return df_file_names

    # ===================================================================
    # INCLUSION CRITERIA
    # ===================================================================
 
    def _get_inclusion_criteria(self) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
        base = super()._get_inclusion_criteria()
        base.update({
            "data_availability": self._criterion_has_hr_data,
        })
        return base

    # -- Data availability criterion ------------------------------------
 
    def _criterion_has_hr_data(self, demog: pd.DataFrame) -> pd.Series:
        """Check which patients have a non-empty HR parquet file."""
        patient_ids = set(demog["patient_id"].values)
        file_index = self.df_file_names[
            self.df_file_names["patient_id"].isin(patient_ids)
        ]

        has_data = {}
        for _, row in tqdm(file_index.iterrows(), total=len(file_index), desc="Probing HR data"):
            path = resolve_parquet_path(
                hospital=row["hospital"],
                file_name=row.get("HR"),
                variable_name="HR",
                path_to_time_series=self.path_to_time_series,
                variables_to_extract=self.variables_to_extract,
            )
            if path is None or not os.path.isfile(path):
                has_data[row["patient_id"]] = False
                continue

            try:
                has_data[row["patient_id"]] = pd.read_parquet(path).shape[0] > 0
            except Exception:
                has_data[row["patient_id"]] = False

        return demog["patient_id"].map(has_data).fillna(False)
    
    # ===================================================================
    # DEMOGRAPHICS: extraction + merge
    # ===================================================================

    def _load_demog_raw(self) -> pd.DataFrame:
        # Read queries
        queries = self.conn["GraphDB"].read_queries(
            os.path.join(self.path_to_queries, "demographics")
        )

        # Execute queries
        dataframes = {}

        # Turn off inference
        self.conn["GraphDB"].conn.clearParameter("infer")
        self.conn["GraphDB"].conn.addParameter("infer", "false")
        try:
            for query_name, query in tqdm(queries.items(), desc="Extracting queries"):
                dataframes[query_name] = demographics_dict[query_name]().process(self.conn["GraphDB"].query(query))
        finally:
            # Turn on inference
            self.conn["GraphDB"].conn.clearParameter("infer")
            self.conn["GraphDB"].conn.addParameter("infer", "true")

        # Merge the dataframes on patient_ids (assuming patient_id is the common column)
        demog = dataframes["patient_info"]

        for name, df in dataframes.items():
            if name != "patient_info":
                try:
                    demog = demog.merge(df, on="patient_id", how="left")
                except:
                    demog = demog.merge(df, on="stay_id", how="left")

        # Process demographics
        demog = self._process_demographics(demog)
        
        return demog

    # ===================================================================
    # DEMOGRAPHICS: derived fields
    # ===================================================================

    def _process_demographics(self, demog: pd.DataFrame) -> pd.DataFrame:
        demog = self._rename_hospitals(demog)
        demog = self._compute_age_at_admission(demog)
        demog = self._compute_length_of_stay(demog)
        demog = self._compute_death_flags(demog)
        demog = self._compute_sepsis_timing(demog)
        demog = self._compute_sepsis_outcomes(demog)
        demog = self._compute_negative_outcomes(demog)
        demog = self._compute_postmenstrual_age(demog)
        demog = self._compute_inclusion_flags(demog)
        demog = self._compute_control_trajectories(demog)
        demog = self._load_and_merge_antibiotics(demog)
        demog = self._compute_antibiotic_timing(demog)
        demog["start_record"] = pd.to_datetime(0, format="mixed", utc=True)
        return demog
    
    # -- Hospital renaming ----------------------------------------------
 
    def _rename_hospitals(self, demog: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            info["original_name"]: name
            for name, info in self.hospitals.items()
        }
        demog["hospital"] = demog["hospital"].replace(rename_map)
        return demog
 
    # -- Age at admission -----------------------------------------------
 
    def _compute_age_at_admission(self, demog: pd.DataFrame) -> pd.DataFrame:
        demog = demog.dropna(subset=["birth_date"]).copy()

        demog["age_at_admission_days"] = (
            demog["adm_date"].dt.tz_convert("Europe/Zurich") - demog["birth_date"]
        ).dt.days
        return demog
 
    # -- Length of stay (NAT) -------------------------------------
 
    def _compute_length_of_stay(self, demog: pd.DataFrame) -> pd.DataFrame:
        # Filter to valid hospital-ward combinations
        ward_rows = []
        for hospital, info in self.hospitals.items():
            mask = (demog["hospital"] == hospital) & demog["ward"].isin(info["wards"])
            ward_rows.append(demog.loc[mask])
 
        df = (
            pd.concat(ward_rows, axis=0)[
                ["hospital", "patient_id", "stay_id", "start_date", "end_date", "ward"]
            ]
            .drop_duplicates()
            .sort_values(["stay_id", "start_date"])
            .reset_index(drop=True)
        )
 
        # Identify contiguous ward blocks per stay
        df["prev_end"] = df.groupby("stay_id")["end_date"].shift()
        df["is_continuous"] = df["start_date"] == df["prev_end"]
        df["block"] = (~df["is_continuous"].fillna(False)).groupby(df["stay_id"]).cumsum()
 
        # Keep only the first contiguous block
        first_block = df.groupby("stay_id")["block"].transform("min")
        df = df[df["block"] == first_block]
 
        df = (
            df.groupby("stay_id")
            .agg(adm_date_nat=("start_date", "min"), disc_date_nat=("end_date", "max"))
            .reset_index()
        )
 
        los = df["disc_date_nat"] - df["adm_date_nat"]
        # Add 1 to ensure that both admission and discharge dates are included, even if they don't span full calendar days
        # Example: los = 3 days means that there are at least 48 hours between timestamps, but not necessarily 72 (i.e., not full 3×24 hours)
        df["length_stay_nat"] = los.dt.days + 1
        df["length_stay_hours_nat"] = los.dt.days * 24 + los.dt.components.hours
 
        demog = demog.merge(df, on="stay_id", how="left")
        demog = demog.drop_duplicates(subset=["stay_id", "adm_date_nat", "disc_date_nat"])
        return demog
 
    # -- Death -----------------------------------------------------------
 
    def _compute_death_flags(self, demog: pd.DataFrame) -> pd.DataFrame:
        for col, adm, disc in [
            ("death_hospital", "adm_date", "disc_date"),
            ("death_nat", "adm_date_nat", "disc_date_nat"),
        ]:
            demog[col] = (
                (demog["death"] == 1)
                & demog["death_date"].notna()
                & (demog[adm] <= demog["death_date"])
                & (demog["death_date"] <= demog[disc])
            ).astype(int)
        return demog
 
    # -- LOS Sepsis ------------------------------------------------------
 
    def _compute_sepsis_timing(self, demog: pd.DataFrame) -> pd.DataFrame:
        has_hemo = demog["datetime_hemoculture"].notna()
        is_sepsis = demog["sepsis"] == 1
        demog["time_hours_sepsis"] = np.where(
            has_hemo & is_sepsis,
            (demog["datetime_hemoculture"] - demog["birth_date"]).dt.total_seconds() // 3600,
            np.nan,
        )
        return demog
 
    # -- Sepsis outcomes (death within 7/30 d, PNA at infection) -----------

    def _compute_sepsis_outcomes(self, demog: pd.DataFrame) -> pd.DataFrame:
        sepsis_mask = demog["datetime_hemoculture"].notna() & (demog["sepsis"] == 1)

        demog["days_sepsis_to_death"] = np.where(
            sepsis_mask & (demog["death_nat"] == 1),
            (demog["death_date"] - demog["datetime_hemoculture"]).dt.total_seconds() / 86400,
            np.nan,
        )
        demog["death_hospital_7d"] = np.where(
            sepsis_mask & (demog["death_hospital"] == 1) & (demog["days_sepsis_to_death"] <= 7),
            1,
            np.where(sepsis_mask, 0, np.nan),
        )
        demog["death_hospital_30d"] = np.where(
            sepsis_mask & (demog["death_hospital"] == 1) & (demog["days_sepsis_to_death"] <= 30),
            1,
            np.where(sepsis_mask, 0, np.nan),
        )
        demog["pna_at_infection_days"] = np.where(
            sepsis_mask,
            (demog["datetime_hemoculture"] - demog["birth_date"]).dt.total_seconds() / 86400,
            np.nan,
        )
        return demog

    # -- Negative outcomes (MNDS) ----------------------------------------

    def _compute_negative_outcomes(self, demog: pd.DataFrame) -> pd.DataFrame:
        for col in ("infection", "rop", "bpd", "pvl", "ivh"):
            demog[col] = 0
 
        outcome_cols = ["sepsis", "nec", "rop", "bpd", "pvl", "ivh", "death_hospital"]
        demog["neg_outcomes"] = demog[outcome_cols].sum(axis=1).clip(upper=1).astype(int)
        return demog
 
    # -- Postmenstrual age -----------------------------------------------
 
    def _compute_postmenstrual_age(self, demog: pd.DataFrame) -> pd.DataFrame:
        demog = demog.dropna(subset=["ga_weeks"]).copy()
 
        age = demog["age_at_admission_days"]
        full_weeks = (age / 7).astype(int)
        remaining_days = age - full_weeks * 7
 
        demog["pma_at_admission"] = full_weeks + remaining_days / 10 + demog["ga_weeks"]
        demog["pma_at_admission_days"] = (demog["pma_at_admission"] * 7).astype(int)
        demog["pma_at_admission_weeks"] = np.floor(demog["pma_at_admission"]).astype(int)
        return demog
 
    # -- Inclusion / control flags ---------------------------------------
 
    def _compute_inclusion_flags(self, demog: pd.DataFrame) -> pd.DataFrame:
        # General inclusion criteria: newborns
        # Newborns are defined as infants aged less than 28 days for those born at a gestational age ≥ 37 weeks,
        # and infants with a corrected age (gestational + postnatal age) < 44 weeks for those born < 37 weeks.
        ga = demog["ga_weeks"]
        age = demog["age_at_admission_days"]
 
        is_term_newborn = (ga >= 37) & (age < 28)
        is_preterm_newborn = (ga < 37) & (ga + age / 7 < 44)
 
        demog["incl_criteria_prediction"] = (is_term_newborn | is_preterm_newborn).astype(int)
        demog["incl_criteria_trajectories"] = (
            (demog["incl_criteria_prediction"] == 1) & (age <= 1)
        ).astype(int)
        return demog
 
    def _compute_control_trajectories(self, demog: pd.DataFrame) -> pd.DataFrame:
        # Inclusion criteria:
        # 1st criteria: admission in the participating NICU or PICU for more than 48 hours
        # 2nd criteria: metavision data are available is satisfied by definition given that the set of patients is taken from the adonis table on the datawarehouse
        # 3rd criteria: survival until hospital discharge
        # 4th criteria: absence of NEC and sepsis diagnosis
        demog["control_trajectories"] = (
            (demog["length_stay_hours_nat"] >= self.min_length_stay_hours)
            & (demog["death_hospital"] == 0)
            & (demog["sepsis"] == 0)
            & (demog["nec"] == 0)
        ).astype(int)
        return demog

    # -- Antibiotics -----------------------------------------------------
 
    def _load_and_merge_antibiotics(self, demog: pd.DataFrame) -> pd.DataFrame:
        path = os.path.join(self.path_to_checkpoint, "antibiotics.parquet")
        df_abx = cached_extract(path, self._extract_antibiotics, force=self.extract_demographics)
 
        demog = demog.merge(df_abx, on="stay_id", how="left")
        demog["time_to_cut_5d_abx"] = demog["time_to_cut_5d_abx"].fillna(pd.NaT)
        demog["to_cut_5d_abx"] = demog["to_cut_5d_abx"].fillna(0).astype(int)
        return demog
 
    def _compute_antibiotic_timing(self, demog: pd.DataFrame) -> pd.DataFrame:
        has_abx = demog["time_to_cut_5d_abx"].notna() & (demog["to_cut_5d_abx"] == 1)
        hours = (demog["time_to_cut_5d_abx"] - demog["adm_date"]).dt.total_seconds() // 3600
 
        demog["time_hours_abx"] = np.where(has_abx, hours, np.nan)
        # Remove 48 h before the first day of antibiotics
        demog["time_hours_abx"] = demog["time_hours_abx"].apply(
            lambda x: max(0, x - 48) if pd.notna(x) else x
        )
        return demog
 
    def _extract_antibiotics(self) -> pd.DataFrame:
        df = self._query_all_antibiotics()
 
        if df.empty:
            return pd.DataFrame(
                columns=["stay_id", "time_to_cut_5d_abx", "to_cut_5d_abx", "antibiotic_group_name"]
            )
 
        return self._find_long_antibiotic_sequences(df)
 
    def _query_all_antibiotics(self) -> pd.DataFrame:
        # Build VALUES entries and a lookup for group names
        values_rows = []

        for group_name, group_codes in self.medications.items():
            for terminology, codes in group_codes.items():
                for code in codes:
                    values_rows.append(f'    ( {code} "{terminology}" "{group_name}")')

        values_block = "\n".join(values_rows)

        query = f"""
            PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX sphn:   <https://biomedit.ch/rdf/sphn-schema/sphn#>
            PREFIX snomed: <http://snomed.info/id/>

            SELECT DISTINCT ?stay_id ?hospital ?start_date ?terminology ?antibiotic_group_name
            WHERE {{
                VALUES (?parent_code ?terminology ?antibiotic_group_name) {{
                    {values_block}
                }}

                ?patient a sphn:SubjectPseudoIdentifier ;
                        sphn:hasIdentifier ?patient_id ;
                        sphn:hasDataProvider/sphn:hasInstitutionCode/sphn:hasName ?hospital .

                ?stay a sphn:AdministrativeCase ;
                    sphn:hasSubjectPseudoIdentifier ?patient ;
                    sphn:hasIdentifier ?stay_id .

                ?adm a sphn:DrugAdministrationEvent ;
                    sphn:hasAdministrativeCase ?stay ;
                    sphn:hasStartDateTime ?start_date ;
                    sphn:hasEndDateTime ?end_date ;
                    sphn:hasDrug/sphn:hasActiveIngredient ?substance .

                ?substance a sphn:Substance ;
                        sphn:hasCode ?substance_code .

                ?substance_code rdf:type ?code .
                ?code rdfs:subClassOf ?parent_code .
            }}
        """

        df = self.conn["GraphDB"].query(query)

        if df.empty:
            return pd.DataFrame()
        
        # Ensure 'start_date' columns is in datetime format
        df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", utc=True)

        return df
 
    def _find_long_antibiotic_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify continuous antibiotic administrations >= 5 days."""
        results = []
 
        for (group_name, stay_id), stay_df in df.groupby(
            ["antibiotic_group_name", "stay_id"]
        ):
            stay_df = (
                stay_df.sort_values("start_date")
                .drop_duplicates(subset=["start_date"])
                .reset_index(drop=True)
            )
 
            # Tag continuous sequences
            gaps = stay_df["start_date"].diff().fillna(pd.Timedelta(0))
            stay_df["sequence_id"] = (gaps > BREAK_THRESHOLD).cumsum()
 
            sequences = stay_df.groupby("sequence_id")["start_date"].agg(["min", "max"])
            sequences["duration"] = sequences["max"] - sequences["min"]
 
            for _, row in sequences[sequences["duration"] >= MIN_ABX_DURATION].iterrows():
                results.append({
                    "stay_id": stay_id,
                    "time_to_cut_5d_abx": row["min"],
                    "to_cut_5d_abx": 1,
                    "antibiotic_group_name": group_name,
                })
 
        if not results:
            return pd.DataFrame(
                columns=["stay_id", "time_to_cut_5d_abx", "to_cut_5d_abx", "antibiotic_group_name"]
            )
 
        out = pd.DataFrame(results)
        out = out.sort_values(
            by=["to_cut_5d_abx", "time_to_cut_5d_abx"], ascending=[False, True]
        )
        return out.drop_duplicates(subset=["stay_id"])

    def _load_data(self, batch):
        raise NotImplementedError(
            "BioMedIT uses the two-phase parallel pipeline (_run_parallel); "
            "sequential batch loading is not supported."
        )
