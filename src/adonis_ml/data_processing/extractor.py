"""
Measurement extraction and processing pipeline.

Refactored for:
- Selective signal extraction (extract a subset of variables, not just all-or-nothing)
- Clearer separation of concerns (scoring logic, unit conversion, signal processing)
- Reduced code duplication across measurement subclasses
- Centralized constants and configuration
"""

from __future__ import annotations
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd

from adonis_ml.data_processing.connector import GraphDBConnector
from adonis_ml.utils import AdonisConfig
from adonis_ml.data_processing.utils import cached_extract, rename_hospitals, resolve_parquet_path
from adonis_ml.data_processing.scores import ScoreEngine
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


INVASIVE_VENTILATION_MODES = frozenset([
    "Invasive mechanical ventilation (regime/therapy)",
    "High frequency oscillatory ventilation (regime/therapy)",
])

DEFAULT_FIO2 = 21

# Prefix-based unit conversion factors (relative to base)
METRIC_PREFIXES = {
    "": 1,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
}

BASE_UNIT_CONVERSIONS = {
    "g": {"g": 1},
    "[IU]": {"[IU]": 1},
}


# ---------------------------------------------------------------------------
# UNIT CONVERSION UTILITIES
# ---------------------------------------------------------------------------

def split_unit(unit: str, prefixes=None, unit_conversions=None):
    """Parse a unit string into (prefix, base_unit)."""
    prefixes = prefixes or METRIC_PREFIXES
    unit_conversions = unit_conversions or BASE_UNIT_CONVERSIONS

    for p in sorted(prefixes.keys(), key=len, reverse=True):
        base = unit[len(p):]
        if unit.startswith(p) and base in unit_conversions:
            return p, base

    raise ValueError(f"Unknown unit: {unit}")


def convert_units_vectorized(
    df: pd.DataFrame,
    target_unit: str,
    value_col: str = "dose",
    unit_col: str = "unit",
    prefixes: dict = None,
    unit_conversions: dict = None,
) -> pd.DataFrame:
    """
    General unit conversion separating base unit and magnitude prefix.

    Converts values in `value_col` from source units in `unit_col` to `target_unit`.
    """
    prefixes = prefixes or METRIC_PREFIXES
    unit_conversions = unit_conversions or BASE_UNIT_CONVERSIONS

    target_prefix, target_base = split_unit(target_unit, prefixes, unit_conversions)
    target_prefix_factor = prefixes[target_prefix]

    parsed = df[unit_col].apply(lambda u: split_unit(u, prefixes, unit_conversions))
    src_prefix = parsed.apply(lambda x: x[0])
    src_base = parsed.apply(lambda x: x[1])

    src_prefix_factor = src_prefix.map(prefixes)

    base_factors = []
    for s in src_base:
        if target_base not in unit_conversions.get(s, {}):
            raise ValueError(f"Cannot convert {s} to {target_base}")
        base_factors.append(unit_conversions[s][target_base])

    df = df.copy()
    df[value_col] = df[value_col] * src_prefix_factor * base_factors / target_prefix_factor
    df[unit_col] = target_unit

    return df


# ---------------------------------------------------------------------------
# SPHN COLUMN MAPPING
# ---------------------------------------------------------------------------

SPHN_DEFAULT_COLUMNS = {
    "sphn_hasResult_sphn_hasDateTime": "time",
    "sphn_hasResult_sphn_hasQuantity_sphn_hasValue": "__VALUE__",
    "sphn_hasResult_sphn_hasQuantity_sphn_hasUnit_sphn_hasCode_iri": "unit",
}


def clean_sphn_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace double underscores with single underscores in column names."""
    return df.rename(columns={c: c.replace("__", "_") for c in df.columns})


# ---------------------------------------------------------------------------
# MEASUREMENTS
# ---------------------------------------------------------------------------

class AbstractMeasurement(ABC):
    """Base class for all measurement types.

    Subclasses can override:
        - _rename_columns: custom column mapping from SPHN format
        - _convert_units: unit normalization
        - _process_raw_signal: signal cleaning / resampling
        - extract: full extraction pipeline (rarely needed)
    """

    def __init__(self, config, info, conn, demog):
        self.config = config
        self.info = info
        self.demog = demog
        self._parse_args()

    def _parse_args(self):
        self.measurement_name = self.info["name_param"]
        self.impute = self.info["impute"]
        self.units = self.info["units"]
        self.range_param_min = self.info["min"]
        self.range_param_max = self.info["max"]
        self.ffill_min = self.info["ffill_min"]
        self.ffill_max = self.info["ffill_max"]
        self.ffill_limit = self.info["ffill_limit"]

    # -- Extraction pipeline ------------------------------------------------

    def extract(self, file_path, patient_id, process=True):
        if file_path is None:
            return pd.DataFrame(columns=["time", self.measurement_name])
        df = pd.read_parquet(file_path)
        df = clean_sphn_columns(df)
        df = self._rename_columns(df)
        
        if self.measurement_name not in df.columns:
            return pd.DataFrame(columns=["time", self.measurement_name])
        
        df = df[["time", self.measurement_name, "unit"]].copy()
        df = df.dropna()

        df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
        df[self.measurement_name] = pd.to_numeric(df[self.measurement_name])
        df["unit"] = df["unit"].apply(lambda x: x.split("/")[-1])

        # Unit filtering and conversion
        df = df.loc[df["unit"].isin(self.units)]
        df = self._convert_units(df)
        df = df.drop(columns=["unit"]).reset_index(drop=True)

        # Aggregate duplicate timestamps
        df["time"] = df["time"].dt.floor("min")
        df = df.groupby("time").mean().reset_index()
        df = df.sort_values("time").reset_index(drop=True)

        if process and not df.empty:
            df["patient_id"] = patient_id
            df = self._process_raw_signal(df)
            df = df.drop(columns="patient_id")

            if self.impute:
                df[self.measurement_name] = (
                    self._conditional_ffill(df[self.measurement_name])
                    .reset_index(drop=True)
                )

        return df

    # -- Column mapping (override in subclasses) ----------------------------

    def _rename_columns(self, df):
        return df.rename(columns={
            "sphn_hasResult_sphn_hasDateTime": "time",
            "sphn_hasResult_sphn_hasQuantity_sphn_hasValue": self.measurement_name,
            "sphn_hasResult_sphn_hasQuantity_sphn_hasUnit_sphn_hasCode_iri": "unit",
        })

    def _convert_units(self, df):
        return df

    # -- Signal processing --------------------------------------------------

    def _process_raw_signal(self, df):
        """Range filtering + minute-level resampling."""
        df.loc[
            ~df[self.measurement_name].between(self.range_param_min, self.range_param_max),
            self.measurement_name,
        ] = np.nan

        df = df.set_index("time")
        df = df.resample("1min").asfreq()
        df["patient_id"] = df["patient_id"].ffill()
        df = df.reset_index()

        return df

    # -- Forward fill -------------------------------------------------------

    def _conditional_ffill(self, s: pd.Series) -> pd.Series:
        """Forward-fill with range-dependent limits.

        Values in [ffill_min, ffill_max] are filled without limit.
        Outlier values are filled up to `ffill_limit` steps.
        """
        last_val = s.ffill()

        grp = s.notna().cumsum()
        dist = s.groupby(grp).cumcount()

        can_fill = s.isna() & last_val.notna()
        typical_src = last_val.between(self.ffill_min, self.ffill_max, inclusive="both")

        fill_mask = can_fill & (typical_src | (~typical_src & (dist <= self.ffill_limit)))

        out = s.copy()
        out[fill_mask] = last_val[fill_mask]
        return out


# ---------------------------------------------------------------------------
# MEASUREMENT SUBCLASSES
# ---------------------------------------------------------------------------

class RespiratoryRate(AbstractMeasurement):
    def _parse_args(self):
        super()._parse_args()
        self.priority = self.info["priority"]

    def _rename_columns(self, df):
        method_col = "sphn_hasResult_sphn_hasDataDetermination_sphn_hasMethodCode_termid"
        if method_col in df.columns:
            df = df[df[method_col].isin(self.priority)]
        return super()._rename_columns(df)


class SystolicBloodPressure(AbstractMeasurement):
    def _rename_columns(self, df):
        if "sphn_hasResult_sphn_hasDateTime" in df.columns:
            return df.rename(columns={
                "sphn_hasResult_sphn_hasDateTime": "time",
                "sphn_hasResult_sphn_hasSystolicPressure_sphn_hasValue": self.measurement_name,
                "sphn_hasResult_sphn_hasSystolicPressure_sphn_hasUnit_sphn_hasCode_iri": "unit",
            })
        else:
            return df.rename(columns={
                "sphn_hasResult_DateTime": "time",
                "sphn_hasResult_SystolicPressure_hasQuantity_hasValue": self.measurement_name,
                "sphn_hasResult_SystolicPressure_Quantity_Unit_Code_iri": "unit",
            })


class DiastolicBloodPressure(AbstractMeasurement):
    def _rename_columns(self, df):
        if "sphn_hasResult_sphn_hasDateTime" in df.columns:
            return df.rename(columns={
                "sphn_hasResult_sphn_hasDateTime": "time",
                "sphn_hasResult_sphn_hasDiastolicPressure_sphn_hasValue": self.measurement_name,
                "sphn_hasResult_sphn_hasDiastolicPressure_sphn_hasUnit_sphn_hasCode_iri": "unit",
            })
        else:
            return df.rename(columns={
                "sphn_hasResult_DateTime": "time",
                "sphn_hasResult_DiastolicPressure_hasQuantity_hasValue": self.measurement_name,
                "sphn_hasResult_DiastolicPressure_Quantity_Unit_Code_iri": "unit",
            })


class BodyTemperature(AbstractMeasurement):
    def _parse_args(self):
        super()._parse_args()
        self.perc_deviation = self.info["perc_deviation"]

    def _process_raw_signal(self, df):
        df = super()._process_raw_signal(df)

        # Patients with GA < 34 have a lower temperature threshold of 34°C
        df = df.merge(
            self.demog[["patient_id", "ga_weeks"]].drop_duplicates(),
            on="patient_id",
            how="left",
        )
        df.loc[
            (df["ga_weeks"] < 34) & (df[self.measurement_name] < 34),
            self.measurement_name,
        ] = np.nan
        df = df.drop(columns="ga_weeks")

        # Forward fill before temperature-specific cleaning
        df[self.measurement_name] = (
            self._conditional_ffill(df[self.measurement_name])
            .reset_index(drop=True)
        )

        df = self._clean_temperature(df)
        return df

    def _clean_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove sudden unrealistic temperature drops/spikes."""
        data = df[self.measurement_name].to_numpy(dtype=float)
        i = 0

        while i < len(data) - 1:
            current = data[i]
            next_val = data[i + 1]

            if np.isnan(current) or np.isnan(next_val):
                i += 1
                continue

            current_median = np.median(data[max(0, i - 30):i]) if i > 0 else current
            index_15min_before = max(0, i - 14)
            value_15min_before = data[index_15min_before]

            diff = abs(current - next_val)
            diff_15 = abs(value_15min_before - next_val)

            if diff_15 > 2:
                data[index_15min_before:i + 1] = np.nan
                base_val = current_median if value_15min_before > 38 else value_15min_before

                j, k = i + 1, 0
                while (
                    j < len(data)
                    and not np.isnan(data[j])
                    and abs(data[j] - base_val) > self.perc_deviation * diff_15
                    and k < 180
                ):
                    data[j] = np.nan
                    j += 1
                    k += 1
                i = j - 1

            elif diff > 1.25:
                base_val = next_val
                if base_val > 38.5 or base_val < 35.5:
                    base_val = current_median

                j, k = i + 1, 0
                while (
                    j < len(data)
                    and not np.isnan(data[j])
                    and abs(data[j] - base_val) > self.perc_deviation * diff
                    and k < 60
                ):
                    data[j] = np.nan
                    j += 1
                    k += 1
                i = j - 1

            i += 1

        df[self.measurement_name] = data
        return df


class FractionOfInspiredOxygen(AbstractMeasurement):
    def _parse_args(self):
        super()._parse_args()
        self.default_value = self.info["default_value"]

    def _process_raw_signal(self, df):
        df = super()._process_raw_signal(df)

        # Values between min and default are set to default (e.g., 20-21 → 21)
        df.loc[
            df[self.measurement_name].between(self.range_param_min, self.default_value),
            self.measurement_name,
        ] = self.default_value

        # Fill the first row with the default value if missing
        df.loc[0:0, self.measurement_name] = (
            df.loc[0:0, self.measurement_name].fillna(value=self.default_value)
        )

        return df


# ---------------------------------------------------------------------------
# PRELOADED MEASUREMENTS (load data once in __init__, filter per patient)
# ---------------------------------------------------------------------------

class PreloadedMeasurement(AbstractMeasurement):
    """Base for measurements that load all patient data upfront (from GraphDB or parquet)
    and then filter per patient at extraction time.

    Subclasses must implement `_load_data() -> pd.DataFrame`.
    """

    def __init__(self, config, info, conn, demog):
        super().__init__(config, info, conn, demog)
        path = os.path.join(self.path_to_checkpoint, f"{self.measurement_name}.parquet")
        self._preloaded_data = cached_extract(
            path, lambda: self._load_data(conn), force=self.extract_demographics
        )

    def _parse_args(self):
        super()._parse_args()
        self.hospitals = self.config.experiment.get("hospitals", {})
        self.extract_demographics = self.config.experiment["extract_demographics"]
        self.path_to_checkpoint = self.config.paths["checkpoint"]

    @abstractmethod
    def _load_data(self, conn) -> pd.DataFrame:
        """Load the full dataset (all patients). Called once and cached."""
        ...

    def _get_patient_data(self, patient_id: str) -> pd.DataFrame:
        """Filter the preloaded data for a single patient."""
        return self._preloaded_data[
            self._preloaded_data["patient_id"] == patient_id
        ].copy().reset_index(drop=True)

    def _process_raw_signal(self, df):
        """Simplified resampling (no range filtering) for preloaded measurements."""
        df = df.set_index("time")
        df = df.resample("1min").asfreq()
        df = df.reset_index()
        return df

    def extract(self, file_path, patient_id, process=True):
        df = self._get_patient_data(patient_id)
        df = self._prepare_patient_df(df)

        if process and not df.empty:
            df = self._process_raw_signal(df)
            if self.impute:
                df[self.measurement_name] = (
                    self._conditional_ffill(df[self.measurement_name])
                    .reset_index(drop=True)
                )

        return df

    def _prepare_patient_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and select time + measurement. Override if needed."""
        raise NotImplementedError


class BodyWeightMeasurement(PreloadedMeasurement):
    def _load_data(self, conn):
        # Turn off inference
        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "false")

        query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX sphn: <https://biomedit.ch/rdf/sphn-schema/sphn#>

            SELECT DISTINCT ?patient_id ?stay_id ?hospital ?time ?value ?unit
            WHERE {
                ?patient a sphn:SubjectPseudoIdentifier ;
                    sphn:hasIdentifier ?patient_id ;
                    sphn:hasDataProvider/sphn:hasInstitutionCode/sphn:hasName ?hospital .

                ?stay a sphn:AdministrativeCase ;
                    sphn:hasSubjectPseudoIdentifier ?patient ;
                    sphn:hasIdentifier ?stay_id .

                ?measurement a sphn:BodyWeightMeasurement ;
                    sphn:hasSubjectPseudoIdentifier ?patient ;
                    sphn:hasResult ?body_weight .

                ?body_weight a sphn:BodyWeight ;
                    sphn:hasDateTime ?time ;
                    sphn:hasQuantity/sphn:hasValue ?value ;
                    sphn:hasQuantity/sphn:hasUnit/sphn:hasCode/rdf:type/rdfs:label ?unit .
            }
        """

        df = conn["GraphDB"].query(query)

        # Turn on inference
        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "true")

        df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True).dt.floor("min")
        df["value"] = pd.to_numeric(df["value"])

        for hospital_name, hospital_info in self.hospitals.items():
            df.loc[df["hospital"] == hospital_info["original_name"], "hospital"] = hospital_name

        df = (
            df.drop_duplicates(subset=["patient_id", "stay_id", "time"])
            .sort_values(["stay_id", "time"])
            .reset_index(drop=True)
        )

        # Convert kg → g
        df.loc[df["unit"] == "kg", "value"] *= 1000
        df = df.drop(columns=["unit"])

        return df

    def _prepare_patient_df(self, df):
        df = df.rename(columns={"value": self.measurement_name})
        return (
            df[["time", self.measurement_name]]
            .copy()
            .sort_values("time")
            .drop_duplicates()
            .reset_index(drop=True)
        )


class PlateletsMeasurement(PreloadedMeasurement):
    def _load_data(self, conn):
        df = pd.read_parquet(
            os.path.join(self.path_to_checkpoint, "platelets.parquet")
        )
        df = df.loc[:, ["patient_id", "lab_val", "time"]]
        df = df.rename(columns={"lab_val": self.measurement_name})

        df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
        df[self.measurement_name] = pd.to_numeric(df[self.measurement_name])

        return df

    def _prepare_patient_df(self, df):
        df = df[["time", self.measurement_name]].copy()
        df["time"] = df["time"].dt.floor("min")
        df = df.groupby("time").mean().reset_index()
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def extract(self, file_path, patient_id, process=True):
        df = self._get_patient_data(patient_id)
        df = self._prepare_patient_df(df)

        if process and not df.empty:
            df["patient_id"] = patient_id
            # Use the parent AbstractMeasurement's range-filtering version
            df = AbstractMeasurement._process_raw_signal(self, df)
            df = df.drop(columns="patient_id")

            if self.impute:
                df[self.measurement_name] = (
                    self._conditional_ffill(df[self.measurement_name])
                    .reset_index(drop=True)
                )

        return df


# ---------------------------------------------------------------------------
# DRUG ADMINISTRATION
# ---------------------------------------------------------------------------

class DrugAdministrationMeasurement(PreloadedMeasurement):
    def _parse_args(self):
        super()._parse_args()
        self.drugs = self.info["drugs"]

    def _load_data(self, conn):
        # Turn off inference
        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "false")

        # Build the VALUES block
        code_entries = []  # (sparql_code_uri, drug_name, default_unit)
        for drug_name, drug_info in self.drugs.items():
            for terminology, drug_codes in drug_info["terminology"].items():
                for drug_code in drug_codes.keys():
                    if terminology == "snomed":
                        uri = f"snomed:{drug_code}"
                    else:
                        uri = drug_code
                    code_entries.append((uri, drug_name, drug_info["default_unit"]))

        # Each entry becomes a pair (?drug_code_type ?drug_label)
        values_rows = "\n".join(
            f'        ( {uri} "{drug_name}" )'
            for uri, drug_name, _ in code_entries
        )

        # Build a lookup for post-processing
        drug_unit_map = {
            drug_name: drug_info["default_unit"]
            for drug_name, drug_info in self.drugs.items()
        }

        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX sphn: <https://biomedit.ch/rdf/sphn-schema/sphn#>
            PREFIX snomed: <http://snomed.info/id/>

            SELECT DISTINCT ?adm ?patient_id ?stay_id ?hospital ?start_date ?end_date
                            ?drug_dose ?drug_unit ?substance_dose ?substance_unit
                            ?drug_label
            WHERE {{
                VALUES (?drug_code_type ?drug_label) {{
                    {values_rows}
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
                    sphn:hasDrug ?drug .

                ?drug sphn:hasActiveIngredient ?substance ;
                    sphn:hasQuantity ?drug_quantity .

                ?substance a sphn:Substance ;
                    sphn:hasCode ?substance_code .

                ?substance_code rdf:type ?drug_code_type .

                OPTIONAL {{
                    ?drug_quantity sphn:hasValue ?drug_dose ;
                        sphn:hasUnit/sphn:hasCode/rdf:type/rdfs:label ?drug_unit .
                }}

                OPTIONAL {{
                    ?substance sphn:hasQuantity ?substance_quantity .
                    ?substance_quantity sphn:hasValue ?substance_dose ;
                        sphn:hasUnit/sphn:hasCode/rdf:type/rdfs:label ?substance_unit .
                }}
            }}
        """

        df = conn["GraphDB"].query(query)

        # Turn on inference
        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "true")

        if df.empty:
            return pd.DataFrame()

        # --- Post-process (same logic as before, but vectorized over all drugs) ---
        df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", utc=True).dt.floor("min")
        df["end_date"] = pd.to_datetime(df["end_date"], format="mixed", utc=True).dt.floor("min")
        df = df[df["start_date"] < df["end_date"]].reset_index(drop=True)

        hospital_map = {
            info["original_name"]: name
            for name, info in self.hospitals.items()
        }
        df["hospital"] = df["hospital"].replace(hospital_map)

        df["drug_dose"] = pd.to_numeric(df["drug_dose"])
        df["substance_dose"] = pd.to_numeric(df["substance_dose"])

        # Merge dose/unit depending on hospital
        df["dose"] = df["substance_dose"]
        df["unit"] = df["substance_unit"]
        is_drug_hospital = df["hospital"].isin(["CHUV"])
        df.loc[is_drug_hospital, "dose"] = df.loc[is_drug_hospital, "drug_dose"]
        df.loc[is_drug_hospital, "unit"] = df.loc[is_drug_hospital, "drug_unit"]

        # Fix KiSpi's mapping issue
        df["adm_id"] = df["adm"].apply(lambda x: x.split("-")[-1])
        df_missing = pd.read_csv("/data/graphdb-import/adonis_full_dataset/sphn_dem_adonis_20260417T090026/content/adonis_quantities_drugAdministration_subset.csv", sep=";")
        df_missing = df_missing[["id", "AmountGiven", "ucum_sphn"]].rename(columns={
            "id": "adm_id", 
            "AmountGiven": "drug_dose_extra", 
            "ucum_sphn": "drug_unit_extra"
        })
        df_missing.loc[df_missing["drug_unit_extra"]=="sblIUsbr", "drug_unit_extra"] = "[IU]" 
        df_missing = df_missing[df_missing["drug_unit_extra"].isin(["mg", "ug", "[IU]"])]

        df = df.merge(df_missing, on="adm_id", how="left")
        is_missing =~ df["drug_unit_extra"].isna()
        df.loc[is_missing, "dose"] = df.loc[is_missing, "drug_dose_extra"]
        df.loc[is_missing, "unit"] = df.loc[is_missing, "drug_unit_extra"]

        df = df.dropna(subset=["dose", "unit"])
        df = df.drop(columns=["drug_dose", "drug_unit", "substance_dose", "substance_unit", "drug_dose_extra", "drug_unit_extra"])
        df["dose"] = pd.to_numeric(df["dose"])

        df["duration_time"] = (df["end_date"] - df["start_date"]).dt.total_seconds() // 60
        df["duration_unit"] = "min"
        df = df.rename(columns={"drug_label": "drug_name"})

        # Unit conversion per drug (each drug may have a different target unit)
        converted_parts = []
        for drug_name, group in df.groupby("drug_name"):
            target_unit = drug_unit_map[drug_name]
            converted = convert_units_vectorized(
                group, target_unit=target_unit, value_col="dose", unit_col="unit"
            )
            converted_parts.append(converted)

        df_drug = pd.concat(converted_parts, ignore_index=True)

        df_drug = (
            df_drug
            .drop_duplicates(subset=["patient_id", "stay_id", "start_date", "end_date", "drug_name"])
            .sort_values(["stay_id", "start_date"])
            .reset_index(drop=True)
        )

        return df_drug

    def _prepare_patient_df(self, df):
        # Default: not used directly, subclasses override extract()
        raise NotImplementedError


class SteroidsMeasurement(DrugAdministrationMeasurement):
    def extract(self, file_path, patient_id, process=True):
        df = self._get_patient_data(patient_id)
        df = df.rename(columns={"start_date": "time"})

        df[self.measurement_name] = 1
        df = (
            df[["time", self.measurement_name]]
            .copy()
            .sort_values("time")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if process and not df.empty:
            df = self._process_raw_signal(df)

        return df


class InotropesMeasurement(DrugAdministrationMeasurement):
    def _load_data(self, conn):
        df_drug = super()._load_data(conn)

        df_drug = df_drug[
            ["patient_id", "start_date", "end_date", "drug_name", "dose", "unit", "duration_time", "duration_unit"]
        ]
        df_drug = df_drug.sort_values(["patient_id", "start_date"])

        # Expand to minute-level ranges
        df_drug = (
            df_drug
            .assign(
                time=lambda x: x.apply(
                    lambda r: pd.date_range(
                        start=r["start_date"],
                        end=r["end_date"],
                        freq="1min",
                        inclusive="left",
                    ),
                    axis=1,
                )
            )
            .explode("time")
            .loc[:, ["patient_id", "time", "drug_name", "dose", "unit", "duration_time", "duration_unit"]]
            .reset_index(drop=True)
        )
        return df_drug

    def extract(self, file_path, patient_id, process=True):
        df = self._get_patient_data(patient_id)

        drug_cols = ["time", "drug_name", "duration_time", "duration_unit"]
        agg_dict = {
            "inotropes_count": ("drug_name", "nunique"),
            "duration_time": ("duration_time", "first"),
            "duration_unit": ("duration_unit", "first"),
        }

        for drug_name, drug_info in self.drugs.items():
            if drug_info["extract"]:
                mask = df["drug_name"] == drug_name
                df[f"{drug_name}_dose"] = df["dose"].where(mask)
                df[f"{drug_name}_unit"] = df["unit"].where(mask)

                drug_cols.extend([f"{drug_name}_dose", f"{drug_name}_unit"])
                agg_dict[f"{drug_name}_dose"] = (f"{drug_name}_dose", "first")
                agg_dict[f"{drug_name}_unit"] = (f"{drug_name}_unit", "first")

        df = (
            df[drug_cols].copy()
            .sort_values("time")
            .groupby("time")
            .agg(**agg_dict)
            .reset_index()
        )

        df["inotropes"] = (df["inotropes_count"] > 0).astype(int)

        if process and not df.empty:
            df = self._process_raw_signal(df)

        return df


# ---------------------------------------------------------------------------
# VENTILATION PROCEDURE
# ---------------------------------------------------------------------------

class VentilationProcedureMeasurement(AbstractMeasurement):
    def __init__(self, config, info, conn, demog):
        super().__init__(config, info, conn, demog)

        path = os.path.join(self.path_to_checkpoint, f"{self.measurement_name}.parquet")
        self.df_measurement = cached_extract(
            path, lambda: self._load_measurement_type(conn), force=self.extract_demographics
        )

    def _parse_args(self):
        super()._parse_args()
        self.extract_demographics = self.config.experiment["extract_demographics"]
        self.path_to_checkpoint = self.config.paths["checkpoint"]
        self.concept = self.info["concept"]
        self.priority = self.info["priority"]

    def _load_measurement_type(self, conn):
        query = f"""
            PREFIX sphn: <https://biomedit.ch/rdf/sphn-schema/sphn#>
            PREFIX adonis: <https://biomedit.ch/rdf/sphn-schema/adonis#>

            SELECT DISTINCT ?hospital ?measurement ?code
            WHERE {{
                ?patient a sphn:SubjectPseudoIdentifier ;
                    sphn:hasIdentifier ?patient_id ;
                    sphn:hasDataProvider/sphn:hasInstitutionCode/sphn:hasName ?hospital .

                ?measurement a adonis:{self.concept}Measurement ;
                    sphn:hasDataDetermination/sphn:hasMethodCode/a ?code ;
                    sphn:hasSubjectPseudoIdentifier ?patient .
            }}
        """
        # Turn off inference
        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "false")

        df = conn["GraphDB"].query(query)

        # Turn on inference
        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "true")

        df = rename_hospitals(df, self.config.experiment["hospitals"])

        df["code"] = df["code"].apply(lambda x: x.replace("http://snomed.info/id/", "SNOMED-CT-"))

        # Add measurement id
        hospitals = self.config.experiment["hospitals"]
        df["id"] = ""
        df["id"] = df.apply(
            lambda x: "-".join(x["measurement"].split("-")[-5:])
            if x["hospital"] == "USZ"
            else x["id"],
            axis=1,
        )
        df["id"] = df.apply(
            lambda x: x["measurement"].split("-")[-1]
            if x["hospital"] in ["CHUV", "KiSpi"]
            else x["id"],
            axis=1,
        )

        return df

    def _rename_columns(self, df):
        df = super()._rename_columns(df)
        df = df.merge(self.df_measurement[["id", "code"]], on="id", how="inner")
        df = df[df["code"].isin(self.priority)]
        return df

    def _convert_units(self, df):
        # mbar to cmH2O
        df.loc[df["unit"] == "mbar", self.measurement_name] *= 1.0197
        df.loc[df["unit"] == "mbar", "unit"] = "cmsblH2Osbr"
        return df


feature_dict = {
    "HR": AbstractMeasurement,
    "SpO2": AbstractMeasurement,
    "RR": RespiratoryRate,
    "SBP": SystolicBloodPressure,
    "DBP": DiastolicBloodPressure,
    "FiO2": FractionOfInspiredOxygen,
    "bodyT": BodyTemperature,
    "weight": BodyWeightMeasurement,
    "PEEP": VentilationProcedureMeasurement,
    "VE": AbstractMeasurement,
    "PIP": AbstractMeasurement,
    "MAP": VentilationProcedureMeasurement,
    "CET": AbstractMeasurement,
    "CEH": AbstractMeasurement,
    "PLT": PlateletsMeasurement,
    "steroids": SteroidsMeasurement,
    "inotropes": InotropesMeasurement,
}


# ---------------------------------------------------------------------------
# EXTRACTOR
# ---------------------------------------------------------------------------

class Extractor(ABC):
    """
    Orchestrates extraction, merging, and post-processing of patient time series.
    """

    def __init__(self, config, conn, demog, df_file_names):
        self.config = config
        self.demog = demog
        self.df_file_names = df_file_names
        self._parse_args()

        # Create cache dir
        self.path_to_cache = os.path.join(self.path_to_checkpoint, "cache")
        os.makedirs(self.path_to_cache, exist_ok=True)

        # Load ventilation — conn used locally, not stored
        ventilation_path = os.path.join(self.path_to_checkpoint, "ventilation.parquet")
        self.df_ventilation = cached_extract(
            ventilation_path, lambda: self._load_ventilation(conn), force=self.extract_demographics
        )

        # Build ALL measurement handlers (they are lightweight except for preloaded ones)
        self._all_measurements = {}
        for _, variables in self.variables_to_extract.items():
            for variable_name, variable_info in variables.items():
                if variable_info["extract"]:
                    self._all_measurements[variable_name] = feature_dict[variable_name](
                        config, variable_info, conn, self.demog
                    )

        logger.info(
            "ExtractEngine initialized with: %s",
            [(k, type(v).__name__) for k, v in self._all_measurements.items()],
        )

        # Initialize score engine
        self.scorer = ScoreEngine(self.config)                    

    def _parse_args(self):
        self.extract_demographics = self.config.experiment["extract_demographics"]
        self.hospitals = self.config.experiment["hospitals"]
        self.variables_to_extract = self.config.experiment["variables"]
        self.path_to_checkpoint = self.config.paths["checkpoint"]
        self.path_to_time_series = self.config.paths["time_series"]
        self.path_to_queries = self.config.paths["queries"]

    def get_available_variables(self) -> list[str]:
        """Return the list of all extractable variable names."""
        return list(self._all_measurements.keys())

    def get_measurements(self, variables: list[str] | None = None) -> dict:
        """Return measurement handlers, optionally filtered to a subset.

        Args:
            variables: List of variable names (e.g., ["HR", "SpO2"]).
                       If None, returns all configured measurements.

        Returns:
            Dict mapping variable names to measurement handler instances.

        Raises:
            ValueError: If any requested variable is not available.
        """
        if variables is None:
            return dict(self._all_measurements)

        unknown = set(variables) - set(self._all_measurements)
        if unknown:
            available = ", ".join(sorted(self._all_measurements.keys()))
            raise ValueError(
                f"Unknown variables: {sorted(unknown)}. Available: {available}"
            )

        return {k: self._all_measurements[k] for k in variables}

    # -- Demographics -------------------------------------------------------

    def get_demog(self):
        return self.demog.copy()

    def _add_demographics(self, df):
        demog = self.get_demog()
        demog = demog[["patient_id", "stay_id", "ga_weeks", "sex", "adm_date", "disc_date"]].copy()

        demog["adm_date"] = pd.to_datetime(demog["adm_date"], format="mixed", utc=True)
        demog["disc_date"] = pd.to_datetime(demog["disc_date"], format="mixed", utc=True)
        demog = demog.drop_duplicates()

        df = df.merge(demog, on="patient_id", how="left")
        df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
        df = df[(df["time"] >= df["adm_date"]) & (df["time"] <= df["disc_date"])]
        df = df.reset_index(drop=True)
        df["ga_weeks"] = np.floor(df["ga_weeks"])

        return df
    
    # -- Ventilation --------------------------------------------------------

    def _add_ventilation(self, df):
        df["ventilation"] = 0
        df["invasive"] = 0

        # Index df by stay_id once for fast lookup
        stay_groups = df.groupby("stay_id").indices

        for _, vent in self.df_ventilation.iterrows():
            stay_id = vent["stay_id"]
            if stay_id not in stay_groups:
                continue

            idx = stay_groups[stay_id]
            times = df["time"].iloc[idx]

            mask = (times >= vent["start_ventilation"]) & (times <= vent["end_ventilation"])
            matched = idx[mask.values]

            df.loc[matched, "ventilation"] = 1
            if vent["invasive"]:
                df.loc[matched, "invasive"] = 1

        return df
    
    # -- Phase-1 cache helpers -------------------------------------------

    def _processed_cache_path(self, patient_id: str) -> str:
        """Path for the fully-processed per-patient cache file (Phase 1 output)."""
        return os.path.join(self.path_to_cache, f"{patient_id}_processed.parquet")

    def extract_single_to_cache(self, patient_id: str) -> bool:
        """Phase 1 worker entry point.

        Reads all variable parquets concurrently (ThreadPoolExecutor), merges,
        applies full _process_patient_single pipeline, writes to cache.
        Returns True on success, False if the patient should be skipped.
        """
        import time as _time
        t0 = _time.perf_counter()

        cache_path = self._processed_cache_path(patient_id)
        if os.path.exists(cache_path):
            logger.info("Patient %s already cached — skipping", patient_id)
            return True

        patient_file_names = self.df_file_names.loc[
            self.df_file_names["patient_id"] == patient_id
        ]
        if patient_file_names.empty:
            logger.warning("No file names found for patient %s — skipping", patient_id)
            return False

        hospital = patient_file_names["hospital"].values[0]
        measurements = self.get_measurements()

        # Resolve file paths up front (sequential — fast, no I/O)
        resolved = []
        for variable_name, variable in measurements.items():
            try:
                file_name = patient_file_names[variable_name].values[0]
            except (KeyError, IndexError):
                file_name = None
            file_path = resolve_parquet_path(
                hospital=hospital,
                file_name=file_name,
                variable_name=variable_name,
                path_to_time_series=self.path_to_time_series,
                variables_to_extract=self.variables_to_extract,
            )
            resolved.append((variable_name, variable, file_path))

        # Concurrent parquet reads (cap at 8 threads to avoid memory pressure when many loky workers run in parallel on the same node)
        raw_results: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(resolved))) as pool:
            future_to_var = {
                pool.submit(variable.extract, file_path, patient_id, True): variable_name
                for variable_name, variable, file_path in resolved
            }
            for future in as_completed(future_to_var):
                variable_name = future_to_var[future]
                try:
                    raw_results[variable_name] = future.result()
                    logger.debug(
                        "Patient %s: read variable %s (%d rows)",
                        patient_id, variable_name,
                        len(raw_results[variable_name]),
                    )
                except Exception as exc:
                    logger.error(
                        "Patient %s: error reading variable %s: %s",
                        patient_id, variable_name, exc, exc_info=True,
                    )
                    raw_results[variable_name] = None

        # Merge in deterministic order
        patient_data = None
        missing_variables = []
        for variable_name, variable, _ in resolved:
            df = raw_results.get(variable_name)
            if df is None or df.empty:
                missing_variables.append(variable_name)
                continue
            df["time"] = df["time"].dt.floor("us").astype("datetime64[us, UTC]")
            if patient_data is None:
                patient_data = df
            else:
                merge_cols = ["time"]
                if variable_name in patient_data.columns and variable_name in df.columns:
                    merge_cols.append(variable_name)
                patient_data = pd.merge(patient_data, df, on=merge_cols, how="outer")

        if patient_data is None:
            logger.warning("No data files found for patient %s — skipping", patient_id)
            return False

        for variable_name in missing_variables:
            if variable_name not in patient_data.columns:
                patient_data[variable_name] = (
                    DEFAULT_FIO2 if variable_name == "FiO2" else np.nan
                )
            logger.warning(
                "Patient %s: variable %s missing — filling with %s",
                patient_id, variable_name,
                DEFAULT_FIO2 if variable_name == "FiO2" else "NaN",
            )

        patient_data["patient_id"] = patient_id
        patient_data["hospital"] = hospital
        patient_data = patient_data.sort_values("time").reset_index(drop=True)

        logger.debug(
            "Patient %s: merged %d variables → %d minute rows",
            patient_id, len(measurements) - len(missing_variables), len(patient_data),
        )

        # Full per-patient processing pipeline
        patient_data = self._process_patient_single(patient_data)

        patient_data.to_parquet(cache_path, index=False)
        elapsed = _time.perf_counter() - t0
        logger.info(
            "Cached patient %s — %d rows, %.2fs",
            patient_id, len(patient_data), elapsed,
        )
        return True

    def _process_patient_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full processing pipeline to a single patient's minute-level data."""
        df = self._add_demographics(df)
        df = self._add_ventilation(df)
        df = self.scorer.compute(df)

        # Delta signals — direct shift (no groupby, single patient)
        delta_exclude = {"inotropes", "steroids", "PLT"}
        for signal in self._all_measurements.keys():
            if signal in df.columns:
                df[f"{signal}_last"] = df[signal].shift(1)
                df[f"{signal}_delta"] = (
                    (df[signal] - df[f"{signal}_last"]) / df[signal] * 100
                )

        last_cols = [f"{s}_last" for s in self._all_measurements if f"{s}_last" in df.columns]
        delta_drop = [
            f"{s}_delta" for s in delta_exclude if f"{s}_delta" in df.columns
        ]
        drop_cols = last_cols + delta_drop
        drop_cols += [
            c for c in ["inotropes", "inotropes_count", "steroids", "PLT"]
            if c in df.columns
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Time differences
        df["time_difference_minutes"] = np.floor(
            (df["time"] - df["adm_date"]).dt.total_seconds() / 60
        ).astype(int)
        df["time_difference_hours"] = np.floor(
            (df["time"] - df["adm_date"]).dt.total_seconds() / 3600
        ).astype(int)
        df["time_difference_days"] = np.floor(
            (df["time"] - df["adm_date"]).dt.total_seconds() / 86400
        ).astype(int)

        df = df.drop(
            columns=[c for c in ["adm_date", "disc_date", "time"] if c in df.columns]
        )
        return df

    # -- File loaders ----------------------------------------------------

    def _load_ventilation(self, conn):
        file_path = os.path.join(self.path_to_queries, "ventilation.txt")
        with open(file_path, "r", encoding="utf-8") as file:
            query = file.read()

        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "false")

        df_ventilation = conn["GraphDB"].query(query)

        conn["GraphDB"].conn.clearParameter("infer")
        conn["GraphDB"].conn.addParameter("infer", "true")

        df_ventilation = rename_hospitals(df_ventilation, self.hospitals)

        df_ventilation["start_date"] = pd.to_datetime(df_ventilation["start_date"], format="mixed", utc=True)
        df_ventilation["end_date"] = pd.to_datetime(df_ventilation["end_date"], format="mixed", utc=True)

        df_ventilation = df_ventilation.drop_duplicates(subset=["stay_id", "start_date", "end_date"])
        df_ventilation = df_ventilation.sort_values(
            ["hospital", "patient_id", "stay_id", "start_date"]
        ).reset_index(drop=True)

        df_ventilation["invasive"] = df_ventilation["mode"].isin(INVASIVE_VENTILATION_MODES)

        # Merge continuous ventilation blocks
        df_ventilation["prev_end"] = df_ventilation.groupby(["stay_id", "invasive"])["end_date"].shift()
        df_ventilation["is_continuous"] = (
            (df_ventilation["start_date"] - df_ventilation["prev_end"]) <= pd.to_timedelta(1, unit="h")
        )
        df_ventilation["block"] = (
            (~df_ventilation["is_continuous"].fillna(False))
            .groupby(df_ventilation["stay_id"])
            .cumsum()
        )
        df_ventilation = (
            df_ventilation.groupby(["hospital", "stay_id", "patient_id", "invasive", "block"])
            .agg(
                start_ventilation=("start_date", "min"),
                end_ventilation=("end_date", "max"),
            )
            .reset_index()
        )

        return df_ventilation