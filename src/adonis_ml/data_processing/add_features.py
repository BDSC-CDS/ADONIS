from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import warnings

import logging

import numpy as np
import pandas as pd
import scipy
from EntropyHub import ApEn, FuzzEn, SampEn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AbstractFeature(ABC):
    def __init__(self, config: dict, demog: pd.DataFrame, feature_info: dict):
        self.config = config
        self.demog = demog
        self.feature_info = feature_info

        self._parse_args()

        # All vitals defined at the experiment level (used as fallback)
        all_vitals = list(config.experiment["variables"]["vitals"].keys())

        # Per-feature variable list: explicit from config, or fall back to all vitals
        if "variables" in self.feature_info:
            self.variables = self.feature_info["variables"]
        else:
            self.variables = all_vitals

    def _parse_args(self):
        self.variables_to_extract = self.config.experiment["variables"]

    def validate_variables(self, df: pd.DataFrame, suffix: str = "") -> List[str]:
        """Return the subset of self.variables present in df.
        
        Args:
            df: The dataframe to check against.
            suffix: Optional suffix appended to variable names when checking
                    (e.g. '_median' for TrendsFeature).
        """
        valid = []
        for var in self.variables:
            col = f"{var}{suffix}" if suffix else var
            if col not in df.columns:
                warnings.warn(
                    f"{self.__class__.__name__}: variable '{col}' not found in "
                    f"dataframe (columns: {df.columns.tolist()[:10]}...). Skipping."
                )
            else:
                valid.append(var)

        if not valid:
            raise ValueError(
                f"{self.__class__.__name__}: none of the configured variables "
                f"{self.variables} found in dataframe."
            )
        return valid

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        pass


class SpikesDropsFeature(AbstractFeature):
    def extract(
        self, df: pd.DataFrame, extreme: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Single-patient variant — no groupby on stay_id."""
        variables = self.validate_variables(df)
        k = 3 if extreme else 1.5
        var_time = "hours"

        for var in variables:
            new_cols = {}

            # Direct rolling — no groupby (single patient)
            moving_median = df[var].rolling(60, min_periods=10).median().shift(-30)
            new_cols[f"{var}_moving_median"] = moving_median

            df["MAD"] = abs(df[var] - moving_median)
            mad = df["MAD"].rolling(60, min_periods=10).median().shift(-30)
            new_cols[f"{var}_MAD"] = mad

            lower_bound = moving_median - k * mad
            upper_bound = moving_median + k * mad
            new_cols[f"{var}_lower_bound"] = lower_bound
            new_cols[f"{var}_upper_bound"] = upper_bound

            drops = (df[var] < lower_bound).astype(int).fillna(0)
            spikes = (df[var] > upper_bound).astype(int).fillna(0)
            outliers = drops + spikes
            new_cols[f"{var}_drops"] = drops
            new_cols[f"{var}_spikes"] = spikes
            new_cols[f"{var}_outliers"] = outliers

            lower_diff = abs(df[var] - lower_bound) / abs(lower_bound)
            upper_diff = abs(df[var] - upper_bound) / abs(upper_bound)
            diff_bounds = pd.concat([lower_diff, upper_diff], axis=1).min(axis=1)
            new_cols[f"diff_bounds_and_{var}"] = diff_bounds

            drops_corr = drops * diff_bounds
            spikes_corr = spikes * diff_bounds
            outliers_corr = outliers * diff_bounds
            new_cols[f"{var}_drops_corrected"] = drops_corr
            new_cols[f"{var}_spikes_corrected"] = spikes_corr
            new_cols[f"{var}_outliers_corrected"] = outliers_corr

            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

            # Hourly counts — groupby on time_difference_hours only
            df[
                [
                    f"{var}_nb_drops_{var_time}",
                    f"{var}_nb_spikes_{var_time}",
                    f"{var}_nb_outliers_{var_time}",
                ]
            ] = df.groupby("time_difference_hours")[
                [f"{var}_drops", f"{var}_spikes", f"{var}_outliers"]
            ].transform("sum")
            df[
                [
                    f"{var}_nb_drops_corrected",
                    f"{var}_nb_spikes_corrected",
                    f"{var}_nb_outliers_corrected",
                ]
            ] = df.groupby("time_difference_hours")[
                [
                    f"{var}_drops_corrected",
                    f"{var}_spikes_corrected",
                    f"{var}_outliers_corrected",
                ]
            ].transform("sum")

        df.drop_duplicates(subset=["stay_id", "time_difference_hours"], inplace=True)

        feats_to_keep = (
            [var + "_nb_drops_corrected" for var in variables]
            + [var + "_nb_spikes_corrected" for var in variables]
            + [var + "_nb_outliers_corrected" for var in variables]
        )
        key = ["stay_id", "time_difference_hours"]
        df = df[key + feats_to_keep]
        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        if extreme:
            rename_dict = {col: col + "_ext" for col in feats_to_keep}
            df = df.rename(columns=rename_dict)
            feats_to_keep = [col + "_ext" for col in feats_to_keep]
        return df, feats_to_keep


class TrendsFeature(AbstractFeature):
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Single-patient variant — reindex instead of cartesian product, direct shift."""
        variables = self.validate_variables(df, suffix="_median")

        # Reindex to a full contiguous hour range (no cross-join)
        hour_range = range(
            int(df["time_difference_hours"].min()),
            int(df["time_difference_hours"].max()) + 1,
        )
        df = (
            df.set_index("time_difference_hours")
            .reindex(hour_range)
            .rename_axis("time_difference_hours")
            .reset_index()
        )
        df.sort_values("time_difference_hours", inplace=True)

        slopes = {}
        feats = []
        for var in variables:
            for nb_hours in [4, 6, 12, 24, 48]:
                try:
                    # Direct shift — no groupby (single patient)
                    value_t = df[f"{var}_median"].shift(periods=nb_hours)
                    slope = (df[f"{var}_median"] - value_t) / nb_hours
                    slopes[f"{var}_slope_{nb_hours}"] = slope
                    feats.append(f"{var}_slope_{nb_hours}")
                except Exception as exc:
                    logger.warning("Trends: error for %s at %dh: %s", var, nb_hours, exc)

        slope_df = pd.DataFrame(slopes)
        df = pd.concat([df, slope_df], axis=1)

        results_df = df[["stay_id", "time_difference_hours"] + feats].copy()
        return results_df, feats


class TimeDifferenceSepsisFeature(AbstractFeature):
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        This functions adds time_hours_sepsis and time_difference_hours_sepsis thanks to the blood culture datetime.

        Sometimes there is a bloodculture datetime whereas the patient is NON septic.
        Nonetheless, in this function, we only add the features for SEPTIC patients (who always have bloodculture datetime).


        Returns:
            results_df (DataFrame): dataframe with columns stay_id, time_difference_hours, time_hours_sepsis, time_difference_hours_sepsis

        """
        # Process to align with sepsis_time == hemoculture
        # demog['datetime_hemoculture'] = pd.to_datetime(demog['datetime_hemoculture'])
        # demog['date_entree_sejour'] = pd.to_datetime(demog['date_entree_sejour'])

        # Compute the difference for each time_difference_hours
        df = df.merge(
            self.demog[
                [
                    "stay_id",
                    "time_hours_sepsis",
                    "adm_date",
                    "datetime_hemoculture",
                    "sepsis",
                ]
            ],
            on="stay_id",
            how="left",
        )
        df["time_hours_sepsis_admission"] = np.where(
            (~df["datetime_hemoculture"].isnull()) & (df["sepsis"] == 1),
            (df["datetime_hemoculture"] - df["adm_date"]).dt.total_seconds() // 3600,
            np.nan,
        )
        df["time_difference_hours_sepsis"] = df["time_difference_hours"].sub(
            df["time_hours_sepsis_admission"], fill_value=np.nan
        )

        # Select only useful features
        results_df = df[
            [
                "stay_id",
                "time_difference_hours",
                "time_hours_sepsis",
                "time_difference_hours_sepsis",
            ]
        ].copy()

        return results_df, ["time_hours_sepsis", "time_difference_hours_sepsis"]


class StdFeature(AbstractFeature):
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Single-patient variant — groupby on time_difference_hours only."""
        variables = self.validate_variables(df)
        results_df = (
            df.groupby("time_difference_hours")[variables]
            .agg("std")
            .reset_index()
        )
        results_df.rename(columns={var: var + "_std" for var in variables}, inplace=True)
        list_feats = results_df.columns[1:]
        return results_df, list(list_feats.values)


class KurtosisSkewnessFeature(AbstractFeature):
    def _parse_args(self):
        super()._parse_args()
        self.kurtosis = self.config.experiment["features"]["kurt_skew"]["kurtosis"]
        self.skewness = self.config.experiment["features"]["kurt_skew"]["skewness"]

    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Single-patient variant — one agg over all variables, no per-variable loop."""
        variables = self.validate_variables(df)
        agg_funcs = []
        if self.skewness:
            agg_funcs.append("skew")
        if self.kurtosis:
            agg_funcs.append(scipy.stats.kurtosis)

        results_df = (
            df.groupby("time_difference_hours")[variables]
            .agg(agg_funcs)
            .reset_index()
        )
        results_df.columns = (
            ["time_difference_hours"]
            + [
                f"{var}_{'skew' if fn == 'skew' else 'kurt'}"
                for var in variables
                for fn in agg_funcs
            ]
        )
        feats_to_keep = results_df.columns[1:]
        return results_df, list(feats_to_keep.values)


class EntropyFeature(AbstractFeature):
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Single-patient variant — groupby on time_difference_hours only, plain loop."""
        variables = self.validate_variables(df)
        grouped_df = df.groupby("time_difference_hours")

        results = [
            process_group(name, group, variables)
            for name, group in grouped_df
        ]

        records = []
        for name, res in results:
            record = [name]
            for var in variables:
                record.extend(res[var])
            records.append(record)

        columns = ["time_difference_hours"]
        for var in variables:
            columns.extend([f"{var}_ApEn", f"{var}_SampEn", f"{var}_FuzzEn"])

        results_df = pd.DataFrame(records, columns=columns)
        list_feats = results_df.columns[1:]

        filtered_df = results_df[list_feats].replace(np.inf, np.nan)
        max_plus_std = filtered_df.max() + filtered_df.std()
        for col in list_feats:
            results_df[col] = results_df[col].replace(np.inf, max_plus_std[col])

        return results_df, list(list_feats.values)


def apply_entropy_funcs(group, variables):
    results = {}
    for var in variables:
        try:
            ApEn_result = ApEn(group[var].values)[0][-1]
        except Exception as e:
            ApEn_result = np.nan

        try:
            SampEn_result = SampEn(group[var].values)[0][-1]
        except Exception as e:
            SampEn_result = np.nan

        try:
            FuzzEn_result = FuzzEn(group[var].values)[0][-1]
        except Exception as e:
            FuzzEn_result = np.nan

        results[var] = (ApEn_result, SampEn_result, FuzzEn_result)
    return results


def process_group(name, group, variables):
    return name, apply_entropy_funcs(group, variables)


feature_dict = {
    "spikes_drops": SpikesDropsFeature,
    "trends": TrendsFeature,
    "time_sepsis": TimeDifferenceSepsisFeature,
    "std_feat": StdFeature,
    "kurt_skew": KurtosisSkewnessFeature,
    "entropy": EntropyFeature,
}


class FeatureExtractor:
    def __init__(self, config: dict, demog: pd.DataFrame, list_of_variables: List[str]):
        self.config = config
        self.list_of_variables = list_of_variables
        self.demog = demog

        self.features_to_extract = self.config.experiment["features"]
        self.features = {}
        for feature_name, feature_info in self.features_to_extract.items():
            self.features[feature_name] = feature_dict[feature_name](
                self.config, self.demog, feature_info
            )

    def extract(
        self,
        df_min: pd.DataFrame,
        df_hour: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        df = df_hour.copy()
        list_of_variables = {
            group: variables.copy()
            for group, variables in self.list_of_variables.items()
        }

        for feature_name, feature in self.features.items():
            on_basis = self.features_to_extract[feature_name]["on_basis"]
            src = df_min if on_basis == "min" else df_hour

            if feature_name == "spikes_drops":
                results_df, list_of_variables_feature = feature.extract(src, extreme=True)
                df = self._merge_features(df, results_df)
                list_of_variables[f"{feature_name}_extreme"] = list_of_variables_feature

            results_df, list_of_variables_feature = feature.extract(src)
            df = self._merge_features(df, results_df)
            list_of_variables[feature_name] = list_of_variables_feature

        return df, list_of_variables

    def _merge_features(self, df_original: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        merge_on = ["stay_id", "time_difference_hours"] if "stay_id" in results_df.columns else ["time_difference_hours"]
        return df_original.merge(results_df, how="left", on=merge_on)
