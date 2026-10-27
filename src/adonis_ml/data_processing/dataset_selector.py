import logging
import os
from abc import ABC, abstractmethod
import pickle

import pandas as pd
from pandas.util import hash_pandas_object
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm
from adonis_ml.data_processing.select_patients import patient_selector_dict

logger = logging.getLogger(__name__)

class DatasetSelector(ABC):
    def __init__(self, config, demog):
        self.config = config
        self.demog = demog

        self.path_to_checkpoint = self.config.paths["checkpoint"]

        self.forewarning_time = self.config.experiment["params"]["forewarning_time"]
        self.nb_time_windows_sepsis = self.config.experiment["params"]["nb_time_windows_sepsis"]
        self.shift_time_windows_sepsis = self.config.experiment["params"]["shift_time_windows_sepsis"]
        self.include_all_sepsis = self.config.experiment["params"]["include_all_sepsis"]
        self.prediction_time = self.config.experiment["params"]["prediction_time"]
        self.days_to_exclude = self.config.experiment["params"]["days_to_exclude"]
        self.self_controls = self.config.experiment["params"]["self_controls"]
        self.self_controls_time = self.config.experiment["params"]["self_controls_time"]
        self.hours_to_exclude_abx = self.config.experiment["params"]["hours_to_exclude_abx"]
        self.sepsis_rate = self.config.experiment["params"]["sepsis_rate"]

        self.hospitals = self.config.experiment["hospitals"]
        self.patient_selection = self.config.experiment["patient_selection"]

        # Patient selector
        self.patient_selectors = []
        for matching_strategy, matching_info in self.patient_selection.items():
            if matching_info["extract"]:
                self.patient_selectors.append(patient_selector_dict[matching_strategy](self.config, self.demog))
        
    def _add_labels(self, df, hospital):
        df["sepsis_time"] = 0
        df["control_type"] = "control"

        # LOS patients
        sepsis_ids = self.patient_selectors[0].patients_ids[hospital]["sepsis"]
        sepsis_idx = (
            (df["stay_id"].isin(sepsis_ids)) &
            (df["time_difference_hours_sepsis"] >= -(self.forewarning_time)) &
            (df["time_difference_hours_sepsis"] <= -1)
        )
        df.loc[sepsis_idx, "sepsis_time"] = 1
        df.loc[sepsis_idx, "control_type"] = "sepsis"

        # Self controls
        self_control_idx = (
            (df["stay_id"].isin(sepsis_ids)) & 
            (df["time_difference_hours_sepsis"] < -(self.forewarning_time))
        )
        df.loc[self_control_idx, "control_type"] = "self_control"

        return df

    @abstractmethod
    def _select_time_windows(self, df, hospital):
        pass

    def select_time_windows(self, df, hospital):
        # Add ABX time in-place (avoids a full merge copy of the large df)
        abx_map = self.demog.set_index("stay_id")["time_hours_abx"]
        df["time_hours_abx"] = df["stay_id"].map(abx_map)

        sepsis_ids = self.patient_selectors[0].patients_ids[hospital]["sepsis"]
        control_ids = self.patient_selectors[0].patients_ids[hospital]["controls"]
        abx_idx = (
            (
                (df["stay_id"].isin(sepsis_ids)) &
                (df["time_difference_hours_sepsis"] <= -1)
            ) |
            (
                (df["stay_id"].isin(control_ids)) &
                (
                    (df["time_hours_abx"].isna()) |
                    (df["time_difference_hours"] <= df["time_hours_abx"] - self.hours_to_exclude_abx)
                )
            )
        )
        # Drop rows in-place so the original blocks are freed immediately
        df.drop(index=df.index[~abx_idx], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns="time_hours_abx", inplace=True)

        return self._select_time_windows(df, hospital)

    def select_dataset(self, df, hospital):
        df.reset_index(drop=True, inplace=True)
        df = self._add_labels(df, hospital)

        df_reduced = self.select_time_windows(df, hospital)

        df_matched = df_reduced.copy()

        for patient_selector in self.patient_selectors:
            for p in tqdm(self.sepsis_rate, desc=f"Patient matching ({hospital})"):
                logger.info(
                    "[%s] selector=%r  rate=%s  include_all_sepsis=%s",
                    hospital, patient_selector.name, p, self.include_all_sepsis,
                )
                if self.include_all_sepsis:
                    df_matched = patient_selector.select_patients(df_matched, hospital, p, include_all_sepsis=True)

                df_matched = patient_selector.select_patients(df_matched, hospital, p, include_all_sepsis=False)

        counts = df_matched["control_type"].value_counts().to_dict()
        logger.info("[%s] final composition: %s", hospital, counts)

        # Save final dataset
        logger.info("Saving final dataset (%s) ...", hospital)
        df_matched.to_parquet(
            os.path.join(
                self.path_to_checkpoint,
                f"df_cases_controls_{hospital}.parquet",
            )
        )

        # Save hash final dataset
        logger.info("Saving hash final dataset (%s) ...", hospital)
        with open(
            os.path.join(
                self.path_to_checkpoint,
                f"hash_{hospital}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(hash_pandas_object(df_matched, index=True).values, f)

    def select_HeRO_dataset(self, df, hero, hospital_name):
        df = self._add_labels(df, hospital_name)

        df_hero = df.merge(hero, on=["stay_id","time_difference_hours"], how="inner").reset_index(drop=True)

        logger.info("HeRO dataset merged for %s — %d rows", hospital_name, len(df_hero))
        # Save final dataset
        logger.info("Saving HeRO dataset (%s)...", hospital_name)
        df_hero.to_parquet(
            os.path.join(
                self.path_to_checkpoint,
                f"df_hero_{hospital_name}.parquet",
            )
        )

    def select_nsofa_dataset(self, df, hospital_name):
        sepsis_ids = self.patient_selectors[0].patients_ids[hospital_name]["sepsis"]
        sepsis = df[df["stay_id"].isin(sepsis_ids)]

        pre  = sepsis[(sepsis["time_difference_hours_sepsis"] >= -72) &
                      (sepsis["time_difference_hours_sepsis"] < 0)]
        post = sepsis[(sepsis["time_difference_hours_sepsis"] > 0) &
                      (sepsis["time_difference_hours_sepsis"] <= 72)]

        nsofa_min_pre  = pre.groupby("stay_id")["nSOFA_max"].min().rename("nsofa_min_pre")
        nsofa_max_post = post.groupby("stay_id")["nSOFA_max"].max().rename("nsofa_max_post")

        df_nsofa = pd.concat([nsofa_min_pre, nsofa_max_post], axis=1).reset_index()
        df_nsofa["delta_nsofa"] = df_nsofa["nsofa_max_post"] - df_nsofa["nsofa_min_pre"]

        logger.info("nSOFA dataset for %s — %d sepsis patients", hospital_name, len(df_nsofa))
        logger.info("Saving nSOFA dataset (%s)...", hospital_name)
        df_nsofa.to_parquet(
            os.path.join(
                self.path_to_checkpoint,
                f"df_nsofa_{hospital_name}.parquet",
            )
        )

class ResampleTimeWindowSelector(DatasetSelector):
    def __init__(self, config, demog):
        super().__init__(config, demog)
    
        self.resampling_frequency = self.config.experiment["time_window_selection"]["resampling_frequency"]

    def _select_time_windows(self, df, hospital):
        # Dynamically compute the time points before sepsis
        time_points = [
            -(i * self.shift_time_windows_sepsis)
            for i in range(self.nb_time_windows_sepsis)
            if i * self.shift_time_windows_sepsis <= self.forewarning_time
        ]
        time_points[0] = -(self.prediction_time)

        sepsis_ids = self.patient_selectors[0].patients_ids[hospital]["sepsis"]
        sepsis_idx = (
            (df["stay_id"].isin(sepsis_ids)) &
            (
                (df["time_difference_hours_sepsis"].isin(time_points)) | 
                (df["time_difference_hours_sepsis"] == -(self.self_controls_time))
            )
        )
        df_sepsis = df[sepsis_idx]
        df_sepsis.reset_index(drop=True, inplace=True)

        # Sort by stay_id and time for consistency (in-place avoids a sorted copy)
        df.sort_values(["stay_id", "time_difference_hours"], inplace=True)

        # Compute window_id = floor((time - min_time_per_stay) / freq)
        min_times = df.groupby("stay_id", sort=False)["time_difference_hours"].transform("min")
        df["window_id"] = ((df["time_difference_hours"] - min_times) // self.resampling_frequency).astype(int)

        # Randomly sample 1 row per (stay_id, window_id)
        df_regular = (
            df
            .groupby(["stay_id", "window_id"], group_keys=False, sort=False)
            .sample(n=1, random_state=42)
            .drop(columns=["window_id"])
        )
        df_regular.reset_index(drop=True, inplace=True)

        control_ids = self.patient_selectors[0].patients_ids[hospital]["controls"]
        regular_idx = (
            (df_regular["stay_id"].isin(control_ids)) |
            (
                (df_regular["stay_id"].isin(sepsis_ids)) &
                (df_regular["time_difference_hours_sepsis"] < -(self.self_controls_time))
            )
        )
        df_regular = df_regular[regular_idx]
        df_regular.reset_index(drop=True, inplace=True)

        df_resampled = pd.concat([df_sepsis, df_regular], axis=0, ignore_index=True)
        df_resampled.drop_duplicates(subset=["stay_id", "time_difference_hours"], inplace=True)
        df_resampled.sort_values(["stay_id", "time_difference_hours"], inplace=True)
        df_resampled.reset_index(drop=True, inplace=True)
        return df_resampled
    
class ContinuousTimeWindowSelector(DatasetSelector):
    def __init__(self, config, demog):
        super().__init__(config, demog)
    
        self.resampling_frequency = self.config.experiment["time_window_selection"]["resampling_frequency"]

    def _select_time_windows(self, df, hospital):
        # Sort by stay_id and time for consistency (in-place avoids a sorted copy)
        df.sort_values(["stay_id", "time_difference_hours"], inplace=True)

        sepsis_ids = self.patient_selectors[0].patients_ids[hospital]["sepsis"]
        sepsis_idx = (
            (df["stay_id"].isin(sepsis_ids)) &
            (
                (df["time_difference_hours_sepsis"] <= 0) &
                (df["time_difference_hours_sepsis"] >= -(self.self_controls_time + self.forewarning_time))
            )
        )
        df_sepsis = df[sepsis_idx]
        df_sepsis.reset_index(drop=True, inplace=True)

        def split_sepsis_blocks(df_sepsis):
            sepsis_blocks = []

            for stay_id, group in df_sepsis.groupby("stay_id"):
                window_counter = 0

                # Split 96h into 24h chunks
                for i, start_hour in enumerate(range(0, (self.self_controls_time + self.forewarning_time), self.forewarning_time)):
                    end_hour = start_hour + self.forewarning_time
                    block = group[
                        (group["time_difference_hours_sepsis"] <= -(start_hour + 1)) &
                        (group["time_difference_hours_sepsis"] >= -end_hour)
                    ].copy()

                    # Assign unique 24 window id
                    block["window_id"] = f"sepsis_{stay_id}_{window_counter}"
                    block["time_difference_hours_window"] = group["time_difference_hours_sepsis"] + start_hour

                    window_counter += 1

                    sepsis_blocks.append(block)

            return pd.concat(sepsis_blocks, axis=0).reset_index(drop=True)

        df_sepsis = split_sepsis_blocks(df_sepsis)

        # Define block size 
        block_size = self.resampling_frequency * self.forewarning_time  
        df["block_id"] = (df["time_difference_hours"] // block_size).astype(int)

        # For each stay_id and block_id, take the 24 consecutive hours 
        def sample_block(group):
            sampled_blocks = []
            window_counter = 0  # for unique window ids

            for block_id, group_block in group.groupby("block_id"):
                max_hour = group_block["time_difference_hours"].max()

                block_start = block_id * block_size
                block_end = min(max_hour, block_start + block_size)

                # Ensure there are at least 24 hours in the block
                max_start = block_end - self.forewarning_time
                if max_start < block_start:
                    start_hour = block_start
                else:
                    start_hour = np.random.randint(block_start, max_start + 1)

                end_hour = start_hour + self.forewarning_time
                sampled_block = group_block[
                    (group_block["time_difference_hours"] >= start_hour) &
                    (group_block["time_difference_hours"] < end_hour)
                ]

                # Assign unique 24 window id
                sampled_block["window_id"] = f"control_{group.name}_{window_counter}"
                sampled_block["time_difference_hours_window"] =  sampled_block["time_difference_hours"] - (end_hour - 1)
                window_counter += 1

                sampled_blocks.append(sampled_block)

            return pd.concat(sampled_blocks, axis=0) if sampled_blocks else pd.DataFrame()

        # Randomly sample 24 consecutive rows per (stay_id, block_id)
        # sort=False: data is already sorted by stay_id above, avoids internal re-sort copy
        df_regular = (
            df.groupby("stay_id", group_keys=False, sort=False)
            .apply(sample_block)
            .drop(columns=["block_id"])
        )
        df_regular.reset_index(drop=True, inplace=True)

        control_ids = self.patient_selectors[0].patients_ids[hospital]["controls"]
        regular_idx = (
            (df_regular["stay_id"].isin(control_ids)) |
            (
                (df_regular["stay_id"].isin(sepsis_ids)) &
                (df_regular["time_difference_hours_sepsis"] < -(self.self_controls_time + self.forewarning_time + 1))
            )
        )
        df_regular = df_regular[regular_idx]
        df_regular.reset_index(drop=True, inplace=True)

        df_resampled = pd.concat([df_sepsis, df_regular], axis=0, ignore_index=True)
        df_resampled.drop_duplicates(subset=["stay_id", "time_difference_hours"], inplace=True)
        df_resampled.sort_values(["stay_id", "time_difference_hours"], inplace=True)
        df_resampled.reset_index(drop=True, inplace=True)
        return df_resampled
        
class SepsisRiskSelector(DatasetSelector):
    def __init__(self, config, demog):
        super().__init__(config, demog)

        self.demog = demog
        self.resampling_factor = self.config.experiment["time_window_selection"]["resampling_factor"]

    def _time_sepsis_sampler(self, time):
        sepsis_times = self.demog.loc[self.demog["sepsis_halos"] == 1, "time_hours_sepsis"]

        kde = gaussian_kde(sepsis_times)

        weights = kde.evaluate(time)
        weights = np.clip(weights, a_min=1e-12, a_max=None)  # avoid zeros
        weights /= weights.sum()

        return weights

    def _select_time_windows(self, df, hospital):
        sepsis_ids = self.patient_selectors[0].patients_ids[hospital]["sepsis"]
        sepsis_idx = (
            (df["stay_id"].isin(sepsis_ids)) & 
            (df["time_difference_hours_sepsis"] <= -(self.prediction_time)) &
            (df["time_difference_hours_sepsis"] >= -(self.prediction_time + 2*self.forewarning_time))
        )
        df_sepsis = df[sepsis_idx]
        df_sepsis.reset_index(drop=True, inplace=True)

        control_ids = self.patient_selectors[0].patients_ids[hospital]["controls"]
        control_idx = (df["stay_id"].isin(control_ids))
        df_controls = df[control_idx]
        df_controls.reset_index(drop=True, inplace=True)
        df_controls["age_at_admission_days"] = df_controls["stay_id"].map(
            self.demog.set_index("stay_id")["age_at_admission_days"]
        )

        sampled_groups = []
        for _, group in df_controls.groupby("stay_id"):
            n_total = len(group)

            # Compute number of samples per patient
            n_sample = max(1, n_total // self.resampling_factor)

            # Compute weights for this patient's windows
            weights = self._time_sepsis_sampler(
                (group["time_difference_hours"] + 24 * group["age_at_admission_days"]).values
            )

            # Sample time windows based on weights
            sampled = group.sample(
                n=n_sample,
                weights=weights,
                random_state=42,
            )
            sampled_groups.append(sampled)

        df_controls = pd.concat(sampled_groups, axis=0)
        df_controls.reset_index(drop=True, inplace=True)
        df_controls.drop(columns="age_at_admission_days", inplace=True)

        df_resampled = pd.concat([df_sepsis, df_controls], axis=0)
        df_resampled.sort_values(["stay_id", "time_difference_hours"], inplace=True)
        df_resampled.reset_index(drop=True, inplace=True)
        return df_resampled