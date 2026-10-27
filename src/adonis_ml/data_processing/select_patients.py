from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AbstractPatientSelector(ABC):
    def __init__(self, config, demog):
        self.config = config
        self.demog = demog
        self.name = None 
        self._parse_args()
        self.patients_ids = self._build_patient_index()

    def _parse_args(self):
        self.max_gestational_age = self.config.experiment["params"]["max_gestational_age"]
        self.forewarning_time = self.config.experiment["params"]["forewarning_time"]
        self.nb_time_windows_sepsis = self.config.experiment["params"]["nb_time_windows_sepsis"]
        self.shift_time_windows_sepsis = self.config.experiment["params"]["shift_time_windows_sepsis"]
        self.prediction_time = self.config.experiment["params"]["prediction_time"]
        self.days_to_exclude = self.config.experiment["params"]["days_to_exclude"]
        self.self_controls = self.config.experiment["params"]["self_controls"]
        self.self_controls_time = self.config.experiment["params"]["self_controls_time"]
        self.transition = self.config.experiment["params"]["transition"]
        self.hospitals = self.config.experiment["hospitals"]
        self.patients_to_select = self.config.experiment["patients_to_select"]

    def get_demog(self, hospital=[]):
        if hospital is None:
            return self.demog.copy()
        else:
            return self.demog[self.demog["hospital"] == hospital].copy()

    def _build_patient_index(self) -> dict[str, dict[str, list]]:
        """Build a nested dict: hospital → patient_group → [stay_ids]."""
        return {
            hospital: {
                group_name: self._get_patient_ids(
                    hospital=hospital,
                    include_cols=group_info["include"],
                    exclude_cols=group_info["exclude"],
                )
                for group_name, group_info in self.patients_to_select.items()
            }
            for hospital, info in self.hospitals.items()
            if info["include"]
        }

    def _get_patient_ids(
        self,
        hospital: str,
        include_cols: list[str] = None,
        exclude_cols: list[str] = None,
        include_op: str = "any",
        exclude_op: str = "all",
    ) -> list:
        """Select stay_ids matching inclusion/exclusion criteria.

        Args:
            hospital: hospital short name.
            include_cols: columns that should be 1 (patient is included if
                *any* of them is 1 by default).
            exclude_cols: columns that should be 0 (patient is excluded
                unless *all* of them are 0 by default).
            include_op: "any" (union) or "all" (intersection) for inclusion.
            exclude_op: "all" (intersection) or "any" (union) for exclusion.

        Returns:
            List of stay_ids satisfying the criteria.
        """
        demog = self.get_demog(hospital)

        mask = pd.Series(True, index=demog.index)

        if include_cols:
            op = getattr(demog[include_cols].eq(1), include_op)
            mask &= op(axis=1)

        if exclude_cols:
            op = getattr(demog[exclude_cols].eq(0), exclude_op)
            mask &= op(axis=1)

        if self.max_gestational_age:
            mask &= demog["ga_weeks"] < self.max_gestational_age

        return demog.loc[mask, "stay_id"].tolist()

    def select_patients(self, df, hospital, sepsis_rate, include_all_sepsis, random_state=42):
        df = self._select_patients(df, hospital, sepsis_rate, include_all_sepsis, random_state)
        assert self._get_identifier(sepsis_rate, include_all_sepsis) in df.columns.values       
        return df

    @abstractmethod
    def _select_patients(self, df, hospital, include_all_sepsis, random_state=42):
        pass

    def _get_identifier(self, sepsis_rate, include_all_sepsis):
        if include_all_sepsis:
            return f"train_{self.name}_{sepsis_rate}_all"
        else:
            return f"train_{self.name}_{sepsis_rate}"


class PatientSelectorkNN(AbstractPatientSelector):
    def __init__(self, config, demog):
        super().__init__(config, demog)

        self.name = "knn"

        self.matching_strategy = self.config.experiment["patient_selection"]["knn"]
        self.nb_time_windows_control = self.matching_strategy["nb_time_windows_control"]
        self.shift_time_windows_controls = self.matching_strategy["shift_time_windows_controls"]

        # Matching variables
        self.match_vars_num = self.matching_strategy["columns"]["numerical"]
        self.match_vars_cat = self.matching_strategy["columns"]["categorical"]

    def get_controls(self, df, hospital, sepsis_rate, random_state=None):
        # Load demographic table
        demog = self.get_demog(hospital)

        control_ids_unique = df["stay_id"].unique()
        control_ids = self.patients_ids[hospital]["controls"]
        control_ids_df = [stay_id for stay_id in control_ids if stay_id in control_ids_unique]
        sepsis_ids = self.patients_ids[hospital]["sepsis"]

        if len(sepsis_ids) == 0:
            raise ValueError(f"No sepsis patients available for hospital {hospital}.")
        if len(control_ids_df) == 0:
            return pd.DataFrame(columns=["control_id", "matched_sepsis_id", "time_hours_sepsis", "time_hours_ECM"])

        feats = demog[["stay_id"] + self.match_vars_num + self.match_vars_cat].dropna().set_index("stay_id")

        # One-hot encode categorical vars efficiently
        feats = pd.get_dummies(feats, columns=self.match_vars_cat, prefix=self.match_vars_cat)
        X_all = feats.copy()

        # Normalize numerical vars
        scaler = StandardScaler()
        X_all[self.match_vars_num] = scaler.fit_transform(X_all[self.match_vars_num])

        # Split into sepsis and control matrices
        X_sepsis = X_all.loc[X_all.index.intersection(sepsis_ids)]
        X_controls = X_all.loc[X_all.index.intersection(control_ids_df)]

        # Precompute neighbor indices
        n_neighbors = len(X_controls)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(X_controls.values)

        i, neighbor_indices = nn.kneighbors(X_sepsis.values)

        # Precompute maps
        age_at_admission_map = dict(zip(demog["stay_id"], demog["age_at_admission_days"]))
        time_hours_sepsis_map = dict(zip(demog["stay_id"], demog["time_hours_sepsis"]))
        time_hours_abx_map = dict(zip(demog["stay_id"], demog["time_hours_abx"]))
        time_hours_max_map = dict(zip(df["stay_id"].unique(), df.groupby("stay_id")["time_difference_hours"].max()))

        # Matching 
        matches = []
        used_controls = set()
        
        if self.self_controls:
            target_controls_per_sepsis = (self.nb_time_windows_sepsis / self.nb_time_windows_control) * ((1 - 2 * sepsis_rate) / sepsis_rate)
        else:
            target_controls_per_sepsis = (self.nb_time_windows_sepsis / self.nb_time_windows_control) * ((1 - sepsis_rate) / sepsis_rate)

        n_sepsis = len(sepsis_ids)
        offset = int(np.ceil((target_controls_per_sepsis % 1) * n_sepsis))
        target_controls_per_sepsis = int(np.ceil(target_controls_per_sepsis))

        sepsis_ids_array = np.array(X_sepsis.index)
        control_ids_array = np.array(X_controls.index)

        for i in tqdm(range(target_controls_per_sepsis), desc=f"Matching control patients ({hospital})"):
            for j, sepsis_id in enumerate(sepsis_ids_array):

                if (
                    (i == (target_controls_per_sepsis - 1)) &
                    (j > offset)
                ):
                    break

                time_hours_sepsis = time_hours_sepsis_map[sepsis_id]

                for control_idx in neighbor_indices[j][i:]:
                    control_id = control_ids_array[control_idx]

                    if control_id in used_controls:
                        continue

                    age_at_admission_days = age_at_admission_map[control_id]
                    time_hours_abx = time_hours_abx_map.get(control_id, np.nan)
                    time_hours_max = time_hours_max_map.get(control_id, np.nan)
                    time_hours_ECM = time_hours_sepsis - 24 * age_at_admission_days

                    # Validity checks
                    is_valid_abx = (
                        (np.isnan(time_hours_abx)) |
                        (time_hours_abx > time_hours_ECM)
                    )
                    is_valid_data = (
                        (time_hours_ECM > 0) &
                        (time_hours_ECM <= time_hours_max) &
                        (not np.isnan(time_hours_max))
                    )

                    if is_valid_abx and is_valid_data:
                        used_controls.add(control_id)
                        matches.append({
                            "control_id": control_id,
                            "matched_sepsis_id": sepsis_id,
                            "time_hours_sepsis": float(time_hours_sepsis),
                            "time_hours_ECM": float(time_hours_ECM)
                        })
                        break  

        if not matches:
            return pd.DataFrame(columns=["control_id", "matched_sepsis_id", "time_hours_sepsis", "time_hours_ECM"])
        return pd.DataFrame(matches)

    def _select_patients(self, df, hospital, sepsis_rate, include_all_sepsis, random_state=42):
        columns_id = self._get_identifier(sepsis_rate, include_all_sepsis)

        df = (
            df
            .reset_index(drop=True)
            .copy()
        )
        df[columns_id] = 0

        # Sepsis patient selection
        time_points = [
            -(i * self.shift_time_windows_sepsis + 1)
            for i in range(self.nb_time_windows_sepsis)
            if i * self.shift_time_windows_sepsis <= self.forewarning_time
        ]
        time_points[0] = -(self.prediction_time)

        stay_ids_unique = df["stay_id"].unique()
        sepsis_ids = self.patients_ids[hospital]["sepsis"]
        sepsis_ids_df = [stay_id for stay_id in sepsis_ids if stay_id in stay_ids_unique]
        control_ids = self.patients_ids[hospital]["controls"]
        control_ids_df = [stay_id for stay_id in control_ids if stay_id in stay_ids_unique]
        logger.info(
            "[%s] kNN selector — rate=%s  include_all_sepsis=%s  n_sepsis=%d  n_controls=%d",
            hospital, sepsis_rate, include_all_sepsis, len(sepsis_ids_df), len(control_ids_df),
        )

        sepsis_idx = (
            (df["stay_id"].isin(sepsis_ids)) &
            (
                (df["time_difference_hours_sepsis"].isin(time_points)) |
                (df["time_difference_hours_sepsis"] == -(self.self_controls_time))
            )
        )
        df.loc[sepsis_idx, columns_id] = 1
        logger.info("[%s] kNN — sepsis windows selected: %d", hospital, int(sepsis_idx.sum()))

        # Control patient selection
        control_patients_df = (
            self.get_controls(
                df, hospital, sepsis_rate
            )
            .rename(columns={
                "control_id": "stay_id"
            })
            .drop(columns=["time_hours_sepsis"])
        )

        df_controls = df.merge(control_patients_df, on="stay_id", how="left")
        df_controls["time_difference_hours_ECM"] = df_controls["time_difference_hours"] - df_controls["time_hours_ECM"]

        # Define desired sampling points
        target_times = time_points = [
            -(i * self.shift_time_windows_controls)
            for i in range(self.nb_time_windows_control)
        ]

        selected_indices = []

        # Iterate per patient to find closest available time for each target window
        matched_controls = df_controls[~df_controls["time_hours_ECM"].isna()]
        n_controls = matched_controls["stay_id"].nunique()
        for _, group in tqdm(matched_controls.groupby("stay_id"), total=n_controls, desc=f"Selecting control windows ({hospital})"):
            group = group.sort_values("time_difference_hours_ECM")

            candidates = group[group["time_difference_hours_ECM"] <= 0]
            if candidates.empty:
                continue
            max_hour = candidates.iloc[-1]["time_difference_hours_ECM"]

            chosen_rows = []
            for target_time in target_times:
                # Find the closest available time ≤ target_time and > last_chosen_time
                candidates = group[group["time_difference_hours_ECM"] <= (max_hour + target_time)]
                if not candidates.empty:
                    chosen_row = candidates.iloc[-1]  # last (closest to target_time)
                    chosen_rows.append(chosen_row.name)

            selected_indices.extend(chosen_rows)

        # Mark selected windows
        control_idx = np.zeros(df.shape[0], dtype=bool)
        control_idx[selected_indices] = True
        df.loc[control_idx, columns_id] = 1
        logger.info("[%s] kNN — control windows selected: %d", hospital, int(control_idx.sum()))

        # Add groups variable
        group_column = f"groups_knn_{sepsis_rate}"
        df[group_column] = df_controls["matched_sepsis_id"].fillna(df_controls["stay_id"])

        return df


class RandomPatientSelector(AbstractPatientSelector):
    def __init__(self, config, demog):
        super().__init__(config, demog)

        self.name = "random"

    def _select_patients(self, df, hospital, sepsis_rate, include_all_sepsis, random_state=42):
        np.random.seed(random_state)
        rng = np.random.default_rng(random_state)

        columns_id = self._get_identifier(sepsis_rate, include_all_sepsis)

        df = (
            df
            .reset_index(drop=True)
            .copy()
        )
        df[columns_id] = 0

        # Sepsis patient selection
        if include_all_sepsis:
            time_points = [
                -(i + 1)
                for i in range(0, self.forewarning_time)
            ]
        else:
            time_points = [
                -(i * self.shift_time_windows_sepsis + 1)
                for i in range(self.nb_time_windows_sepsis)
                if i * self.shift_time_windows_sepsis <= self.forewarning_time
            ]
            time_points[0] = -(self.prediction_time)

        stay_ids_unique = df["stay_id"].unique()
        sepsis_ids = self.patients_ids[hospital]["sepsis"]
        sepsis_ids_df = [stay_id for stay_id in sepsis_ids if stay_id in stay_ids_unique]
        sepsis_idx = (
            (df["stay_id"].isin(sepsis_ids)) &
            (
                (df["time_difference_hours_sepsis"].isin(time_points)) |
                (df["time_difference_hours_sepsis"] == -(self.self_controls_time))
            )
        )
        df.loc[sepsis_idx, columns_id] = 1

        # Control patient selection
        control_ids = self.patients_ids[hospital]["controls"]
        control_ids_df = [stay_id for stay_id in control_ids if stay_id in stay_ids_unique]

        if include_all_sepsis:
            n_sepsis = len(sepsis_ids_df) * self.forewarning_time
        else:
            n_sepsis = len(sepsis_ids_df) * self.nb_time_windows_sepsis
        n_controls_total = len(control_ids_df)

        n_self_controls = 0
        if self.self_controls:
            n_self_controls = len(sepsis_ids_df)

        # Desired number of controls based on sepsis_rate
        n_controls_target = int((n_sepsis / sepsis_rate) - (n_sepsis + n_self_controls))

        logger.info(
            "[%s] random selector — rate=%s  include_all_sepsis=%s  "
            "n_sepsis_patients=%d  n_sepsis_windows=%d  n_controls_patients=%d  "
            "n_controls_target=%d  samples_per_patient=%d",
            hospital, sepsis_rate, include_all_sepsis,
            len(sepsis_ids_df), n_sepsis, n_controls_total,
            n_controls_target, int(np.ceil(n_controls_target / n_controls_total)),
        )

        samples_per_patient = int(np.ceil(n_controls_target / n_controls_total))
        sampled = (
            df[df["stay_id"].isin(control_ids_df)]
            .groupby("stay_id")
            .sample(n=samples_per_patient, replace=True, random_state=random_state)
            .drop_duplicates()
        )

        # Randomly drop excess rows to match target count
        n_excess = len(sampled) - n_controls_target
        if n_excess > 0:
            removable_idx = (
                sampled
                .groupby("stay_id")
                .sample(n=1, random_state=random_state)
            ).index
            drop_idx = rng.choice(removable_idx, size=n_excess, replace=False)
            sampled = sampled.drop(index=drop_idx)

        control_idx = np.zeros(df.shape[0], dtype=bool)
        control_idx[sampled.index.values] = True

        df.loc[control_idx, columns_id] = 1
        logger.info("[%s] random — control windows selected: %d", hospital, int(control_idx.sum()))
        return df
    
patient_selector_dict = {
    "knn": PatientSelectorkNN,
    "random": RandomPatientSelector,
}