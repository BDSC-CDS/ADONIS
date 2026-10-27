import logging
import os
import pickle

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_adonis_feature_dataset_new(config):
    df = pd.read_parquet(
        os.path.join(config.paths["data_path"], config.paths["dataset_file_name"])
    )
    with open(
        os.path.join(config.paths["data_path"], config.paths["list_file_name"]),
        "rb",
    ) as f:
        list_of_variables = pickle.load(f)

    return df, list_of_variables


def extend_list_of_variables_upstream(config, list_of_variables):
    for group, features in list_of_variables.items():
        if group in config.dataset["extended_feature_groups"]:
            list_of_variables_upstream = []

            for feature in features:
                if group == "vitals" and (
                    feature
                    in [f"{signal}_mean" for signal in config.dataset["signals"]]
                ):
                    continue

                list_of_variables_upstream.extend(
                    [
                        f"{feature}_{aggregate}"
                        for aggregate in config.dataset["aggregates"]
                    ]
                )

            list_of_variables[group].extend(list_of_variables_upstream)

    return list_of_variables


def get_and_extend_list_of_variables(config, list_of_variables):
    list_of_variables_ext = {
        "vitals": [
            f"{signal}_{aggregate}"
            for signal in config.dataset["signals"]
            for aggregate in ["mean", "median"]
        ],
        "demographics": config.dataset["demographics"],
        "measurements": config.dataset["measurements"],
    }
    for key in [
        "spikes_drops",
        "spikes_drops_extreme",
        "trends",
        "trend",
        "std_feat",
        "kurt_skew",
        "entropy",
    ]:
        if key in list_of_variables.keys():
            list_of_variables_ext[key] = list_of_variables[key]

    list_of_variables_ext = extend_list_of_variables_upstream(
        config,
        list_of_variables_ext,
    )
    return list_of_variables_ext


def load_adonis_feature_dataset(config):
    data_load_fn = load_adonis_feature_dataset_new
    df, list_of_variables_orig = data_load_fn(config)

    list_of_variables = get_and_extend_list_of_variables(config, list_of_variables_orig)

    print(f"Shape of loaded dataframe: {df.shape}")
    print(
        f"Found {df['ref_septic_patient'].value_counts().shape[0]} sepsis patients, with {df.shape[0]} samples (sepsis, self-control, match-control) overall"
    )
    return df, list_of_variables


def drop_columns(config, columns_names):
    columns_to_drop = config.dataset["columns_to_drop"]

    for feature_name, feature_info in config.features.items():
        if feature_info["extract"]:
            if not feature_info["upstream"]:
                for column_name in columns_names:
                    for feature_identifier in feature_info["identifier"]:
                        for aggregate in config.dataset["aggregates"]:
                            if (
                                feature_identifier in column_name
                                and column_name.endswith(aggregate)
                            ):
                                columns_to_drop.append(column_name)
        else:
            for column_name in columns_names:
                for feature_identifier in feature_info["identifier"]:
                    if feature_identifier in column_name:
                        columns_to_drop.append(column_name)

    return [
        column_name
        for column_name in columns_names
        if column_name not in columns_to_drop
    ]


def get_list_of_features_from_df(config, df):
    list_of_features = drop_columns(config, df.columns.values)

    logger.info("Number of features (after excluding 'columns_to_drop'): %s", len(list_of_features))

    list_of_features = [
        feature for feature in list_of_features if "dist" not in feature
    ]
    return list_of_features

def get_sampling_column(config):
    return config.model["undersampler"].sampling_column

def get_sklearn_inputs(config, df):
    list_of_features = get_list_of_features_from_df(config, df)
    sampling_column = get_sampling_column(config) 

    arr = sampling_column.split("_")
    sampling_startegy = arr[1]
    sepsis_rate = arr[2]

    meta_cols = [sampling_column]
    if "train_knn_0.1" != sampling_column:
        meta_cols.append("train_knn_0.1")
    if sampling_startegy == "knn":
        groups_col = f"groups_knn_{sepsis_rate}"
        if groups_col in df.columns and groups_col not in meta_cols:
            meta_cols.append(groups_col)

    X = df[list_of_features + meta_cols]
    y = df["sepsis_time"]

    if sampling_startegy == "knn":
        groups = df[f"groups_knn_{sepsis_rate}"].values
    else:
        groups = df["stay_id"].values

    t = df["time_difference_hours_sepsis"].values if "time_difference_hours_sepsis" in df.columns else None
    if t is not None:
        eclipse_mask = pd.notna(t) & (t >= -72) & (t < -12)
    else:
        eclipse_mask = np.zeros(len(df), dtype=bool)

    return X, y, groups, list_of_features, sampling_column, eclipse_mask

def get_df_nsofa(config):
    return pd.read_parquet(
        os.path.join(config.paths["data_path"], "df_nsofa_CHUV.parquet")
    )


def get_df_hero(config):
    """Load the hero parquet produced by BioMedITDataLoader._load_hero().

    Columns: stay_id, time_difference_hours, hero_max, source_hero.
    hero_max is the maximum HeRO score within each hour for that patient.
    source_hero is either 'MV5_NAT' or 'internal register'.
    """
    return pd.read_parquet(
        os.path.join(config.mlflow["path_to_checkpoint"], "hero.parquet")
    )

def get_group_map_from_df(df):
    # Get unique sepsis patients 
    sepsis_ids = df.loc[df["control_type"] == "sepsis", "stay_id"].unique()

     # Get unique control patients 
    control_ids = df.loc[df["control_type"] == "control", "stay_id"].unique()

    # Split all controls as evenly as possible across sepsis groups
    control_chunks = np.array_split(control_ids, len(sepsis_ids))

    # Assign group IDs
    group_map = {}
    for group_id, (sepsis_id, control_ids) in enumerate(zip(sepsis_ids, control_chunks)):
        group_map[sepsis_id] = group_id

        # Map each control stay_id to this group
        for control_id in control_ids:
            group_map[control_id] = group_id

    return group_map


def get_subgroups_from_df(config, df, demog):
    """
    Generate subgroup categorical variables based on rules in config.subgroups.
    
    Parameters
    ----------
    config : object
        Configuration object with a `.subgroups` dict. Each key should define:
          - type: 'categorical', 'range', 'date', or 'custom'
          - mapping or bins/range depending on type
    df : pd.DataFrame
        Feature-level dataset containing 'stay_id' and columns to transform.
    demog : pd.DataFrame
        Demographics table containing at least:
        ['stay_id', 'birth_weight', 'death_nat', 'death_date', 'birth_date']

    Returns
    -------
    pd.DataFrame
        DataFrame of transformed subgroup variables.

    """
    df_extended = df.merge(
        demog[["stay_id", "birth_weight", "death_nat", "death_date", "birth_date", "death_hospital_7d", "death_hospital_30d"]],
        on="stay_id", how="left"
    )
    
    df_extended["time_hours_death"] = np.where(
        (df_extended["death_date"] is not pd.NaT),
        (df_extended["death_date"] - df_extended["birth_date"]).dt.total_seconds() // 3600,
        np.nan,
    )

    _nsofa_keys = {"delta_nsofa", "nsofa_min_pre", "nsofa_max_post"}
    if _nsofa_keys & set(config.subgroups.keys()):
        df_nsofa = get_df_nsofa(config)
        df_extended = df_extended.merge(
            df_nsofa[["stay_id", "nsofa_min_pre", "nsofa_max_post", "delta_nsofa"]],
            on="stay_id", how="left",
        )

    df_subgroups = df_extended[list(config.subgroups.keys())].copy()

    for group_name, group_info in config.subgroups.items():
        col = df_subgroups[group_name]
        group_type = group_info["type"]

        if group_type == "categorical":
            reverse_map = {v: k for k, v in group_info["mapping"].items()}
            df_subgroups[group_name] = col.map(reverse_map).astype("category")
        elif group_type == "range":
            bins_info = group_info["bins"]
            labels = list(bins_info.keys())
            intervals = list(bins_info.values())

            # Flatten intervals into a list of edges
            # Ensure bins are sorted in ascending order by lower bound
            intervals = sorted(intervals, key=lambda x: x[0])

            # Build bin edges for pd.cut
            bin_edges = [intervals[0][0]] + [high for (_, high) in intervals]

            df_subgroups[group_name] = pd.cut(
                col,
                bins=bin_edges,
                labels=labels,
                include_lowest=True,
                right=True
            )
        elif group_type == "date":
            min_time, max_time = group_info["range"]

            weeks = (col // max_time).astype(int)
            weeks = weeks.clip(upper=max_time) 
            labels = [str(i) for i in range(0, max_time)] + [f">{max_time}"]

            df_subgroups[group_name] = pd.Categorical(
                weeks.replace(7, f">{max_time}").astype(str),
                categories=labels,
                ordered=True
            )
        elif group_type == "custom":
            reverse_map = {v: k for k, v in group_info["mapping"].items()}

            if group_name == "death_date":
                col = df_extended.apply(
                    lambda row: (
                        1
                        if (
                            (row["death_nat"] == 1) &
                            (
                                (row["time_hours_sepsis"] is pd.NaT) |
                                (row["time_hours_sepsis"] <= row["time_hours_death"] <= row["time_hours_sepsis"] + 24 * 7)
                            )
                        )
                        else 0
                    ),
                    axis=1,
                )
                df_subgroups[group_name] = col.map(reverse_map).astype("category")
            elif group_name == "time_difference_hours_sepsis":
                min_time, max_time = group_info["range"]
                mask = (
                    df_extended["time_difference_hours_sepsis"].notna() &
                    (df_extended["time_difference_hours_sepsis"] > min_time) &
                    (df_extended["time_difference_hours_sepsis"] < max_time)
                )
                df_subgroups[group_name] = mask.map({True: reverse_map[1], False: reverse_map[0]}).astype("category")
            else:
                raise ValueError(f"Unhandled custom group: {group_name}")
        else:
            raise ValueError(f"Unknown subgroup type '{group_type}' for {group_name}")

    return df_subgroups

def get_weights_from_df(config, df, demog):
    df_extended = df.merge(
        demog[["stay_id", "age_at_admission_days"]],
        on="stay_id", how="left"
    )
    
    # Compute Post-Natal Age (PNA)
    df_extended["time_difference_weeks"] = df_extended["time_difference_hours"] // (24 * 7)
    df_extended["pna_weeks"] = df_extended["ga_weeks"] + (df_extended["age_at_admission_days"] // 7) + df_extended["time_difference_weeks"]

    tuning_strategy = config.params["tuning_strategy"]

    if tuning_strategy == "decay":
        decay_rate = config.params["decay_rate"]
        weights = np.exp(-decay_rate * (df_extended["pna_weeks"] - df_extended["ga_weeks"].min()))
        weights = weights.values
    elif tuning_strategy == "cutoff":
        cutoff = config.params["cutoff"]
        weights = (df_extended["pna_weeks"].values <= cutoff).astype(int)
    elif tuning_strategy == "all":
        weights = np.ones(shape=(df_extended.shape[0],))
    else:
        raise NotImplementedError
        
    # weights = weights / weights.sum()

    return weights
