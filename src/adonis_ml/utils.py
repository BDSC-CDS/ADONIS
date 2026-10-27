import json
from typing import Any

import pandas as pd


class AdonisConfig:
    """
    Utility class to turn config dict into class with one attribute for each dict key.
    """

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError(f"dict expected, found {type(config).__name__}")

        self._raw = config
        for key, value in self._raw.items():
            setattr(self, key, value)

    def return_config_dict(self):
        config_dict = {}
        for key, value in self._raw.items():
            config_dict[key] = value
        return config_dict

    def get_tags(self):
        return self.experiment["tags"]

    def get_params(self):
        return self.experiment["params"]


def load_config(filename):
    with open(filename, "r") as f:
        return json.load(f)


def parse_config(config_dict):
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("arg", type=Any)
    anyobject = parser.parse_object({"arg": config_dict})
    obj = parser.instantiate_classes(anyobject).arg
    return obj


def load_and_parse_config(filename):
    config_dict = load_config(filename)
    return parse_config(config_dict)


def check_column_names_and_ranges(df: pd.DataFrame, columns_to_check: dict):
    column_names = df.columns.values

    for column_name, columns_info in columns_to_check.items():
        assert (
            column_name in column_names
        ), f"Missing required column: '{column_name}' not found in the specified dataset"

        try:
            min = columns_info["min"]
            try:
                max = columns_info["max"]
            except KeyError:
                assert (
                    df.loc[~df[column_name].isna(), column_name].ge(min).all()
                ), f"Invalid value detected: '{column_name}' contains entries outside the expected range"
                continue
            assert (
                df.loc[~df[column_name].isna(), column_name]
                .between(min, max, inclusive="both")
                .all()
            ), f"Invalid value detected: '{column_name}' contains entries outside the expected range"
        except KeyError:
            continue


def check_controls(df: pd.DataFrame, demog: pd.DataFrame):
    # Check counts
    counts = df["control_type"].value_counts()

    for control_type in ["self_control", "matched"]:
        assert (
            counts["sepsis"] >= counts[control_type]
        ), f"The number of 'sepsis' patients must be greater than or equal to the number of '{control_type}' patients"

    # Inclusion criteria controls
    matched = demog.merge(
        df.loc[df["control_type"] == "matched", ["stay_id", "ref_septic_patient"]],
        on="stay_id",
        how="right",
    )
    sepsis = demog.merge(
        matched["ref_septic_patient"],
        left_on="stay_id",
        right_on="ref_septic_patient",
        how="right",
    )

    # 1. Dead patients in NAT
    variable_name = "death_nat"
    assert (
        matched.loc[~matched[variable_name].isna(), variable_name].eq(0).all()
    ), f"Invalid value detected: '{variable_name}' contains entries outside the expected range"

    # 2. Antibiotics
    variable_name = "time_hours_abx"
    time_hours_abx = matched.loc[
        ~matched["time_hours_abx"].isna(), "time_hours_abx"
    ] + (24 * matched.loc[~matched["time_hours_abx"].isna(), "age_at_admission_days"])
    assert time_hours_abx.ge(
        sepsis.loc[time_hours_abx.index, "time_hours_sepsis"]
    ).all(), f"Invalid value detected: '{variable_name}' contains entries outside the expected range"
