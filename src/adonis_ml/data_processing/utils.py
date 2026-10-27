from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Callable
import re
 
import pandas as pd


def cached_extract(
    path: str,
    extract_fn: Callable[[], pd.DataFrame],
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame from *path* if it exists, otherwise run *extract_fn*,
    persist the result as parquet, and return it."""
    if os.path.isfile(path) and not force:
        return pd.read_parquet(path)
    df = extract_fn()
    print(f"Saving {os.path.basename(path)}...")
    df.to_parquet(path)
    return df


def rename_hospitals(df: pd.DataFrame, hospitals: dict) -> pd.DataFrame:
    """Replace raw hospital identifiers with short names."""
    rename_map = {
        info["original_name"]: name
        for name, info in hospitals.items()
    }
    df["hospital"] = df["hospital"].replace(rename_map)
    return df


def resolve_parquet_path(
    hospital: str,
    file_name,
    variable_name: str,
    path_to_time_series: dict,
    variables_to_extract: dict,
) -> str | None :
    """Resolve the full filesystem path for a raw parquet file.
 
    Single source of truth for the hospital-specific directory layout."""
    if not isinstance(file_name, str):
        return None
 
    dir_path = path_to_time_series[hospital]
 
    if hospital == "CHUV":
        return os.path.join(dir_path, file_name)
 
    # Find the SPHN concept for this variable
    concept = None
    for variables in variables_to_extract.values():
        if variable_name in variables:
            concept = variables[variable_name]["concept"]
            break
 
    if concept is None:
        return None
 
    if hospital == "USZ":
        return os.path.join(dir_path, f"{concept}Measurement", file_name)
    elif hospital == "KISPI":
        dir_name = re.sub(r"(?<!^)(?=[A-Z])", "_", f"{concept}Measurement").lower()
        return os.path.join(dir_path, dir_name, file_name)
 
    return None