"""
Clinical score computation.

Each score is a self-contained class that:
    1. Declares its required DataFrame columns via `required_columns`.
    2. Declares which columns it produces via `output_columns`.
    3. Declares which columns should be dropped after all scores run via `cleanup_columns`.
    4. Implements `compute(df) -> df`.

Scores are driven by config:

    scores:
      nSOFA:
        compute: true
      VIS:
        compute: true
        drugs:
          dopamine:  { unit: "ug", weight: 1 }
          dobutamine: { unit: "ug", weight: 1 }
          epinephrine: { unit: "ug", weight: 100 }
          norepinephrine: { unit: "ug", weight: 100 }
          milrinone: { unit: "ug", weight: 10 }
          vasopressin: { unit: "m[IU]", weight: 10 }
      sf_ratio:
        compute: true
      delta_signals:
        compute: true
        exclude: ["inotropes", "steroids", "PLT"]

Usage:
    scorer = ScoreEngine(config.experiment["scores"])
    df = scorer.compute_all(df, available_measurements=["HR", "SpO2", ...])
"""

from __future__ import annotations
import logging
import sys
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SCORES
# ---------------------------------------------------------------------------

class Score(ABC):
    """Base class for all derived clinical scores / post-processors."""

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Score-specific configuration from the experiment config.
                    Can be None for scores that need no parameters.
        """
        self.config = config or {}

    @abstractmethod
    def required_columns(self) -> set[str]:
        """Columns that must be present in the DataFrame for this score to run."""
        pass

    @abstractmethod
    def output_columns(self) -> set[str]:
        """Columns this score adds to the DataFrame."""
        pass

    missing_fill_value: float = np.nan
    """Value used to fill output columns when required inputs are absent."""

    def cleanup_columns(self) -> set[str]:
        """Columns that should be dropped after ALL scores have run.

        Useful for intermediate columns (e.g., 'inotropes', 'PLT') that are
        consumed by a score but shouldn't appear in the final output.
        Override in subclasses as needed.
        """
        return set()

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the score and add result column(s) to df."""
        pass

class NSOFAScore(Score):
    """Neonatal Sequential Organ Failure Assessment score"""

    missing_fill_value: float = 0.0

    def required_columns(self) -> set[str]:
        return {
            "SpO2", "FiO2", "invasive",
            "steroids", "inotropes", "inotropes_count",
            "PLT",
            "stay_id", "time", "patient_id",
        }

    def output_columns(self) -> set[str]:
        return {"nSOFA"}

    def cleanup_columns(self) -> set[str]:
        return {"inotropes", "inotropes_count", "steroids", "PLT"}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["nSOFA"] = 0

        # Respiratory score
        invasive = df["invasive"] == 1
        ox = df["SpO2"] / df["FiO2"]
        df.loc[invasive & (ox < 1), "nSOFA"] = 8
        df.loc[invasive & (ox >= 1) & (ox < 1.5), "nSOFA"] = 6
        df.loc[invasive & (ox >= 1.5) & (ox < 2), "nSOFA"] = 4
        df.loc[invasive & (ox >= 2) & (ox < 3), "nSOFA"] = 2

        # Steroids — direct rolling (single patient, no groupby loop)
        df["steroids"] = df["steroids"].fillna(0)
        df["steroids"] = (
            df.set_index("time")["steroids"]
            .rolling("12h", closed="both")
            .max()
            .reset_index(drop=True)
            .values >= 1
        ).astype(int)

        df["inotropes"] = df["inotropes"].fillna(0)
        df["inotropes_count"] = df["inotropes_count"].fillna(0)

        ster = df["steroids"]
        ino = df["inotropes"]
        ino_cnt = df["inotropes_count"]

        df.loc[(ster == 1) & (ino == 0), "nSOFA"] += 1
        df.loc[(ster == 0) & (ino == 1) & (ino_cnt == 1), "nSOFA"] += 2
        df.loc[
            ((ster == 1) & (ino == 1) & (ino_cnt == 1))
            | ((ster == 0) & (ino == 1) & (ino_cnt >= 2)),
            "nSOFA",
        ] += 3
        df.loc[(ster == 1) & (ino == 1) & (ino_cnt >= 2), "nSOFA"] += 4

        # PLT — direct ffill (single patient, no groupby)
        df["PLT"] = df["PLT"].ffill()
        plt_vals = df["PLT"]

        df.loc[plt_vals < 50, "nSOFA"] += 3
        df.loc[(plt_vals >= 50) & (plt_vals < 100), "nSOFA"] += 2
        df.loc[(plt_vals >= 100) & (plt_vals <= 149), "nSOFA"] += 1

        return df


class VISScore(Score):
    """Vasoactive-Inotropic Score"""

    missing_fill_value: float = 0.0

    # Default drug configuration
    DEFAULT_DRUGS = {
        "dopamine": {
            "unit": "ug",
            "weight": 1
        },
        "dobutamine": {
            "unit": "ug",    
            "weight": 1
        },
        "epinephrine": {
            "unit": "ug",    
            "weight": 100
        },
        "norepinephrine": {
            "unit": "ug",    
            "weight": 100
        },
        "milrinone": {
            "unit": "ug",    
            "weight": 10
        },
        "vasopressin": {
            "unit": "m[IU]", 
            "weight": 10
        },
    }

    def required_columns(self) -> set[str]:
        cols = {"weight", "duration_time"}
        for drug_name in self.DEFAULT_DRUGS:
            cols.add(f"{drug_name}_dose")
            cols.add(f"{drug_name}_unit")
        return cols

    def output_columns(self) -> set[str]:
        cols = {"VIS"}
        for drug_name in self.DEFAULT_DRUGS:
            cols.add(f"{drug_name}_dose_normalized")
        return cols

    def cleanup_columns(self) -> set[str]:
        # Drug detail columns are consumed by VIS and not needed downstream
        cols = {"duration_time", "duration_unit"}
        for drug_name in self.DEFAULT_DRUGS:
            cols.add(f"{drug_name}_dose")
            cols.add(f"{drug_name}_unit")
        return cols

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        weight_kg = df["weight"] / 1000

        vis = pd.Series(0.0, index=df.index)

        for drug_name, drug_info in self.DEFAULT_DRUGS.items():
            dose_col = f"{drug_name}_dose"
            unit_col = f"{drug_name}_unit"
            expected_unit = drug_info["unit"]
            weight_factor = drug_info["weight"]

            dose = df[dose_col].copy()
            duration = df["duration_time"].copy()
            unit = df[unit_col].copy()

            # Zero out rows with unexpected units
            dose = dose.where(unit == expected_unit, np.nan)

            # Convert (unit) to (unit/min) then normalize by weight
            normalized = (dose / duration) / weight_kg
            df[f"{drug_name}_dose_normalized"] = normalized

            vis += weight_factor * normalized.fillna(0)

        df["VIS"] = vis
        return df


class SFRatioScore(Score):
    """SpO2 / FiO2 ratio."""

    def required_columns(self) -> set[str]:
        return {"SpO2", "FiO2"}

    def output_columns(self) -> set[str]:
        return {"sf_ratio"}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sf_ratio"] = df["SpO2"] / df["FiO2"]
        return df


score_dict: dict[str, type[Score]] = {
    "nSOFA": NSOFAScore,
    "VIS": VISScore,
    "sf_ratio": SFRatioScore,
}


# ---------------------------------------------------------------------------
# SCORE ENGINE
# ---------------------------------------------------------------------------

class ScoreEngine:
    """
    Orchestrates score computation based on config.

    Usage:
        engine = ScoreEngine(config)
        df = engine.compute(df, available_measurements=["HR", "SpO2", ...])
    """

    def __init__(self, config: dict):
        self.config = config
        self._parse_args()

        # Score handlers
        self.scores: list[Score] = []

        for score_name, score_config in self.scores_to_compute.items():
            if not score_config.get("compute", False):
                continue

            if score_name not in score_dict:
                logger.warning("Unknown score '%s' in config, skipping.", score_name)
                continue

            score_cls = score_dict[score_name]
            self.scores.append(score_cls(score_config))

        logger.info(
            "ScoreEngine initialized with: %s",
            [type(s).__name__ for s in self.scores],
        )

    def _parse_args(self):
        self.scores_to_compute = self.config.experiment["scores"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all enabled scores on the DataFrame."""
        all_cleanup_cols: set[str] = set()

        for score in self.scores:
            missing = score.required_columns() - set(df.columns)
            if missing:
                logger.warning(
                    "Skipping %s: missing columns %s — filling outputs with %s",
                    type(score).__name__, sorted(missing), score.missing_fill_value,
                )
                for col in score.output_columns():
                    df[col] = score.missing_fill_value
                continue

            df = score.compute(df)
            all_cleanup_cols |= score.cleanup_columns()

        cols_to_drop = [c for c in all_cleanup_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df