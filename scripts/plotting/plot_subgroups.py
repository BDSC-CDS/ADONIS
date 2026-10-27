"""Re-plot subgroup analysis figures from saved parquet files.

Produces two figures per subgroup group:
  - subgroups_{group}_by_category.png  (models on x-axis, categories as bars)
  - subgroups_{group}_by_model.png     (models on x-axis, categories as bar colors)

Usage
-----
    python scripts/plotting/plot_subgroups.py \\
        --config config/machine_learning/config.json \\
        --group age_group

CLI flags
---------
--config      Path to JSON config file (required).
--group       Subgroup name(s) from config.subgroups (default: all).
--metrics     Metric names to plot (default: all in parquet).
--models      Model types to include (default: all from config).
--ncols       Number of subplot columns (default: 2).
--log-level   Logging verbosity (default: INFO).
"""
from __future__ import annotations

import os

os.environ["TABPFN_NO_ANALYTICS"] = "true"

import argparse
import logging
import sys

import pandas as pd

from adonis_ml.machine_learning.plotting import (
    plot_subgroups_by_category,
    plot_subgroups_by_model,
)
from adonis_ml.machine_learning.utils import get_results_path
from adonis_ml.utils import AdonisConfig, load_and_parse_config

logger = logging.getLogger(__name__)


def load_subgroups_from_parquet(config, model_types: list[str], group_names: list[str]) -> dict:
    results_path = get_results_path(config)
    results_subgroups: dict = {}
    for model_type in model_types:
        model_path = os.path.join(results_path, model_type)
        results_subgroups[model_type] = {}
        for group_name in group_names:
            pq_path = os.path.join(model_path, f"subgroups_{group_name}.parquet")
            if os.path.exists(pq_path):
                results_subgroups[model_type][group_name] = pd.read_parquet(pq_path)
                logger.info("Loaded %s / %s (%d rows)", model_type, group_name,
                            len(results_subgroups[model_type][group_name]))
            else:
                logger.warning("Not found: %s", pq_path)
    return results_subgroups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-plot subgroup figures from saved parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", nargs="+", default=None,
                        help="Subgroup name(s) from config.subgroups (default: all).")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metric names to plot (default: all).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types to include (default: all from config).")
    parser.add_argument("--ncols", type=int, default=2,
                        help="Number of subplot columns (default: 2).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)

    if config.environment == "BioMedIT":
        user = os.getlogin()
        for k, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[k] = v.replace("$USER", user)

    model_types = args.models or list(config.model["models"].keys())
    group_names = args.group or list(config.subgroups.keys())

    results_subgroups = load_subgroups_from_parquet(config, model_types, group_names)

    for group_name in group_names:
        plot_subgroups_by_category(
            config, results_subgroups, group_name,
            metrics=args.metrics, model_types=model_types, ncols=args.ncols,
        )
        plot_subgroups_by_model(
            config, results_subgroups, group_name,
            metrics=args.metrics, model_types=model_types, ncols=args.ncols,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
