"""Entry point for the data-processing pipeline.

Usage:
    python scripts/data_processing/run_data_processing.py --config config/data_processing/config.json

    # Debug: first 20 patients from CHUV only
    python scripts/data_processing/run_data_processing.py --config config/data_processing/config.json \
        --n-patients 20 --hospital CHUV --log-level DEBUG

Log file
--------
If ``paths.logs`` is set in the config, a timestamped log file
``data_processing_YYYYMMDD_HHMMSS.log`` is written to that directory in addition
to stdout.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime


def configure_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    fmt = "%(asctime)s [PID:%(process)d] %(name)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"data_processing_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data-processing pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file (e.g. config/data_processing/config.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=None,
        help="Limit extraction to the first N patients (debug mode)",
    )
    parser.add_argument(
        "--hospital",
        default=None,
        help="Restrict extraction to a single hospital (e.g. CHUV, USZ, KISPI)",
    )
    args = parser.parse_args()

    assert args.config.endswith(".json"), (
        f"Expected a .json config file, got: {args.config}"
    )
    with open(args.config) as f:
        config = json.load(f)

    # $USER substitution for all string paths
    if config.get("environment") == "BioMedIT":
        user = os.getlogin()
        for k, v in config.get("paths", {}).items():
            if isinstance(v, str) and "$USER" in v:
                config["paths"][k] = v.replace("$USER", user)

    log_dir = config.get("paths", {}).get("logs")
    configure_logging(level=args.log_level, log_dir=log_dir)
    logger = logging.getLogger(__name__)

    from adonis_ml.utils import AdonisConfig
    from adonis_ml.data_processing.pipeline import DataProcessingPipeline

    logger.info("Starting data-processing pipeline with config: %s", args.config)
    pipeline = DataProcessingPipeline(
        config=AdonisConfig(config),
        debug_n_patients=args.n_patients,
        debug_hospital=args.hospital,
    )
    pipeline.run()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
