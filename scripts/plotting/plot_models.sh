#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/machine_learning/config.json"
HOSPITAL="CHUV"

python scripts/plotting/plot_models.py \
    --config "${CONFIG}" \
    --hospital "${HOSPITAL}" 
