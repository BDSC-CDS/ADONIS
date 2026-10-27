#!/usr/bin/env bash
# Generate the full set of ADONIS-ML publication figures and statistics.
#
# Prerequisites: training (and external validation) must have been run so that the
# saved per-model result parquets exist in the paths configured in
# config/machine_learning/config.json.
#
# Usage (from the repository root):
#   bash scripts/plotting/plot_figures.sh

python scripts/plotting/plot_models.py --config config/machine_learning/config.json --hospital CHUV --metrics accuracy balanced_accuracy specificity recall precision f1 roc_auc auprc --models lr knn svm rf lgbm xgb tabpfn tabicl --ncols 2 --add-stat

python scripts/plotting/plot_curves.py --config config/machine_learning/config.json --hospital CHUV --curves roc prc dca cal --models lr knn svm rf lgbm xgb tabpfn tabicl --ncols 2

python scripts/plotting/plot_curves.py --config config/machine_learning/config.json --hospital CHUV --hero --curves roc prc dca cal --models lr knn svm rf lgbm xgb tabpfn tabicl --ncols 2

python scripts/plotting/plot_subgroups.py --config config/machine_learning/config.json --metrics recall --models lr knn svm rf lgbm xgb tabpfn tabicl

python scripts/plotting/plot_ablation_fractional.py --config config/machine_learning/config.json --fractions 0.1 0.5 1.0 --metrics accuracy balanced_accuracy specificity recall precision f1 roc_auc auprc --models lr knn svm rf lgbm xgb tabpfn --ncols 3

python scripts/plotting/plot_ablation_features.py --config config/machine_learning/config.json --metrics recall precision roc_auc auprc --models lr knn svm rf lgbm xgb tabpfn --groups demographics_basic --group-labels "Lite"

python scripts/plotting/plot_forewarning_time.py --config config/machine_learning/config.json --forewarning-time 72 --hero
