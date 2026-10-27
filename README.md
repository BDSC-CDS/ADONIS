# ADONiS: A Multicenter Benchmark for Machine Learning Prediction of Neonatal Bloodstream Infection

Code for the machine-learning pipeline of the **ADONiS** ("Accelerating Detection Of Neonatal Sepsis") study.

Link to the paper: TBD

The repository contains two parts:
- A **data-processing pipeline** that assembles the modelling dataset from raw hospital records (published for transparency; the raw clinical data are not public, so this pipeline cannot be rerun from this repository).
- A **machine-learning pipeline** that trains, tunes, validates, and plots results for a panel of classifiers predicting neonatal sepsis. This part *is* reproducible from the preprocessed per-hospital parquet files.

---

## Contents

- [Setup](#setup)
- [Folder structure](#folder-structure)
- [Configuration](#configuration)
- [Data processing](#data-processing)
- [Machine learning](#machine-learning)
- [License](#license)
- [Citation](#citation)

---

## Setup

Clone the repository and create the conda environment:

```bash
git clone <repository-url> adonis
cd adonis

conda env create -f environment.yml
conda activate adonis
```

Install the package in editable mode (required — the scripts import `adonis`):

```bash
pip install -e .
```

---

## Repository structure

```
adonis/
├── config/
│   ├── data_processing/config.json         # Data-processing pipeline config
│   └── machine_learning/config.json        # ML pipeline config (edit paths here)
├── data/                                   # Input parquets + list_variables.pkl
├── checkpoints/                            # Optional TabPFN/TabICL weights
├── results/                                # Saved per-model result pkl/parquet
│   └── {sampling_column}/                  # Sampling column used for training (e.g. train_knn_0.1)
├── figures/                                # Rendered publication figures
│   └── {sampling_column}/
├── logs/                                   # Timestamped log files
├── queries/                                # SPARQL queries used by the data-processing pipeline
├── secrets/                                # GraphDB credentials (graphdb.json)
├── scripts/
│   ├── data_processing/                    # Data-processing entry point
│   ├── machine_learning/
│   │   ├── training/                       # run_training_tuning.py
│   │   └── evaluation/                     # external validation, statistical tests
│   └── plotting/                           # plot_*.py and plot_figures.sh
├── src/adonis/                             # Package source
│   ├── data_processing/
│   ├── machine_learning/
│   └── model_evaluation/
├── notebooks/                              # Exploratory notebooks
├── environment.yml
└── pyproject.toml
```

The ML pipeline uses a **custom undersampler**, `NonRandomUnderSampler` ([src/adonis_ml/machine_learning/undersampling.py:8](src/adonis_ml/machine_learning/undersampling.py#L8)), which inherits from `imblearn.under_sampling.RandomUnderSampler` and keeps the rows selected by a boolean column in the dataset. It is attached as the first step of the per-fold sklearn Pipeline in [src/adonis_ml/machine_learning/train_tune.py](src/adonis_ml/machine_learning/train_tune.py) (see `get_estimator_pipeline()`), so each nested-CV fold undersamples its training split before imputation / scaling / modelling.

The column that drives the sampler is configured by `model.undersampler.init_args.sampling_column` in [config/machine_learning/config.json](config/machine_learning/config.json) (e.g. `"train_knn_0.1"`). This string names a **dataset column** that was added to the dataset during **Step 3** of the data-processing pipeline (in the example, `0.1` is the target sepsis prevalence of that sampling scheme). Results and figures land in `results/{sampling_column}/` and `figures/{sampling_column}/` so parallel experiments with different sampling strategies do not overwrite each other.

---

## Configuration

Everything is driven by two JSON configs under `config/`:

- `config/data_processing/config.json` — data-processing pipeline (hospitals, cohort definition, feature engineering, raw-data paths).
- `config/machine_learning/config.json` — ML pipeline (models, hyperparameters, paths, cross-validation, calibration, undersampling).

Only paths need editing. All paths in the shipped configs are relative to the repo root (`./data/`, `./results/`, `./figures/`, `./logs/`); point them elsewhere if you want to store outputs on a different drive.

---

## Data processing

The data-processing pipeline reads the raw per-hospital time-series (one parquet per variable per patient) together with demographics, ventilation and HRC index data stored in a GraphDB SPARQL repository, and emits a per-hospital case–control dataset for modelling.

The data is not public; access is available on request ([eric.giannoni@chuv.ch](mailto:eric.giannoni@chuv.ch)).

**Note on terminology.** The paper's HRC index is referred to as `hero` / `HeRO` throughout the code and configuration (variable names, file names like `hero.parquet`, SPARQL query `queries/hero.txt`, and the `extract_hero` / `--hero` flags).

### Queries

Patient-level metadata (demographics, ventilation, HRC index) is extracted from a GraphDB SPARQL endpoint using the queries shipped under [queries/](queries/):

```
queries/
├── demographics/
│   ├── admission_date.txt      # Admission and discharge dates
│   ├── birth_date.txt          # Date of birth
│   ├── birth_weight.txt        # Birth weight (grams)
│   ├── death.txt               # In-hospital mortality
│   ├── gestational_age.txt     # Gestational age (weeks)
│   ├── length_of_stay.txt      # Length of stay (days)
│   ├── nec.txt                 # Necrotising enterocolitis (NEC) diagnosis
│   ├── patient_info.txt        # Patient and stay identifiers
│   ├── sepsis.txt              # Sepsis diagnosis
│   └── sex.txt                 # Sex
├── file_names.txt              # Raw-signal parquet filenames
├── hero.txt                    # HRC index scores
└── ventilation.txt             # Ventilation episodes
```

**GraphDB prerequisite.** The pipeline connects to a GraphDB SPARQL endpoint that holds the records queried by the files above ([src/adonis_ml/data_processing/connector.py:20](src/adonis_ml/data_processing/connector.py#L20)). Credentials live in `secrets/graphdb.json` (path configurable via `paths.GraphDB_credentials` in [config/data_processing/config.json](config/data_processing/config.json)):

```json
{
    "server_name": "https://<graphdb-host>/repositories/",
    "repository_name": "<repository-name>",
    "format": "JSON"
}
```

### Pipeline

Three ordered steps, orchestrated by `DataProcessingPipeline.run()` in [src/adonis_ml/data_processing/pipeline.py](src/adonis_ml/data_processing/pipeline.py).

```
┌──────────────────────────────────────────────────────────────┐
│ INPUTS                                                       │
│   • data/time_series/{HOSPITAL}/*.parquet   raw signals      │
│   • GraphDB SPARQL endpoint                 demographics,    │
│     via queries/*.txt                       ventilation,     │
│                                             HRC index, …     │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1 — Raw-signal extraction & derived variables           │
│                                                              │
│ For each patient:                                            │
│   • load the raw minute-by-minute vital signs (HR, RR,       │
│     SpO2, …)                                                 │
│   • clean / align the raw signals                            │
│   • compute derived variables (nSOFA, VIS, sf-ratio, …)      │
│                                                              │
│ Writes:                                                      │
│   data/data_processing/cache/{patient_id}_processed.parquet  │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2 — Feature computation                                 │
│                                                              │
│ Aggregates each patient's processed signals hourly and       │
│ computes the modelling features (spikes/drops, trends, std,  │
│ kurtosis, skewness, entropy, …).                             │
│                                                              │
│ Writes:                                                      │
│   data/data_processing/data_total_{HOSPITAL}.parquet         │
│   data/data_processing/list_variables.pkl                    │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3 — Time-window selection + case/control matching       │
│                                                              │
│ Selects per-patient time windows, matches cases and          │
│ controls, and produces the final modelling dataset.          │
│                                                              │
│ Writes:                                                      │
│   data/data_processing/df_cases_controls_{HOSPITAL}.parquet  │
│   data/data_processing/hash_{HOSPITAL}.pkl                   │
└──────────────────────────────────────────────────────────────┘
```

Entry point:

```bash
python scripts/data_processing/run_data_processing.py \
    --config config/data_processing/config.json
```

Useful debug flags (first N patients from a single hospital, verbose logs):

```bash
python scripts/data_processing/run_data_processing.py \
    --config config/data_processing/config.json \
    --n-patients 20 \
    --hospital CHUV \
    --log-level DEBUG
```

The final `df_cases_controls_{HOSPITAL}.parquet` and `list_variables.pkl` produced by Step 3 are the exact files the ML pipeline reads from `data/`.

---

## Machine learning

### Model panel

The pipeline evaluates the following models (configured in [config/machine_learning/config.json](config/machine_learning/config.json)):

`lr`, `knn`, `svm`, `rf`, `lgbm`, `xgb`, `tabpfn`, `tabicl`

### Model checkpoints

`tabpfn` and `tabicl` are tabular foundation models that require pretrained checkpoints hosted on Hugging Face. For reproducibility, the exact versions used in the paper are:

- `Prior-Labs/tabpfn_3/tabpfn-v3-classifier-v3_default.ckpt`
- `jingang/TabICL/tabicl-classifier-v2-20260212.ckpt`

Two options:

- **Pre-downloaded** (default) — place the checkpoints anywhere on disk (e.g. `./checkpoints/`) and point `init_args.model_path` at them. This is the right choice for offline runs or air-gapped clusters. This is the only path wired up in the code — an example configuration is shown below.
- **Automatic download from Hugging Face** (optional) — not currently supported by the codebase, but the loader can be adapted.

Example entries in [config/machine_learning/config.json](config/machine_learning/config.json):

```json
"tabpfn": {
    "class_path": "tabpfn.TabPFNClassifier",
    "init_args": {
        "model_path": "./checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
        "ignore_pretraining_limits": true
    }
},
"tabicl": {
    "class_path": "tabicl.TabICLClassifier",
    "init_args": {
        "model_path": "./checkpoints/tabicl-classifier-v2-20260212.ckpt",
        "allow_auto_download": false,
        "kv_cache": true
    }
}
```

### Data placement

Place the preprocessed per-hospital datasets and the variable list in `./data/`:

```
data/
├── df_cases_controls_CHUV.parquet
├── df_cases_controls_USZ.parquet
├── df_cases_controls_KISPI.parquet
├── hash_CHUV.pkl
├── hash_USZ.pkl
├── hash_KISPI.pkl
├── df_nsofa_CHUV.parquet
├── df_nsofa_USZ.parquet
├── df_nsofa_KISPI.parquet
├── demographics.parquet
└── list_variables.pkl
```

These files are not distributed with this repository (see [Data processing](#data-processing) for how to get access to them).

### Pipeline

The pipeline below is meant to be run in order — later steps consume the parquet/pkl files produced by earlier ones.

1. **Internal cross-validation on CHUV** — nested CV; also saves subgroup metrics and ROC/PRC/DCA/calibration curves. HRC index sub-cohort curves are produced automatically within CV when the HRC index data is available.
2. **External validation on USZ and KISPI** — applies the per-fold CV models to each external hospital.
3. **Forewarning time** — per-patient forewarning time from the CV predictions.
4. **Feature importance** — SHapley Additive exPlanations (SHAP)-based feature-importance.
5. **Ablations** — data and feature-group ablations.
6. **Statistical tests** — paired/unpaired comparisons across models, centers, and subgroups.
7. **Figures** — publication plots assembled from the saved parquet/pkl results.

#### 1. Internal cross-validation (CHUV)

```bash
python scripts/machine_learning/training/run_training_tuning.py \
    --config config/machine_learning/config.json \
    --do-cv \
    --do-subgroups \
    --do-curves \
    --threshold-max 0.1 \
    --threshold-n 10000
```

#### 2. External validation (USZ, KISPI)

```bash
for HOSP in USZ KISPI; do
    python scripts/machine_learning/evaluation/run_external_validation.py \
        --config config/machine_learning/config.json \
        --hospital $HOSP \
        --do-cv \
        --do-subgroups \
        --do-curves \
        --threshold-max 0.1 \
        --threshold-n 10000
done
```

#### 3. Forewarning time

```bash
python scripts/machine_learning/training/run_training_tuning.py \
    --config config/machine_learning/config.json \
    --do-forewarning-time
```

#### 4. Feature importance

```bash
python scripts/machine_learning/training/run_training_tuning.py \
    --config config/machine_learning/config.json \
    --do-shap \
    --shap-n-explain 300 \
    --shap-n-background 100
```

#### 5. Ablations

```bash
# Fractional-data ablation — K = round(1/f) disjoint training subsets
python scripts/machine_learning/training/run_training_tuning.py \
    --config config/machine_learning/config.json \
    --do-fractional \
    --data-fraction 0.5 \
    --threshold-max 0.1 \
    --threshold-n 10000

# Feature-group ablation — train on the listed groups
python scripts/machine_learning/training/run_training_tuning.py \
    --config config/machine_learning/config.json \
    --do-cv \
    --feature-groups demographics basic \
    --threshold-max 0.1 \
    --threshold-n 10000
```

#### 6. Statistical tests

```bash
# Multi-center comparison across all three hospitals
python scripts/machine_learning/evaluation/run_statistical_tests.py \
    --config config/machine_learning/config.json \
    --metrics accuracy balanced_accuracy specificity recall precision f1 roc_auc auprc \
    --models lr knn svm rf lgbm xgb tabpfn tabicl \
    --hospital CHUV USZ KISPI \
    --do-center

# Model comparison on the HRC index sub-cohort (CHUV only)
python scripts/machine_learning/evaluation/run_statistical_tests.py \
    --config config/machine_learning/config.json \
    --metrics accuracy balanced_accuracy specificity recall precision f1 roc_auc auprc \
    --models lr knn svm rf lgbm xgb tabpfn tabicl \
    --hospital CHUV \
    --hero \
    --do-center

# Within-model subgroup comparison on CHUV
python scripts/machine_learning/evaluation/run_statistical_tests.py \
    --config config/machine_learning/config.json \
    --metrics recall \
    --models lr knn svm rf lgbm xgb tabpfn tabicl \
    --hospital CHUV \
    --do-subgroups
```

#### 7. Figures

The plot scripts load the saved per-model result files produced by the previous steps (internal CV, external validation, SHAP, ablations, forewarning time). They expect those files to already exist under `results/{sampling_column}/` (and its `ablation_*` subfolders). Running the figures step before the earlier steps will fail with a missing-file error.

Recreate all paper figures with one command:

```bash
bash scripts/plotting/plot_figures.sh
```

This script calls all `plot_*.py` scripts in sequence against the paths in `config/machine_learning/config.json`. Individual plot scripts can also be invoked on their own — see each script's module docstring for its CLI.

---

## License

See [LICENSE.txt](LICENSE.txt)

## Citation

TBD.
