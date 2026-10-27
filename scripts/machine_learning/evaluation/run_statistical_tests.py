"""Statistical tests for ADONIS-ML model comparisons.

Three analyses:
  1. Center-level — best vs others: best ML model vs all other ML models per metric.
  2. Center-level — hero vs others: HeRO clinical score vs all ML models (AUROC only
     in --hero mode; all metrics in standard mode when hero_predictor is present).
  3. Subgroup — within-model: for each model and metric, compare two subgroup
     categories using an unpaired label-shuffling permutation test on window-level
     OOF predictions (patients are not paired across categories).

Statistical methods
-------------------
  AUROC        — DeLong's test (MLstatkit.Delong_test) on OOF window-level predictions.
  all others   — Paired permutation test via _get_metric_fn (sklearn get_scorer /
                 get_custom_scorer). Supports: accuracy, balanced_accuracy, precision,
                 recall, auprc, specificity, f1 — or any metric passed via --metrics.
  Subgroups    — Unpaired label-shuffling permutation test on window-level OOF
                 predictions (bilateral). Pools rows from both categories, randomly
                 reassigns preserving group sizes. CHUV only.

Pairing / OOF construction
---------------------------
  CHUV        — True OOF: concatenate predictions across 5 folds (sorted by fold index).
  USZ / KISPI — Ensemble OOF: group flat entries by shared test_idx (same external split),
                average y_prob across the 5 CHUV fold models within each split,
                concatenate the 2 disjoint splits.

Hero mode (--hero)
------------------
  Loads OOF predictions from *_sklearn.pkl filtered to the HeRO sub-cohort.
  ML models: y_pred comes from the saved tuned-threshold predictions (same as
    the standard OOF), masked to hero windows.
  hero_predictor: HeroPredictor is refitted per fold on the hero windows to
    obtain both y_prob and binary y_pred (mirrors run_evaluation_hero).
  All metrics are compared via the same tests as center-level.
  Runs only hero_vs_others; best_vs_others is skipped.

Multiple-testing correction
---------------------------
  BH (FDR) applied within each comparison family:
    center tests: separately per comparison_type (best_vs_others / hero_vs_others)
    subgroups: per (hospital, group)

Outputs (parquet)
-----------------
  {results_path}/statistical_tests_center_{hospital}.parquet
  {results_path}/statistical_tests_center_{hospital}_hero.parquet   (--hero)
  {results_path}/statistical_tests_subgroups_{group}_{hospital}.parquet

  Columns (center):
    hospital | comparison_type | metric | model_A | model_B
    | n_samples | statistic | p_value | p_adj_bh | reject_h0

  Columns (subgroup):
    hospital | group | metric | model | group_A | group_B
    | n_A | n_B | statistic | p_value | p_adj_bh | reject_h0

Usage
-----
    python scripts/machine_learning/evaluation/run_statistical_tests.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV --do-center --do-subgroups

    python scripts/machine_learning/evaluation/run_statistical_tests.py \\
        --config config/machine_learning/config.json \\
        --hospital CHUV --do-center --hero

    python scripts/machine_learning/evaluation/run_statistical_tests.py \\
        --config config/machine_learning/config.json \\
        --hospital USZ KISPI --do-center
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
from itertools import combinations

from tqdm import tqdm

os.environ["TABPFN_NO_ANALYTICS"] = "true"

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from statsmodels.stats.multitest import multipletests
from MLstatkit.stats import Delong_test

from adonis_ml.machine_learning.scoring import get_custom_scorer
from adonis_ml.machine_learning.utils import get_results_path
from adonis_ml.utils import AdonisConfig, load_and_parse_config

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "auprc", "specificity", "f1"]
MIN_SAMPLES = 10

# Metrics that use y_prob (not y_pred) as the prediction input
_PROB_METRICS = {"roc_auc", "auprc"}

# sklearn scorer name differs from our metric name only for auprc
_SKLEARN_SCORER_NAME = {"auprc": "average_precision"}

# These sklearn scorers don't pass zero_division by default — inject it
_ZERO_DIV_METRICS = {"precision", "recall", "f1"}


# ---------------------------------------------------------------------------
# Logging / CLI
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("posthog").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical tests for ADONIS-ML model comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--hospital", nargs="+", default=["CHUV"],
                        help="Hospital(s): CHUV / USZ / KISPI.")
    parser.add_argument("--do-center", action="store_true",
                        help="Run center-level comparisons.")
    parser.add_argument("--do-subgroups", action="store_true",
                        help="Run within-model subgroup comparisons.")
    parser.add_argument("--hero", action="store_true",
                        help="Compare on the HeRO sub-cohort (CHUV only, AUROC via DeLong).")
    parser.add_argument("--group", nargs="+", default=None,
                        help="Subgroup name(s) from config (default: all).")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help=f"Metric base names (default: {' '.join(DEFAULT_METRICS)}).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types (default: all from config).")
    parser.add_argument("--n-permutations", type=int, default=10_000,
                        help="Number of permutations for permutation tests.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for reject_h0.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# BH correction
# ---------------------------------------------------------------------------

def _apply_bh(rows: list[dict], alpha: float) -> list[dict]:
    """Add p_adj_bh and reject_h0 via BH correction over all rows in this family."""
    p_values = np.array([r["p_value"] for r in rows], dtype=float)
    valid = ~np.isnan(p_values)
    p_adj = np.full(len(rows), np.nan)
    reject = np.zeros(len(rows), dtype=bool)

    if valid.sum() >= 1:
        rej, p_corr, _, _ = multipletests(p_values[valid], alpha=alpha, method="fdr_bh")
        p_adj[valid] = p_corr
        reject[valid] = rej

    for row, pa, rj in zip(rows, p_adj, reject):
        row["p_adj_bh"] = None if np.isnan(pa) else float(pa)
        row["reject_h0"] = bool(rj)
    return rows


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------

def _get_metric_fn(metric: str):
    """Return a (y_true, pred) -> float callable for the given metric name.

    Resolves via get_custom_scorer for 'specificity' (sklearn has no native scorer)
    and via get_scorer for all standard sklearn metrics.
    """
    if metric == "specificity":
        s = get_custom_scorer("specificity")
    else:
        s = get_scorer(_SKLEARN_SCORER_NAME.get(metric, metric))
    fn = s._score_func
    kwargs = {**s._kwargs, **({"zero_division": 0} if metric in _ZERO_DIV_METRICS else {})}
    return lambda y_true, pred: float(fn(y_true, pred, **kwargs))


def _custom_permutation_test(
    y_true: np.ndarray,
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    metric_fn,
    n_perms: int,
    rng: np.random.Generator,
    desc: str = "permutation",
    two_sided: bool = True,
) -> tuple[float, float]:
    """Paired permutation test: randomly swap pred_A[i] / pred_B[i] per sample.

    two_sided=True  — count |perm_diff| >= |obs_diff|  (any direction).
    two_sided=False — count perm_diff >= obs_diff  (one-sided: A better than B).
    p-value floored at 1/(n_perms+1) via the +1 correction (Phipson & Smyth 2010).
    """
    obs_diff = metric_fn(y_true, pred_A) - metric_fn(y_true, pred_B)
    n = len(pred_A)
    count = 0
    for _ in tqdm(range(n_perms), desc=desc, leave=False, unit="perm"):
        swap = rng.random(n) > 0.5
        pA = np.where(swap, pred_B, pred_A)
        pB = np.where(swap, pred_A, pred_B)
        perm_diff = metric_fn(y_true, pA) - metric_fn(y_true, pB)
        if two_sided:
            if abs(perm_diff) >= abs(obs_diff):
                count += 1
        else:
            if perm_diff >= obs_diff:
                count += 1
    return float(obs_diff), (count + 1) / (n_perms + 1)


def _unpaired_permutation_test(
    y_true_A: np.ndarray,
    pred_A: np.ndarray,
    y_true_B: np.ndarray,
    pred_B: np.ndarray,
    metric_fn,
    n_perms: int,
    rng: np.random.Generator,
    desc: str = "permutation",
) -> tuple[float, float]:
    """Unpaired label-shuffling permutation test (bilateral).

    Pools all rows from both categories then randomly reassigns them to
    group A / B (preserving group sizes) for each permutation.
    Patients are not paired — the exchangeable unit is the window row.
    p-value floored at 1/(n_perms+1) via the +1 correction.
    """
    y_true_all = np.concatenate([y_true_A, y_true_B])
    pred_all = np.concatenate([pred_A, pred_B])
    n_A = len(y_true_A)
    n_total = len(y_true_all)
    obs_diff = metric_fn(y_true_A, pred_A) - metric_fn(y_true_B, pred_B)
    idx = np.arange(n_total)
    count = 0
    for _ in tqdm(range(n_perms), desc=desc, leave=False, unit="perm"):
        rng.shuffle(idx)
        perm_diff = (
            metric_fn(y_true_all[idx[:n_A]], pred_all[idx[:n_A]])
            - metric_fn(y_true_all[idx[n_A:]], pred_all[idx[n_A:]])
        )
        if abs(perm_diff) >= abs(obs_diff):
            count += 1
    return float(obs_diff), (count + 1) / (n_perms + 1)


def _run_test(
    metric: str,
    y_true: np.ndarray,
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
    desc: str = "",
    two_sided: bool = True,
) -> tuple[float, float]:
    """Dispatch to the appropriate test. Returns (statistic, p_value).

    For roc_auc, DeLong's test is used (two-sided p halved for one-sided).
    For all other metrics, a paired permutation test is used.
    """
    if len(y_true) < MIN_SAMPLES:
        logger.debug("Too few samples (%d) for %s — skipping", len(y_true), metric)
        return np.nan, np.nan

    try:
        if metric == "roc_auc":
            z, p = Delong_test(y_true, pred_A, pred_B, return_ci=False, return_auc=False)
            # DeLong returns a two-sided p; halve it for one-sided (pred_A is always best, z >= 0)
            return float(z), float(p) / (1 if two_sided else 2)

        perm_desc = f"perm [{metric}] {desc}" if desc else f"perm [{metric}]"
        return _custom_permutation_test(
            y_true, pred_A, pred_B, _get_metric_fn(metric), n_perms, rng,
            desc=perm_desc, two_sided=two_sided,
        )

    except Exception as exc:
        logger.warning("Test failed for metric '%s': %s", metric, exc)

    return np.nan, np.nan


# ---------------------------------------------------------------------------
# OOF prediction loading
# ---------------------------------------------------------------------------

def _test_idx_key(test_idx: np.ndarray) -> bytes:
    """Stable hash of a test_idx array — used to group external CV entries by split."""
    return hashlib.md5(np.asarray(test_idx, dtype=np.int64).tobytes()).digest()


def _extract_fold_predictions(fold_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_prob, y_pred) for a single fold entry.

    Handles two storage conventions:
      - y_true is the full dataset array (CHUV) → index with test_idx
      - y_true is the test-fold subset (external) → use directly
    """
    test_idx = fold_data["test_idx"]
    y_true_raw = np.asarray(fold_data["y_true"])
    y_prob = np.asarray(fold_data["y_prob"])
    y_pred = np.asarray(fold_data["y_pred"])

    if len(y_true_raw) != len(y_prob):
        # y_true is the full dataset array → subset to test fold
        y_true = y_true_raw[test_idx]
    else:
        y_true = y_true_raw

    return y_true, y_prob, y_pred


def _extract_metric_means(saved: dict, split: str = "test") -> dict[str, float]:
    """Pull per-metric mean-across-folds from the pkl's `results` block.

    Mirrors how plot_models reads the pkl so the 'best' model picked here
    matches what plot_models reports.
    """
    results = saved.get("results", {})
    if split in results and isinstance(results[split], dict):
        results = results[split]
    prefix = f"{split}_"
    means: dict[str, float] = {}
    for k, v in results.items():
        name = k[len(prefix):] if k.startswith(prefix) else k
        if isinstance(v, dict) and "mean" in v:
            means[name] = float(v["mean"])
    return means


def load_oof_predictions(
    results_path: str, model_type: str, hospital: str
) -> dict | None:
    """Load OOF predictions for one model / hospital.

    Returns {"y_true": ..., "y_prob": ..., "y_pred": ..., "metric_means": {...}}
    or None if unavailable.
    """
    if hospital == "CHUV":
        pkl_path = os.path.join(results_path, f"{model_type}_sklearn.pkl")
    else:
        pkl_path = os.path.join(
            results_path, f"{model_type}_sklearn_external_{hospital}_cv.pkl"
        )

    if not os.path.isfile(pkl_path):
        logger.warning("pkl not found — skipping %s / %s: %s", hospital, model_type, pkl_path)
        return None

    with open(pkl_path, "rb") as fh:
        saved = pickle.load(fh)

    metric_means = _extract_metric_means(saved)
    raw = saved.get("results_predict", {})
    predict_data: dict = raw.get("test", raw)

    if hospital == "CHUV":
        # True OOF: concatenate in sorted fold order
        y_true_list, y_prob_list, y_pred_list = [], [], []
        for key in sorted(predict_data.keys()):
            yt, yp, yd = _extract_fold_predictions(predict_data[key])
            y_true_list.append(yt)
            y_prob_list.append(yp)
            y_pred_list.append(yd)
        return {
            "y_true": np.concatenate(y_true_list),
            "y_prob": np.concatenate(y_prob_list),
            "y_pred": np.concatenate(y_pred_list),
            "metric_means": metric_means,
        }

    # External hospital: group entries by shared test_idx → ensemble across CHUV folds
    groups: dict[bytes, dict] = {}
    for key in sorted(predict_data.keys()):
        yt, yp, yd = _extract_fold_predictions(predict_data[key])
        idx_hash = _test_idx_key(predict_data[key]["test_idx"])
        if idx_hash not in groups:
            groups[idx_hash] = {"y_true": yt, "y_probs": [], "y_preds": []}
        groups[idx_hash]["y_probs"].append(yp)
        groups[idx_hash]["y_preds"].append(yd)

    y_true_list, y_prob_list, y_pred_list = [], [], []
    for g in groups.values():
        ens_prob = np.mean(g["y_probs"], axis=0)
        # Majority vote across fold-level tuned-threshold predictions
        ens_pred = (np.mean(g["y_preds"], axis=0) >= 0.5).astype(int)
        y_true_list.append(g["y_true"])
        y_prob_list.append(ens_prob)
        y_pred_list.append(ens_pred)

    return {
        "y_true": np.concatenate(y_true_list),
        "y_prob": np.concatenate(y_prob_list),
        "y_pred": np.concatenate(y_pred_list),
        "metric_means": metric_means,
    }


def load_hero_oof_predictions(
    results_path: str, model_types: list[str], config
) -> dict[str, dict]:
    """Load hero sub-cohort OOF predictions (CHUV only).

    For ML models: filters the standard OOF results_predict to hero-masked rows,
    preserving y_true, y_prob, and y_pred (from the tuned threshold).
    For hero_predictor: refits HeroPredictor per fold (mirrors run_evaluation_hero)
    to obtain both y_prob and binary y_pred.

    Requires loading the feature dataset (DataLoader) to align hero_max with test_idx.
    """
    from adonis_ml.machine_learning.data_loader import DataLoader
    from adonis_ml.machine_learning.dataloading import get_df_hero
    from adonis_ml.machine_learning.hero import HeroPredictor

    logger.info("Loading feature dataset and hero_max for hero OOF filtering...")
    data_loader = DataLoader(config)
    df, _ = data_loader.load_adonis_feature_dataset()

    df_hero = get_df_hero(config)
    df_merged = df[["stay_id", "time_difference_hours"]].reset_index(drop=True).merge(
        df_hero[["stay_id", "time_difference_hours", "hero_max"]],
        on=["stay_id", "time_difference_hours"],
        how="left",
    )
    hero_max_full = df_merged["hero_max"].values.astype(float)

    all_oof: dict[str, dict] = {}
    hero_y_true_oof: list[np.ndarray] = []
    hero_prob_oof: list[np.ndarray] = []
    hero_pred_oof: list[np.ndarray] = []
    hero_max_oof_collected: list[np.ndarray] = []

    for mt in model_types:
        pkl_path = os.path.join(results_path, f"{mt}_sklearn.pkl")
        if not os.path.isfile(pkl_path):
            logger.warning("pkl not found — skipping %s: %s", mt, pkl_path)
            continue
        with open(pkl_path, "rb") as fh:
            saved = pickle.load(fh)

        predict_data: dict = saved.get("results_predict", {}).get("test", {})

        y_true_list, y_prob_list, y_pred_list = [], [], []
        hero_max_fold_list: list[np.ndarray] = []

        for key in sorted(predict_data.keys()):
            fold_data = predict_data[key]
            test_idx = fold_data["test_idx"]
            yt, yp, yd = _extract_fold_predictions(fold_data)
            hero_vals = hero_max_full[test_idx]
            hero_mask = ~np.isnan(hero_vals)

            if hero_mask.sum() < MIN_SAMPLES:
                logger.debug("Fold %s [%s]: too few hero windows (%d) — skipping", key, mt, hero_mask.sum())
                continue

            y_true_list.append(yt[hero_mask])
            y_prob_list.append(yp[hero_mask])
            y_pred_list.append(yd[hero_mask])
            hero_max_fold_list.append(hero_vals[hero_mask])

        if not y_true_list:
            logger.warning("No hero data found for model '%s' — skipping", mt)
            continue

        all_oof[mt] = {
            "y_true": np.concatenate(y_true_list),
            "y_prob": np.concatenate(y_prob_list),
            "y_pred": np.concatenate(y_pred_list),
        }

        if not hero_max_oof_collected:
            # Build hero_predictor OOF from first model's folds (same hero windows)
            for yt_h, hmax_h in zip(y_true_list, hero_max_fold_list):
                X_hero = pd.DataFrame({"hero_max": hmax_h})
                hp = HeroPredictor(use_max=True, fixed_prevalence=0.005)
                hp.fit(X_hero, yt_h)
                hero_prob_oof.append(hp.predict_proba(X_hero)[:, 1])
                hero_pred_oof.append(hp.predict(X_hero).ravel())
                hero_y_true_oof.append(yt_h)
                hero_max_oof_collected.append(hmax_h)

    if hero_y_true_oof:
        all_oof["hero_predictor"] = {
            "y_true": np.concatenate(hero_y_true_oof),
            "y_prob": np.concatenate(hero_prob_oof),
            "y_pred": np.concatenate(hero_pred_oof),
        }
        logger.info("Hero predictor OOF: %d windows", len(all_oof["hero_predictor"]["y_true"]))

    return all_oof


# ---------------------------------------------------------------------------
# Center-level tests (analyses 1 & 2)
# ---------------------------------------------------------------------------

def _oof_metric_value(metric: str, oof: dict) -> float:
    """Compute a single metric value from OOF predictions (for finding the best model)."""
    use_prob = metric in _PROB_METRICS
    pred = oof["y_prob"] if use_prob else oof.get("y_pred")
    if pred is None:
        return np.nan
    try:
        return _get_metric_fn(metric)(oof["y_true"], pred)
    except Exception:
        return np.nan


def run_center_tests(
    all_oof: dict[str, dict],
    hospital: str,
    metrics: list[str],
    alpha: float,
    n_perms: int,
    hero_mode: bool = False,
) -> pd.DataFrame:
    """Best-vs-others (ML only) and hero-vs-others comparisons.

    hero_mode=True: only hero_vs_others is run (best_vs_others is skipped).
    BH is applied separately within each comparison_type family.
    """
    HERO_KEY = "hero_predictor"
    rng = np.random.default_rng(42)

    available = list(all_oof.keys())
    ml_models = [m for m in available if m != HERO_KEY]
    active_metrics = metrics

    rows_best: list[dict] = []
    rows_hero: list[dict] = []

    for metric in active_metrics:
        use_prob = metric in _PROB_METRICS

        # --- Analysis 1: best ML model vs other ML models (skipped in hero mode) ---
        if not hero_mode and len(ml_models) >= 2:
            # 'Best' = highest mean-across-folds reported in the pkl (same number
            # plot_models shows). Falls back to a pooled-OOF scalar only when
            # the pkl didn't record a mean for this metric.
            means = {}
            for m in ml_models:
                mm = all_oof[m].get("metric_means", {}).get(metric)
                means[m] = float(mm) if mm is not None else _oof_metric_value(metric, all_oof[m])
            valid_ml = {m: v for m, v in means.items() if not np.isnan(v)}
            if len(valid_ml) >= 2:
                best = max(valid_ml, key=valid_ml.__getitem__)
                best_pred = all_oof[best]["y_prob" if use_prob else "y_pred"]
                y_true_best = all_oof[best]["y_true"]

                for mt in valid_ml:
                    if mt == best:
                        continue
                    other_pred = all_oof[mt]["y_prob" if use_prob else "y_pred"]
                    if len(best_pred) != len(other_pred):
                        logger.warning(
                            "OOF length mismatch (%s vs %s) for %s/%s — skipping",
                            len(best_pred), len(other_pred), hospital, metric,
                        )
                        continue
                    stat, p = _run_test(metric, y_true_best, best_pred, other_pred, n_perms, rng, desc=f"{best} vs {mt}", two_sided=False)
                    rows_best.append({
                        "hospital": hospital,
                        "comparison_type": "best_vs_others",
                        "metric": metric,
                        "model_A": best,
                        "model_B": mt,
                        "n_samples": len(y_true_best),
                        "statistic": stat,
                        "p_value": p,
                    })

        # --- Analysis 2: hero vs all ML models ---
        if HERO_KEY in available:
            hero_oof = all_oof[HERO_KEY]
            hero_pred = hero_oof["y_prob" if use_prob else "y_pred"]
            y_true_hero = hero_oof["y_true"]

            for mt in ml_models:
                if mt not in available:
                    continue
                other_oof = all_oof[mt]
                other_pred = other_oof["y_prob" if use_prob else "y_pred"]
                if len(hero_pred) != len(other_pred):
                    logger.warning(
                        "OOF length mismatch (hero vs %s) for %s/%s — skipping",
                        mt, hospital, metric,
                    )
                    continue
                stat, p = _run_test(metric, y_true_hero, hero_pred, other_pred, n_perms, rng, desc=f"hero vs {mt}", two_sided=True)
                rows_hero.append({
                    "hospital": hospital,
                    "comparison_type": "hero_vs_others",
                    "metric": metric,
                    "model_A": HERO_KEY,
                    "model_B": mt,
                    "n_samples": len(y_true_hero),
                    "statistic": stat,
                    "p_value": p,
                })

    if rows_best:
        rows_best = _apply_bh(rows_best, alpha)
    if rows_hero:
        rows_hero = _apply_bh(rows_hero, alpha)

    all_rows = rows_best + rows_hero
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    col_order = [
        "hospital", "comparison_type", "metric", "model_A", "model_B",
        "n_samples", "statistic", "p_value", "p_adj_bh", "reject_h0",
    ]
    return df[[c for c in col_order if c in df.columns]]


# ---------------------------------------------------------------------------
# Subgroup OOF loading
# ---------------------------------------------------------------------------

def load_subgroup_oof_predictions(
    results_path: str,
    model_type: str,
    group_name: str,
    config,
) -> dict[str, dict] | None:
    """Load window-level OOF predictions split by subgroup category (CHUV only).

    Replicates the controls_in_all masking logic from run_evaluation_subgroups.
    Returns {category: {"y_true": ..., "y_prob": ..., "y_pred": ...}} or None
    if fewer than 2 categories have enough data.
    """
    from adonis_ml.machine_learning.data_loader import DataLoader
    from adonis_ml.machine_learning.dataloading import get_subgroups_from_df

    pkl_path = os.path.join(results_path, f"{model_type}_sklearn.pkl")
    if not os.path.isfile(pkl_path):
        logger.warning("pkl not found — skipping %s: %s", model_type, pkl_path)
        return None

    with open(pkl_path, "rb") as fh:
        saved = pickle.load(fh)
    predict_data: dict = saved.get("results_predict", {}).get("test", {})

    data_loader = DataLoader(config)
    df, _ = data_loader.load_adonis_feature_dataset()
    demog = data_loader.load_demog()
    subgroups = get_subgroups_from_df(config, df, demog)
    patient_ids = df["stay_id"].values

    # Identify sepsis patients: any patient with at least one y_true==1 window
    y_all = np.full(len(patient_ids), np.nan)
    for key in sorted(predict_data.keys()):
        yt, _, _ = _extract_fold_predictions(predict_data[key])
        idx = predict_data[key]["test_idx"]
        y_all[idx] = np.where(np.isnan(y_all[idx]), yt, y_all[idx])
    sepsis_pids = np.unique(patient_ids[~np.isnan(y_all) & (y_all == 1)])
    is_sepsis_pid = np.isin(patient_ids, sepsis_pids)

    controls_in_all = config.subgroups[group_name].get("controls_in_all_groups", False)
    categories = subgroups[group_name].cat.categories.tolist()
    cat_data: dict[str, dict] = {cat: {"y_true": [], "y_prob": [], "y_pred": []} for cat in categories}

    for key in sorted(predict_data.keys()):
        fold_data = predict_data[key]
        test_idx = fold_data["test_idx"]
        yt, yp, yd = _extract_fold_predictions(fold_data)
        subgroups_test = subgroups.iloc[test_idx]
        is_sepsis_test = is_sepsis_pid[test_idx]

        for cat in categories:
            mask = np.asarray(subgroups_test[group_name] == cat)
            if controls_in_all:
                # Sepsis patients in this category + all pure controls
                mask = (mask & is_sepsis_test) | ~is_sepsis_test
            if mask.sum() < MIN_SAMPLES:
                continue
            cat_data[cat]["y_true"].append(yt[mask])
            cat_data[cat]["y_prob"].append(yp[mask])
            cat_data[cat]["y_pred"].append(yd[mask])

    result = {}
    for cat, data in cat_data.items():
        if not data["y_true"]:
            logger.warning(
                "No data for category '%s' / group '%s' / model '%s'", cat, group_name, model_type
            )
            continue
        result[cat] = {
            "y_true": np.concatenate(data["y_true"]),
            "y_prob": np.concatenate(data["y_prob"]),
            "y_pred": np.concatenate(data["y_pred"]),
        }

    if len(result) < 2:
        logger.warning(
            "Fewer than 2 categories with data for group '%s' / model '%s' — skipping",
            group_name, model_type,
        )
        return None
    return result


# ---------------------------------------------------------------------------
# Subgroup tests (analysis 3)
# ---------------------------------------------------------------------------

def run_subgroup_tests(
    results_path: str,
    model_types: list[str],
    hospital: str,
    group_name: str,
    metrics: list[str],
    alpha: float,
    config,
    n_perms: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Within-model unpaired permutation test comparing subgroup categories.

    Loads window-level OOF predictions per category from the pkl (CHUV only).
    Patients are not paired across categories — uses an unpaired label-shuffling
    permutation test (bilateral). BH is applied over all
    (metric × model × category_pair) tests in this (hospital, group) family.
    """
    if hospital != "CHUV":
        logger.warning("Subgroup tests use OOF pkl — CHUV only, skipping %s", hospital)
        return pd.DataFrame()

    rows: list[dict] = []

    for mt in model_types:
        logger.info("Loading subgroup OOF predictions: %s / %s", mt, group_name)
        cat_oof = load_subgroup_oof_predictions(results_path, mt, group_name, config)
        if cat_oof is None:
            continue

        categories = sorted(cat_oof.keys())

        for metric in metrics:
            use_prob = metric in _PROB_METRICS
            for cat_a, cat_b in combinations(categories, 2):
                y_true_a = cat_oof[cat_a]["y_true"]
                y_true_b = cat_oof[cat_b]["y_true"]
                pred_a = cat_oof[cat_a]["y_prob" if use_prob else "y_pred"]
                pred_b = cat_oof[cat_b]["y_prob" if use_prob else "y_pred"]

                if len(y_true_a) < MIN_SAMPLES or len(y_true_b) < MIN_SAMPLES:
                    logger.debug(
                        "Too few samples (%d / %d) for %s/%s/%s vs %s — skipping",
                        len(y_true_a), len(y_true_b), mt, metric, cat_a, cat_b,
                    )
                    continue

                try:
                    perm_desc = f"perm [{metric}] {mt}: {cat_a} vs {cat_b}"
                    fn = _get_metric_fn(metric)
                    stat, p = _unpaired_permutation_test(
                        y_true_a, pred_a, y_true_b, pred_b, fn, n_perms, rng,
                        desc=perm_desc,
                    )
                except Exception as exc:
                    logger.warning("Test failed for %s/%s/%s vs %s: %s", mt, metric, cat_a, cat_b, exc)
                    stat, p = np.nan, np.nan

                rows.append({
                    "hospital": hospital,
                    "group": group_name,
                    "metric": metric,
                    "model": mt,
                    "group_A": cat_a,
                    "group_B": cat_b,
                    "n_A": len(y_true_a),
                    "n_B": len(y_true_b),
                    "statistic": stat,
                    "p_value": p,
                })

    if not rows:
        return pd.DataFrame()

    rows = _apply_bh(rows, alpha)
    df_out = pd.DataFrame(rows)
    col_order = [
        "hospital", "group", "metric", "model", "group_A", "group_B",
        "n_A", "n_B", "statistic", "p_value", "p_adj_bh", "reject_h0",
    ]
    return df_out[[c for c in col_order if c in df_out.columns]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_dict = load_and_parse_config(args.config)
    config = AdonisConfig(config_dict)

    if config.environment == "BioMedIT":
        user = os.getlogin()
        for key, v in config.paths.items():
            if isinstance(v, str) and "$USER" in v:
                config.paths[key] = v.replace("$USER", user)
        config.mlflow["path_to_checkpoint"] = config.mlflow["path_to_checkpoint"].replace(
            "$USER", user
        )

    results_path = get_results_path(config)
    model_types = args.models or list(config.model["models"].keys())
    metrics = args.metrics or DEFAULT_METRICS
    group_names = args.group or list(config.subgroups.keys())

    active = [k for k, v in vars(args).items() if k.startswith("do_") and v]
    logger.info("Hospitals:       %s", args.hospital)
    logger.info("Models:          %s", model_types)
    logger.info("Metrics:         %s", metrics)
    logger.info("Groups:          %s", group_names)
    logger.info("Alpha:           %.3f", args.alpha)
    logger.info("N permutations:  %d", args.n_permutations)
    logger.info("Hero mode:       %s", args.hero)
    logger.info("Active flags:    %s", active)
    logger.info("Results path:    %s", results_path)

    for hospital in args.hospital:

        if args.do_center:
            logger.info("--- Center tests: %s%s ---", hospital, " (hero)" if args.hero else "")

            if args.hero:
                if hospital != "CHUV":
                    logger.warning("--hero is CHUV-only — skipping %s", hospital)
                    continue
                try:
                    all_oof = load_hero_oof_predictions(results_path, model_types, config)
                except Exception as exc:
                    logger.error("Failed to load hero OOF predictions: %s", exc)
                    continue
                out_suffix = "_hero"
            else:
                all_oof = {}
                for mt in model_types:
                    oof = load_oof_predictions(results_path, mt, hospital)
                    if oof is not None:
                        all_oof[mt] = oof
                out_suffix = ""

            if not all_oof:
                logger.warning("No OOF predictions loaded for %s — skipping", hospital)
            else:
                df_center = run_center_tests(
                    all_oof, hospital, metrics, args.alpha,
                    args.n_permutations, hero_mode=args.hero,
                )
                if not df_center.empty:
                    out_path = os.path.join(
                        results_path,
                        f"statistical_tests_center_{hospital}{out_suffix}.parquet",
                    )
                    df_center.to_parquet(out_path, index=False)
                    n_sig = int(df_center["reject_h0"].sum())
                    logger.info(
                        "Saved %d rows → %s  (significant: %d / %d)",
                        len(df_center), out_path, n_sig, len(df_center),
                    )

        if args.do_subgroups:
            rng_sub = np.random.default_rng(42)
            for group_name in group_names:
                logger.info("--- Subgroup tests: %s / %s ---", hospital, group_name)
                df_sub = run_subgroup_tests(
                    results_path, model_types, hospital, group_name, metrics, args.alpha,
                    config=config, n_perms=args.n_permutations, rng=rng_sub,
                )
                if not df_sub.empty:
                    out_path = os.path.join(
                        results_path,
                        f"statistical_tests_subgroups_{group_name}_{hospital}.parquet",
                    )
                    df_sub.to_parquet(out_path, index=False)
                    n_sig = int(df_sub["reject_h0"].sum())
                    logger.info(
                        "Saved %d rows → %s  (significant: %d / %d)",
                        len(df_sub), out_path, n_sig, len(df_sub),
                    )

    logger.info("Statistical tests complete.")


if __name__ == "__main__":
    main()
