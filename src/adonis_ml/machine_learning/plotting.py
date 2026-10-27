import logging
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d

from adonis_ml.machine_learning.utils import get_figures_path, get_results_path

logger = logging.getLogger(__name__)


def _save_fig(fig, png_path: str, **kwargs) -> None:
    """Save figure as PNG and SVG (vectorized)."""
    fig.savefig(png_path, **kwargs)
    svg_path = os.path.splitext(png_path)[0] + ".svg"
    fig.savefig(svg_path, **kwargs)
    logger.info("Saved %s and %s", png_path, svg_path)


# ── shared publication style ─────────────────────────────────────────────────
# Both bar-chart (plot_models.py) and curve (plot_curves.py) scripts import
# these so every figure has the same axis style. Font family is left at the
# matplotlib default; only sizes, spines, grid, and export-safe font types are
# set. Apply via ``plt.rc_context(PUB_RC)`` — never mutate global rc.
PUB_RC: dict = {
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "pdf.fonttype":       42,   # embed TrueType (Illustrator-editable)
    "ps.fonttype":        42,
    "svg.fonttype":       "none",
}


def style_axes(ax) -> None:
    """Force the publication look on an already-drawn axes.

    Delegate plot functions in this module still call
    ``sns.set_theme("whitegrid")`` internally, which mutates global rc and
    re-enables the grid + top/right spines. Call this after those delegates
    to normalise the finished axes.
    """
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", labelsize=8,
                   width=0.8, length=3.0)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    if ax.get_title():
        ax.title.set_size(10)


METRIC_LABELS = {
    "roc_auc":           "AUROC",
    "p_roc_auc":         "Partial AUROC",
    "auprc":             "AUPRC",
    "accuracy":          "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision":         "PPV",
    "recall":            "Sensitivity",
    "specificity":       "Specificity",
    "f1":                "F1 Score",
    "neg_brier_score":   "Brier Score",
    "recall_conservative": "Conservative Sensitivity",
}

# Per-metric axis range overrides for subgroup plots. Low-prevalence metrics
# compress into the bottom of a [0, 1] range — zoom them in for readability.
METRIC_AXIS_LIMITS = {
    "precision": (0.0, 0.06),
    "auprc":     (0.0, 0.15),
    "f1":        (0.0, 0.11),
}

MODEL_LABELS = {
    "xgb":             "XGBoost",
    "lgbm":            "LightGBM",
    "tabpfn":          "TabPFN",
    "tabicl":          "TabICL",
    "lr":              "LR",
    "rf":              "RF",
    "svm":             "SVM",
    "knn":             "kNN",
    "naive":           "Naive Bayes",
    "hero_predictor":  "HRC index",
}

MODEL_GROUPS = {
    "lr":             "baselines",
    "knn":            "baselines",
    "svm":            "baselines",
    "rf":             "baselines",
    "xgb":            "baselines",
    "lgbm":           "baselines",
    "tabpfn":         "baselines",
    "tabicl":         "baselines",
    "hero_predictor": "hero",
}

GROUP_COLORS = {
    "baselines":  "#4393C3",
    "nonlinear":  "#D6604D",
    "foundation": "#4DAC26",
    "hero":       "#8856A7",
}

GROUP_LABELS = {
    "baselines":  "Baselines",
    "nonlinear":  "Non-linear",
    "foundation": "Foundation models",
    "hero":       "HeRO Score",
}

GROUP_ORDER = ["baselines", "nonlinear", "foundation", "hero"]

# Forest-plot geometry (per-figure sizing is inches-based inside each function).
_CAT_ROW_H: float = 0.55      # base height per category row
_CAT_MARGIN_H: float = 1.8    # fixed inches for x-labels + legend allowance

# Canonical display order within the full model list (used to sort legend entries)
_MODEL_ORDER: dict[str, int] = {
    "lr": 0, "knn": 1, "svm": 2, "rf": 3, "lgbm": 4, "xgb": 5,
    "tabpfn": 6, "tabicl": 7, "hero_predictor": 8,
}

# Individual color per model (used in all bar/curve plots)
# MODEL_COLORS: dict[str, str] = {
#     "lr":             "#4393C3",
#     "knn":            "#272988",
#     "svm":            "#B2182B",
#     "rf":             "#E08214",
#     "lgbm":           "#E6D81C",
#     "xgb":            "#D6604D",
#     "tabpfn":         "#4DAC26",
#     "tabicl":         "#1B7837",
#     "naive":          "#7F7F7F",
#     "hero_predictor": "#8856A7",
# }

# MODEL_COLORS = {
#     "lr":             "#6BAED6",  # blue-med
#     "knn":            "#2171B5",  # blue-dark
#     "svm":            "#08306B",  # blue-very-dark
#     "naive":          "#C6DBEF",  # blue-light
#     "rf":             "#FDAE6B",  # orange-med
#     "lgbm":           "#F16913",  # orange-dark
#     "xgb":            "#8C2D04",  # orange-very-dark
#     "tabpfn":         "#74C476",  # green-med
#     "tabicl":         "#238B45",  # green-dark
#     "hero_predictor": "#7F7F7F",  # neutral grey
# }

# MODEL_COLORS = {
#     "lr":             "#56B4E9",  # sky blue
#     "knn":            "#0072B2",  # blue
#     "svm":            "#003C6C",  # navy
#     "naive":          "#8FD3F4",  # light blue
#     "rf":             "#E69F00",  # orange
#     "lgbm":           "#D55E00",  # vermillion
#     "xgb":            "#8C2D04",  # dark vermillion
#     "tabpfn":         "#009E73",  # bluish green
#     "tabicl":         "#006D2C",  # dark green
#     "hero_predictor": "#7F7F7F",  # neutral grey
# }

MODEL_COLORS = {
    "lr":             "#9EBCD4",
    "knn":            "#5B8DB8",
    "svm":            "#285A85",
    "naive":          "#A6A6A6",
    "rf":             "#F0C36A",
    "lgbm":           "#D99A2B",
    "xgb":            "#A96800",
    "tabpfn":         "#4DAC26",
    "tabicl":         "#1B7837",
    "hero_predictor": "#9E2A2B",
}

# MODEL_COLORS = {
#     "lr":             "#482475",
#     "svm":            "#414487",
#     "rf":             "#355F8D",
#     "xgb":            "#2A788E",
#     "tabpfn":         "#22A884",
#     "tabicl":         "#44BF70",
#     "knn":            "#7AD151",
#     "naive":          "#BDDf26",
#     "hero_predictor": "#FDE725",
# }

CENTER_COLORS: dict[str, str] = {
    "CHUV":  "#1F76B2",
    "USZ":   "#F58318",
    "KISPI": "#77C07A",
}

CENTER_LINESTYLES: dict[str, str] = {
    "CHUV":  "-",
    "USZ":   "--",
    "KISPI": "-.",
}

# Positional palette for subgroup categories (Set2)
SUBGROUP_PALETTE: list[str] = [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
]

# Hatch patterns for subgroup categories: solid, full fill, slashes, dots, dashes, crosses
_SUBGROUP_HATCHES: list[str | None] = [None, "///", "...", "---", "xxx"]

# One fixed linestyle per group — all models in the same group share it
_GROUP_LINESTYLES: dict[str, str] = {
    "baselines":  "-",
    "nonlinear":  "--",
    "foundation": "-.",
    "hero":       ":",
}


def _model_style(model_type: str) -> tuple[str, str]:
    """Return (color, linestyle) for a model: individual color + one fixed linestyle per group."""
    group = MODEL_GROUPS.get(model_type, "baselines")
    color = MODEL_COLORS.get(model_type, "#999999")
    linestyle = _GROUP_LINESTYLES.get(group, "-")
    return color, linestyle


def _reset_style_counters() -> None:
    pass  # no longer needed; kept for call-site compatibility

# FIXME refactor non-dry code!


def plot_results(results, config, metrics, type="box", title_appendix=None):

    models = config.model["models"]
    labels = [model for model in models.keys()]

    data = {
        "model": [
            model
            for model in models.keys()
            for _ in range(config.cross_validation["outer_cv"].n_splits)
        ],
    }

    for metric in metrics:
        data[metric] = [
            result[metric]["values"][i]
            for _, result in results.items()
            for i in range(config.cross_validation["outer_cv"].n_splits)
        ]

    df = pd.DataFrame(data)

    for metric in metrics:
        plt.figure(figsize=(8, 6))

        if type == "box":
            sns.boxplot(
                data=df, x="model", y=metric, hue="model", dodge=True, width=0.8
            )

            # Plot single points
            sns.stripplot(
                data=df,
                x="model",
                y=metric,
                hue="model",
                dodge=True,
                jitter=False,
                alpha=0.5,
                marker="o",
                linewidth=0.5,
            )
        elif type == "violin":
            sns.violinplot(
                x="model",
                y=metric,
                hue="model",
                data=df,
                dodge=True,
                split=True,
                inner="quart",
            )
        elif type == "line":
            sns.lineplot(
                data=df,
                x="model",
                y=metric,
                hue="model",
                err_style="bars",
                errorbar=("se", 2),
            )
        else:
            raise NotImplementedError

        # Add axis labels
        plt.xlabel("Model")
        plt.ylabel("outer splits")

        # Limit axis ranges
        # plt.ylim(0, 1)

        # Add title
        plt.title(metric)

        # Add legend
        # handles, _ = plt.gca().get_legend_handles_labels()
        # plt.legend(handles=handles[:len(labels)], labels=labels, title="Model")

        plt.show()


_SUBGROUP_BOX_STYLES = [
    {"facecolor": "black",   "edgecolor": "black", "hatch": None},
    {"facecolor": "white",   "edgecolor": "black", "hatch": None},
    {"facecolor": "#888888", "edgecolor": "black", "hatch": ".."},
    {"facecolor": "#cccccc", "edgecolor": "black", "hatch": "--"},
    {"facecolor": "#444444", "edgecolor": "black", "hatch": "//"},
]

# # Tab10 palette for line plots
# _LINE_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]

# Set2 palette
# _LINE_COLORS = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]

# Okabe-Ito colorblind-safe palette
_LINE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

_LINE_STYLES = [
    {"linestyle": "-",   "marker": "o"},
    {"linestyle": "--",  "marker": "s"},
    {"linestyle": "-.",  "marker": "^"},
    {"linestyle": ":",   "marker": "D"},
    {"linestyle": "-",   "marker": "P"},
    {"linestyle": "--",  "marker": "X"},
]

_NON_METRIC_COLS = {"fold", "subgroup", "n", "n_sepsis", "n_controls", "n_sepsis_total", "n_controls_total", "marker", "model_type"}


def _tick_label(cat: str, counts: pd.DataFrame) -> str:
    if cat in counts.index:
        ns = int(counts.loc[cat, "n_sepsis"])
        nc = int(counts.loc[cat, "n_controls"])
        return f"{cat}\n({ns}/{nc})"
    return str(cat)


def plot_results_subgroups(config, df_results_subgroups, df_results, metrics: list[str] | None = None):
    figures_path = get_figures_path(config)
    model_types = list(config.model["models"].keys())
    model_labels = [MODEL_LABELS.get(m, m) for m in model_types]

    sns.set_theme(style="white", font_scale=1)

    for group_name, group_info in config.subgroups.items():
        frames = []
        for model_type in model_types:
            model_data = df_results_subgroups.get(model_type, {})
            if group_name not in model_data:
                logger.warning("Subgroup '%s' missing for model '%s', skipping", group_name, model_type)
                continue
            df_m = model_data[group_name].copy()
            df_m["model_type"] = MODEL_LABELS.get(model_type, model_type)
            frames.append(df_m)
        if not frames:
            continue
        df_combined = pd.concat(frames, ignore_index=True)

        # Force string dtype so seaborn palette dict keys always match hue values
        df_combined["subgroup"] = df_combined["subgroup"].astype(str)

        all_metrics = [col for col in df_combined.columns if col not in _NON_METRIC_COLS]
        if metrics is not None:
            unknown = [m for m in metrics if m not in all_metrics]
            if unknown:
                logger.warning("Unknown metrics (not in parquet), ignored: %s", unknown)
            metrics_to_plot = [m for m in metrics if m in all_metrics]
        else:
            metrics_to_plot = all_metrics
        metrics = metrics_to_plot
        n_metrics = len(metrics)
        hue_order = group_info.get("order", None)
        hue_order_str = [str(c) for c in hue_order] if hue_order else None

        subgroup_categories = hue_order_str or sorted(df_combined["subgroup"].unique())
        n_subgroups = len(subgroup_categories)

        # Use global totals (n_sepsis_total / n_controls_total) for legend labels —
        # these are unique stay_id counts across the full dataset, computed once in
        # run_evaluation_subgroups and stored as a constant per subgroup category.
        # Per-fold n_sepsis / n_controls remain in the parquet for sanity checking.
        if "n_sepsis_total" in df_combined.columns:
            first_label = model_labels[0]
            counts = (
                df_combined[df_combined["model_type"] == first_label]
                .drop_duplicates("subgroup")
                .set_index("subgroup")[["n_sepsis_total", "n_controls_total"]]
                .rename(columns={"n_sepsis_total": "n_sepsis", "n_controls_total": "n_controls"})
            )
        else:
            counts = pd.DataFrame()

        ncols = group_info.get("ncols", 3)
        nrows = int(np.ceil(n_metrics / ncols))
        single_col = ncols == 1
        sharex = single_col and nrows > 1
        fig_w = 8 if single_col else 5 * ncols
        fig_h = 4 * nrows
        label_fs = 13 if single_col else 11
        tick_fs = 11 if single_col else 10
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, fig_h),
            sharex=sharex,
            squeeze=False,
        )
        axes = axes.flatten()

        # Box width shrinks as subgroup categories grow to avoid overlap
        box_width = min(0.6, 0.8 / max(n_subgroups / 3, 1))

        # Build the shared figure legend once — same handles for every metric subplot.
        is_box_type = group_info["type"] in ["categorical", "custom", "range"]
        seen_groups = dict.fromkeys(
            g for g in GROUP_ORDER if any(MODEL_GROUPS.get(mt) == g for mt in model_types)
        )
        if is_box_type:
            # Group color patches (one per model group present)
            group_patches = [
                mpatches.Patch(facecolor=GROUP_COLORS[g], edgecolor="black", linewidth=1.0,
                               label=GROUP_LABELS[g])
                for g in seen_groups
            ]
            # Hatch patches for subgroup categories
            hatch_patches = []
            for i, cat in enumerate(subgroup_categories):
                hatch = _SUBGROUP_HATCHES[min(i, len(_SUBGROUP_HATCHES) - 1)]
                cat_str = str(cat)
                if cat_str in counts.index:
                    ns = int(counts.loc[cat_str, "n_sepsis"])
                    nc = int(counts.loc[cat_str, "n_controls"])
                    lbl = f"{cat_str} ({ns}/{nc})"
                else:
                    lbl = cat_str
                hatch_patches.append(mpatches.Patch(
                    facecolor="#aaaaaa", edgecolor="black",
                    hatch=hatch or "", label=lbl, linewidth=1.0,
                ))
            fig_legend_handles = group_patches + hatch_patches
        else:
            _reset_style_counters()
            _legend_styles = [(_model_style(mt), ml) for mt, ml in zip(model_types, model_labels)]
            fig_legend_handles = [
                plt.Line2D([0], [0], color=s[0], linestyle=s[1], markersize=5, linewidth=1.5, label=ml)
                for (s, ml) in _legend_styles
            ]
            _reset_style_counters()

        for ax, metric in zip(axes, metrics):
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))

            if is_box_type:
                # Draw boxes directly with matplotlib — no seaborn patch post-processing,
                # so styling is version-agnostic and deterministic.
                step = box_width / max(n_subgroups, 1)
                single_w = step * 0.85
                for mi, model_label in enumerate(model_labels):
                    model_type = model_types[mi]
                    group = MODEL_GROUPS.get(model_type, "baselines")
                    group_color = GROUP_COLORS.get(group, "#999999")
                    model_data = df_combined[df_combined["model_type"] == model_label]
                    for ci, cat in enumerate(subgroup_categories):
                        values = model_data[model_data["subgroup"] == cat][metric].dropna().values
                        if len(values) == 0:
                            continue
                        pos = mi + (ci - (n_subgroups - 1) / 2) * step
                        hatch = _SUBGROUP_HATCHES[min(ci, len(_SUBGROUP_HATCHES) - 1)]
                        bp = ax.boxplot(
                            values,
                            positions=[pos],
                            widths=[single_w],
                            patch_artist=True,
                            boxprops=dict(facecolor=group_color, edgecolor="black", linewidth=1.0),
                            medianprops=dict(color="white", linewidth=1.5),
                            whiskerprops=dict(color="black", linewidth=0.8),
                            capprops=dict(color="black", linewidth=0.8),
                            flierprops=dict(marker="o", markersize=3,
                                           markeredgecolor="black",
                                           markerfacecolor=group_color),
                            manage_ticks=False,
                        )
                        if hatch:
                            bp["boxes"][0].set_hatch(hatch)

                ax.set_xticks(range(len(model_labels)))
                ax.set_xticklabels(
                    [MODEL_LABELS.get(lbl, lbl) for lbl in model_labels],
                    rotation=30, ha="right", fontsize=tick_fs,
                )
                ax.set_xlim(-0.5, len(model_labels) - 0.5)

            elif group_info["type"] in ["date"]:
                x_cats = sorted(df_combined["subgroup"].unique().astype(str))
                x_pos = {cat: xi for xi, cat in enumerate(x_cats)}

                _reset_style_counters()
                for model_type, model_label in zip(model_types, model_labels):
                    color, linestyle = _model_style(model_type)
                    model_df = df_combined[df_combined["model_type"] == model_label]
                    agg = (
                        model_df.groupby("subgroup")[metric]
                        .agg(mean="mean", p5=lambda x: x.quantile(0.05), p95=lambda x: x.quantile(0.95))
                        .reindex(x_cats)
                        .reset_index()
                    )
                    xi = [x_pos[c] for c in agg["subgroup"]]
                    ax.plot(
                        xi, agg["mean"],
                        color=color, linewidth=1.5, label=model_label,
                        linestyle=linestyle, markersize=8, markeredgewidth=0.8,
                    )
                    ax.fill_between(xi, agg["p5"], agg["p95"], color=color, alpha=0.08)

                ax.set_xticks(range(len(x_cats)))
                ax.set_xticklabels(
                    [_tick_label(c, counts) for c in x_cats],
                    rotation=30, ha="right", fontsize=tick_fs,
                )

            # Remove per-subplot legend (replaced by single figure legend below)
            if ax.get_legend():
                ax.get_legend().remove()

            ax.set_ylabel(metric_label, fontsize=label_fs)
            ax.set_xlabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)

        for ax in axes[n_metrics:]:
            ax.set_visible(False)

        # For single-column layouts with shared x-axis: suppress tick labels on all
        # rows except the last visible one, and add a common x-axis label.
        if sharex:
            last_visible = min(n_metrics - 1, len(axes) - 1)
            for idx, ax in enumerate(axes[:n_metrics]):
                if idx < last_visible:
                    ax.tick_params(labelbottom=False)
            axes[last_visible].set_xlabel(group_info.get("xlabel", ""), fontsize=label_fs)

        fig.suptitle(group_info["label"], fontsize=label_fs + 1, fontweight="bold", y=1.01)

        # Single shared legend at the bottom.
        # Bottom margin is computed in inches so a tall single-column figure
        # doesn't waste space: ~0.5 in per legend row / total figure height.
        legend_ncol = min(len(fig_legend_handles), 5)
        legend_nrows = int(np.ceil(len(fig_legend_handles) / legend_ncol))
        legend_h_in = 0.4 * legend_nrows + 0.2
        bottom_margin = min(0.15, legend_h_in / fig_h)
        fig.legend(
            handles=fig_legend_handles,
            loc="lower center",
            ncol=legend_ncol,
            fontsize=9 if not single_col else 11,
            frameon=False,
            bbox_to_anchor=(0.5, 0),
        )
        fig.tight_layout(rect=[0, bottom_margin, 1, 1])

        out_path = os.path.join(figures_path, f"subgroup_{group_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
        plt.close(fig)


def plot_curve(
    config,
    df_results_curve,
    x_col,
    y_col,
    df_results,
    filename,
    ref_line=True,
    metric_suffix="",
    ax=None,
):
    figures_path = get_figures_path(config)
    _standalone = ax is None
    if _standalone:
        sns.set_theme(style="whitegrid", font_scale=1)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    _reset_style_counters()

    for model_type in _sorted_model_types(list(df_results_curve.keys())):
        df = df_results_curve[model_type]
        mean_x = np.linspace(0, 1, 100)
        ys = []

        for fold in df["fold"].unique():
            fold_df = df[df["fold"] == fold].sort_values(x_col)
            x = fold_df[x_col].values
            y = fold_df[y_col].values
            interp_y = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(0, 1))
            ys.append(interp_y(mean_x))

        ys = np.array(ys)
        mean_y = ys.mean(axis=0)
        std_y = ys.std(axis=0)

        y_lower = np.clip(mean_y - 1.96 * std_y / np.sqrt(len(ys)), 0, 1)
        y_upper = np.clip(mean_y + 1.96 * std_y / np.sqrt(len(ys)), 0, 1)

        metric = "roc_auc" if x_col == "fpr" else "auprc"
        mean_metric = df_results[model_type]["test"][f"test_{metric}{metric_suffix}"]["mean"]
        std_metric = df_results[model_type]["test"][f"test_{metric}{metric_suffix}"]["std"]

        model_label = MODEL_LABELS.get(model_type, model_type)
        color, _ = _model_style(model_type)
        ax.plot(mean_x, mean_y, color=color, linestyle="-", linewidth=2)
        ax.fill_between(mean_x, y_lower, y_upper, alpha=0.08, color=color)

    if ref_line:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    ax.set_xlabel("False positive rate" if x_col == "fpr" else "Recall (Sensitivity)")
    ax.set_ylabel("True positive rate" if y_col == "tpr" else "Precision (PPV)")
    # ax.legend(frameon=False)
    if _standalone:
        plt.tight_layout()
        out_path = os.path.join(figures_path, filename)
        _save_fig(plt.gcf(), out_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_dca(config, results_dca_curve, filename, ylim=(-0.001, 0.006), ax_prob=None, ax_or=None):
    figures_path = get_figures_path(config)

    _standalone = ax_prob is None and ax_or is None
    if _standalone:
        sns.set_theme(style="whitegrid", font_scale=1)
        fig, (ax_prob, ax_or) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)

    # Prevalence is stored explicitly on each pooled DCA dataframe (added by
    # run_evaluation_curve). Fall back to treat_all at the smallest threshold
    # for back-compat with older parquets that don't carry the column.
    ref_df = next((df for df in results_dca_curve.values() if not df.empty), None)
    prevalence = None
    if ref_df is not None:
        if "prevalence" in ref_df.columns:
            prevalence = float(ref_df["prevalence"].iloc[0])
        else:
            min_threshold = ref_df["threshold"].min()
            prevalence = float(ref_df.loc[ref_df["threshold"] == min_threshold, "treat_all"].mean())

    # Compute pt_max = threshold equivalent to OR=7, used as left-panel x-limit
    pt_max = 0.05  # fallback
    if prevalence is not None and 0 < prevalence < 1:
        prior_odds = prevalence / (1.0 - prevalence)
        pt_max = (7.0 * prior_odds) / (1.0 + 7.0 * prior_odds)

    def pt_to_or(pt_array):
        prior_odds = prevalence / (1.0 - prevalence)
        return (pt_array / (1.0 - pt_array)) / prior_odds

    # Reference lines — treat-none and treat-all on a shared analytical grid
    if prevalence is not None and 0 < prevalence < 1:
        thresholds_ref = np.linspace(1e-6, pt_max, 500)
        nb_treat_all = prevalence - (thresholds_ref / (1.0 - thresholds_ref)) * (1.0 - prevalence)

        for ax in filter(None, (ax_prob, ax_or)):
            ax.axhline(0, color="black", linewidth=1, linestyle=":", label="Treat none")

        if ax_prob is not None:
            ax_prob.plot(thresholds_ref, nb_treat_all,
                         color="black", linewidth=1, linestyle="--", label="Treat all")

        if ax_or is not None:
            ax_or.plot(pt_to_or(thresholds_ref), nb_treat_all,
                       color="black", linewidth=1, linestyle="--", label="Treat all")
            ax_or.axvline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    _reset_style_counters()
    for model_type in _sorted_model_types(list(results_dca_curve.keys())):
        df = results_dca_curve[model_type]
        if df.empty:
            logger.warning("No DCA data for model '%s', skipping", model_type)
            continue
        df = df.sort_values("threshold")
        thresholds = df["threshold"].to_numpy()
        nb = df["net_benefit"].to_numpy()

        color, _ = _model_style(model_type)
        if ax_prob is not None:
            ax_prob.plot(thresholds, nb, linewidth=2, color=color, linestyle="-")

        if ax_or is not None and prevalence is not None and prevalence > 0:
            or_x = pt_to_or(thresholds)
            ax_or.plot(or_x, nb, linewidth=2, color=color, linestyle="-")

    if ax_prob is not None:
        ax_prob.set_xlim(0, pt_max)
        ax_prob.set_xlabel("Threshold probability")
        ax_prob.set_ylim(*ylim)
        ax_prob.set_ylabel("Net benefit")

    if ax_or is not None:
        ax_or.set_xlim(0, 7)
        ax_or.set_xlabel("Odds ratio relative to prior")
        ax_or.set_ylim(*ylim)

    # Legend on ax_or when available, otherwise ax_prob
    # legend_ax = ax_or if ax_or is not None else ax_prob
    # if legend_ax is not None:
    #     legend_ax.legend(frameon=False)

    if _standalone:
        plt.tight_layout()
        out_path = os.path.join(figures_path, filename)
        _save_fig(plt.gcf(), out_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_calibration_curve(config, df_results_curve, filename="calibration_curves.png", ax=None):
    """Reliability diagram from the pooled OOF calibration_oof_bins parquet.

    Uses the 'quantile' binning strategy on log-log axes with Wilson
    confidence intervals as vertical error bars. Mirrors the quantile
    panel of scripts/plot_calibration.py.
    """
    figures_path = get_figures_path(config)

    _standalone = ax is None
    if _standalone:
        sns.set_theme(style="whitegrid", font_scale=1)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    _reset_style_counters()

    all_x: list[float] = []
    for model_type in _sorted_model_types(list(df_results_curve.keys())):
        bins_df = df_results_curve[model_type]
        if bins_df is None or bins_df.empty:
            continue
        df = bins_df[bins_df["strategy"] == "quantile"].sort_values("prob_pred")
        if df.empty:
            continue

        prob_pred = df["prob_pred"].to_numpy()
        prob_true = df["prob_true"].to_numpy()
        lower = df["wilson_lower"].to_numpy()
        upper = df["wilson_upper"].to_numpy()

        color = MODEL_COLORS.get(model_type, "#999999")

        ax.errorbar(
            prob_pred, prob_true,
            yerr=[np.maximum(prob_true - lower, 0), np.maximum(upper - prob_true, 0)],
            fmt="none", color=color, alpha=0.35,
        )
        ax.plot(prob_pred, prob_true, marker="o", linestyle="-",
                color=color, linewidth=1.5, markersize=4)

        all_x.extend(prob_pred.tolist())

    if not all_x:
        if _standalone:
            plt.close()
        return

    x_lo = 1e-3
    x_hi = float(max(all_x))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([x_lo, x_hi], [x_lo, x_hi], "k--", linewidth=1)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(x_lo, x_hi)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.grid(alpha=0.3, which="both")
    # ax.legend(frameon=False)
    if _standalone:
        plt.tight_layout()
        out_path = os.path.join(figures_path, filename)
        _save_fig(plt.gcf(), out_path, dpi=300, bbox_inches="tight")
        plt.close()


def _sorted_model_types(model_types: list[str]) -> list[str]:
    """Sort models by GROUP_ORDER then by _MODEL_ORDER within group; unknowns appended last."""
    def _rank(m):
        g = MODEL_GROUPS.get(m, "")
        group_rank = GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
        return (group_rank, _MODEL_ORDER.get(m, 999))
    return sorted(model_types, key=_rank)


def _model_x_positions(model_types: list[str], gap: float = 0.5) -> np.ndarray:
    """Return bar x-positions with an extra gap between groups."""
    positions = []
    pos = 0.0
    prev_group = None
    for m in model_types:
        group = MODEL_GROUPS.get(m, "")
        if prev_group is not None and group != prev_group:
            pos += gap
        positions.append(pos)
        pos += 1.0
        prev_group = group
    return np.array(positions)


def plot_subgroups_by_category(
    config,
    df_results_subgroups: dict,
    group_name: str,
    metrics: list[str] | None = None,
    model_types: list[str] | None = None,
    ncols: int = 2,
) -> None:
    """Forest plot: subgroup categories on y-axis, one dot ± std per model, colored by MODEL_COLORS."""
    from matplotlib.lines import Line2D

    figures_path = get_figures_path(config)
    group_info = config.subgroups.get(group_name, {})

    if model_types is None:
        model_types = list(df_results_subgroups.keys())
    model_types = _sorted_model_types([m for m in model_types if m in df_results_subgroups])

    frames = []
    for mt in model_types:
        sub = df_results_subgroups[mt].get(group_name)
        if sub is None or sub.empty:
            continue
        df_m = sub.copy()
        df_m["model_type"] = mt
        frames.append(df_m)
    if not frames:
        logger.warning("plot_subgroups_by_category: no data for group '%s'", group_name)
        return
    df_combined = pd.concat(frames, ignore_index=True)
    df_combined["subgroup"] = df_combined["subgroup"].astype(str)

    all_metric_cols = [c for c in df_combined.columns if c not in _NON_METRIC_COLS and c != "model_type"]
    if metrics is not None:
        metrics = [m for m in metrics if m in all_metric_cols]
    else:
        metrics = all_metric_cols
    if not metrics:
        return

    hue_order = group_info.get("order", None)
    subgroup_categories = [str(c) for c in hue_order] if hue_order else sorted(df_combined["subgroup"].unique())
    n_cats = len(subgroup_categories)
    n_models = len(model_types)

    cat_step = 1.20                                    # inter-category y-spacing
    spacing = 1.05 / max(n_models, 1)
    intra_offset = (np.arange(n_models) - (n_models - 1) / 2.0) * spacing

    tick_labels = list(subgroup_categories)

    model_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODEL_COLORS.get(mt, "#999999"),
               markersize=6, label=MODEL_LABELS.get(mt, mt))
        for mt in model_types
    ]

    n_metrics = len(metrics)
    ncols = max(1, min(ncols, n_metrics))          # never leave empty columns
    nrows = int(np.ceil(n_metrics / ncols))

    # Inches-based layout. Each vertical block is reserved explicitly so the
    # legend and x-label never overlap the axes, regardless of nrows/ncols.
    row_in     = max(0.60, 0.070 * n_models + 0.28)   # per-category row height
    axes_h_in  = row_in * ((n_cats - 1) * cat_step + 1.4)  # axes content only
    xlabel_in  = 0.45                                 # metric name below axes
    top_in     = 0.35                                 # suptitle
    legend_in  = 1.10                                 # figure-level legend
    row_gap_in = 0.55                                 # gap between subplot rows
    subplot_w  = 3.8
    fig_w      = subplot_w * ncols
    fig_h      = (top_in + axes_h_in * nrows
                  + row_gap_in * (nrows - 1) + xlabel_in + legend_in)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                 dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, metrics):
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))

            for mi, mt in enumerate(model_types):
                color = MODEL_COLORS.get(mt, "#999999")
                for ci, cat in enumerate(subgroup_categories):
                    rows = df_combined[(df_combined["model_type"] == mt) & (df_combined["subgroup"] == cat)]
                    if rows.empty or metric not in rows.columns:
                        continue
                    vals = rows[metric].dropna()
                    if vals.empty:
                        continue
                    y_pos = ci * cat_step + intra_offset[mi]
                    ax.errorbar(
                        vals.mean(), y_pos, xerr=vals.std(),
                        fmt="o", color=color, markersize=4,
                        capsize=2.5, elinewidth=0.7, linewidth=0.8, zorder=3,
                    )

            ax.set_yticks([ci * cat_step for ci in range(n_cats)])
            ax.set_yticklabels(tick_labels)
            ax.set_ylim(-0.7, (n_cats - 1) * cat_step + 0.7)
            ax.set_xlim(*METRIC_AXIS_LIMITS.get(metric, (0.0, 1.0)))
            ax.set_xlabel(metric_label)
            ax.set_ylabel("")
            style_axes(ax)

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        legend_ncol = min(len(model_handles), 3)
        fig.legend(model_handles, [h.get_label() for h in model_handles],
                   loc="lower center", ncol=legend_ncol,
                   frameon=False, bbox_to_anchor=(0.5, 0.5 * legend_in / fig_h),
                   columnspacing=1.6, handletextpad=0.6, handlelength=1.4)
        group_title = group_info.get("label", group_name)
        fig.suptitle(group_title, fontsize=10)
        fig.subplots_adjust(
            top=1 - top_in / fig_h,
            bottom=(legend_in + xlabel_in) / fig_h,
            left=0.14 / ncols + 0.06,
            right=0.98,
            hspace=(row_gap_in / axes_h_in) if nrows > 1 else 0.0,
            wspace=0.30,
        )

        out_path = os.path.join(figures_path, f"subgroups_{group_name}_by_category.png")
        _save_fig(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_subgroups_by_model(
    config,
    df_results_subgroups: dict,
    group_name: str,
    metrics: list[str] | None = None,
    model_types: list[str] | None = None,
    ncols: int = 2,
) -> None:
    """Bar chart: models on x-axis, one bar per subgroup category, colored by SUBGROUP_PALETTE.

    One subplot per metric. Models ordered by group with gap between groups.
    """
    figures_path = get_figures_path(config)
    group_info = config.subgroups.get(group_name, {})

    if model_types is None:
        model_types = list(df_results_subgroups.keys())
    model_types = _sorted_model_types([m for m in model_types if m in df_results_subgroups])

    frames = []
    for mt in model_types:
        sub = df_results_subgroups[mt].get(group_name)
        if sub is None or sub.empty:
            continue
        df_m = sub.copy()
        df_m["model_type"] = mt
        frames.append(df_m)
    if not frames:
        logger.warning("plot_subgroups_by_model: no data for group '%s'", group_name)
        return
    df_combined = pd.concat(frames, ignore_index=True)
    df_combined["subgroup"] = df_combined["subgroup"].astype(str)

    all_metric_cols = [c for c in df_combined.columns if c not in _NON_METRIC_COLS and c != "model_type"]
    if metrics is not None:
        metrics = [m for m in metrics if m in all_metric_cols]
    else:
        metrics = all_metric_cols
    if not metrics:
        return

    hue_order = group_info.get("order", None)
    subgroup_categories = [str(c) for c in hue_order] if hue_order else sorted(df_combined["subgroup"].unique())
    n_cats = len(subgroup_categories)

    x_model = _model_x_positions(model_types)
    bar_width = 0.7 / max(n_cats, 1)

    cat_handles = [
        mpatches.Patch(facecolor=SUBGROUP_PALETTE[ci % len(SUBGROUP_PALETTE)], edgecolor="black",
                       linewidth=0.5, label=str(cat))
        for ci, cat in enumerate(subgroup_categories)
    ]

    n_metrics = len(metrics)
    ncols = max(1, min(ncols, n_metrics))
    nrows = int(np.ceil(n_metrics / ncols))

    subplot_w  = 3.8
    axes_h_in  = 2.4                                  # axes content only
    xlabel_in  = 0.55                                 # rotated model tick labels
    top_in     = 0.35
    legend_in  = 1.10
    row_gap_in = 0.65
    fig_w      = subplot_w * ncols
    fig_h      = (top_in + axes_h_in * nrows
                  + row_gap_in * (nrows - 1) + xlabel_in + legend_in)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                 dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, metrics):
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))

            for ci, cat in enumerate(subgroup_categories):
                x_offset = (ci - (n_cats - 1) / 2) * bar_width
                color = SUBGROUP_PALETTE[ci % len(SUBGROUP_PALETTE)]
                for mi, mt in enumerate(model_types):
                    df_m = df_combined[(df_combined["model_type"] == mt) & (df_combined["subgroup"] == cat)]
                    if df_m.empty or metric not in df_m.columns:
                        continue
                    vals = df_m[metric].dropna()
                    if vals.empty:
                        continue
                    ax.bar(
                        x_model[mi] + x_offset, vals.mean(), width=bar_width,
                        yerr=vals.std(), capsize=2.5, color=color, edgecolor="black",
                        linewidth=0.5, ecolor="black",
                        error_kw={"linewidth": 0.7},
                    )

            ax.set_xticks(x_model)
            ax.set_xticklabels([MODEL_LABELS.get(mt, mt) for mt in model_types],
                               rotation=30, ha="right")
            ax.set_xlim(x_model[0] - 0.6, x_model[-1] + 0.6)
            y_lo, y_hi = METRIC_AXIS_LIMITS.get(metric, (0.0, None))
            ax.set_ylim(bottom=y_lo, top=y_hi)
            ax.set_ylabel(metric_label)
            ax.set_xlabel("")
            style_axes(ax)

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        legend_ncol = min(len(cat_handles), 5)
        fig.legend(cat_handles, [h.get_label() for h in cat_handles],
                   loc="lower center", ncol=legend_ncol,
                   frameon=False, bbox_to_anchor=(0.5, 0.5 * legend_in / fig_h),
                   columnspacing=1.6, handletextpad=0.6, handlelength=1.4)
        group_title = group_info.get("label", group_name)
        fig.suptitle(group_title, fontsize=10)
        fig.subplots_adjust(
            top=1 - top_in / fig_h,
            bottom=(legend_in + xlabel_in) / fig_h,
            left=0.14 / ncols + 0.06,
            right=0.98,
            hspace=(row_gap_in / axes_h_in) if nrows > 1 else 0.0,
            wspace=0.30,
        )

        out_path = os.path.join(figures_path, f"subgroups_{group_name}_by_model.png")
        _save_fig(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_metrics_over_time(config, df_results_metrics):
    """
    Plot metrics over time, one metric per subplot.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame with one column for time (`time_col`) and one column per metric.
    time_col : str, default="time_difference_hours_sepsis"
        Name of the time column in df_results.
    figsize : tuple, default=(10, 10)
        Size of the whole figure.
    """
    # Use a scientific style
    sns.set_theme(style="whitegrid", font_scale=1)

    metrics = [col for col in df_results_metrics.columns if col not in ["model", "time_difference_hours_sepsis"]]
    n_metrics = len(metrics)

    models = df_results_metrics["model"].unique()

    # Define style cycles
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]
    colors = sns.color_palette("tab10")

    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))

    figsize=(10, 10)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]  # ensure iterable
    
    for ax, metric in zip(axes.flatten(), metrics):
        # Plot each model separately with consistent style
        for i, model in enumerate(models):
            sns.lineplot(
                data=df_results_metrics[df_results_metrics["model"] == model],
                x="time_difference_hours_sepsis",
                y=metric,
                label=MODEL_LABELS.get(model, model),
                marker=markers[i],
                linestyle=linestyles[i],
                color=colors[i % len(colors)],
                linewidth=2.0,
                ax=ax,
            )
   
        # Add red shaded region from -12 to 0
        ax.axvspan(-12, 0, color="red", alpha=0.1)

        ax.set_ylabel(metric, weight="bold")
        ax.set_xlabel("")  # only last subplot shows x-label
        ax.grid(True, linestyle="--", alpha=0.7)

        sns.move_legend(ax, "upper left")
    
    # Only bottom subplot gets x-label
    for ax in axes[-ncols,:]:
        ax.set_xlabel("Time to CRASH (hours)")
    
    out_path = os.path.join(get_figures_path(config), f"hero_evaluation_metrics_{'_'.join(models)}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)

    # Zoomed-in view [-12, 0]
    for ax in axes.flatten():
        # ax.collections.clear()  # Remove previous shaded region
        ax.set_xlim(-12, 0)
    out_path_zoom = os.path.join(get_figures_path(config), f"hero_evaluation_metrics_zoom_{'_'.join(models)}.png")
    plt.savefig(out_path_zoom, dpi=300, bbox_inches="tight", transparent=False)

    plt.close(fig)  


def plot_probability_over_time(config, df_results_prob):
    # Compute mean + 5–95th percentile per timepoint & group
    summary = (
        df_results_prob
        .groupby(["model", "y_true", "time_difference_hours_sepsis"])
        .agg(
            prob_mean=("y_proba", "mean"),
            lower_bound=("y_proba", lambda x: np.percentile(x, 5)),
            upper_bound=("y_proba", lambda x: np.percentile(x, 95))
        )
        .reset_index()
    )

    models = df_results_prob["model"].unique()
    n_models = len(models)

    # Set style
    sns.set_theme(style="whitegrid", font_scale=1)
    palette = ["#1f77b4", "#d62728"] 
    labels = ["Control", "Sepsis"]

    fig, axes = plt.subplots(
        n_models, 1, figsize=(7, 3.5 * n_models), sharex=True, dpi=300
    )

    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        data = summary[summary["model"] == model]

        for i in range(2):  # fixed order/colors
            subset = data[data["y_true"] == i]
            ax.plot(
                subset["time_difference_hours_sepsis"], subset["prob_mean"],
                label=labels[i],
                color=palette[i], lw=2.0
            )
            ax.fill_between(
                subset["time_difference_hours_sepsis"],
                subset["lower_bound"],
                subset["upper_bound"],
                color=palette[i],
                alpha=0.15
            )

        ax.axvline(x=-1, linestyle="--", color="gray", lw=1)
        # ax.axhline(y=0.5, linestyle="--", color="red", lw=1)

        ax.set_title(f"Model: {model}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted probability", fontsize=11)

        ax.set_xlim([-48.5, 0.5])
        # ax.set_ylim([0, 1])

    axes[-1].set_xlabel("Time to CRASH (hours)", fontsize=11)

    # Common legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.savefig(os.path.join(get_figures_path(config), f"hero_evaluation_prob_{'_'.join(models)}.png"), dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)  


def plot_or_over_time(config, df_results_or):
    """Plot model odds ratio (and raw HeRO OR) over time before CRASH.

    Expects a DataFrame with columns:
        model, y_true, time_difference_hours_sepsis, y_or
    For ML models y_or is the probability converted to OR; for 'hero' it is the
    raw HeRO score.
    """
    summary = (
        df_results_or
        .groupby(["model", "y_true", "time_difference_hours_sepsis"])
        .agg(
            or_mean=("y_or", "mean"),
            lower_bound=("y_or", lambda x: np.percentile(x, 5)),
            upper_bound=("y_or", lambda x: np.percentile(x, 95)),
        )
        .reset_index()
    )

    models = df_results_or["model"].unique()
    n_models = len(models)

    sns.set_theme(style="whitegrid", font_scale=1)
    palette = ["#1f77b4", "#d62728"]
    labels = ["Control", "Sepsis"]

    fig, axes = plt.subplots(
        n_models, 1, figsize=(7, 3.5 * n_models), sharex=True, dpi=300
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        data = summary[summary["model"] == model]

        for i in range(2):
            subset = data[data["y_true"] == i]
            ax.plot(
                subset["time_difference_hours_sepsis"], subset["or_mean"],
                label=labels[i],
                color=palette[i], lw=2.0,
            )
            ax.fill_between(
                subset["time_difference_hours_sepsis"],
                subset["lower_bound"],
                subset["upper_bound"],
                color=palette[i],
                alpha=0.15,
            )

        ax.axhline(y=1, linestyle="--", color="gray", lw=1)
        ax.axvline(x=-1, linestyle="--", color="gray", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"Model: {model}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Odds Ratio", fontsize=11)
        ax.set_xlim([-48.5, 0.5])

    axes[-1].set_xlabel("Time to CRASH (hours)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(
        get_figures_path(config),
        f"hero_evaluation_or_{'_'.join(models)}.png",
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def plot_results_fractional(
    results, config, metrics, type="box", title_appendix=None, fractions=None
):

    models = config.model["models"]
    labels = [model for model in models.keys()]

    data = {
        "fraction": [
            f"{int(100*p)}%"
            for p in fractions
            for _ in range(config.cross_validation["outer_cv"].n_splits)
        ]
        * len(models),
        "model": [
            model
            for model in models.keys()
            for _ in range(
                config.cross_validation["outer_cv"].n_splits * len(fractions)
            )
        ],
    }

    for metric in metrics:
        data[metric] = [
            result[p][metric]["values"][i]
            for _, result in results.items()
            for p in fractions
            for i in range(config.cross_validation["outer_cv"].n_splits)
        ]
        data[f"{metric}_mean"] = [
            result[p][metric]["mean"]
            for _, result in results.items()
            for p in fractions
            for _ in range(config.cross_validation["outer_cv"].n_splits)
        ]

    df = pd.DataFrame(data)

    for metric in metrics:
        plt.figure(figsize=(8, 6))

        if type == "box":
            sns.boxplot(
                data=df,
                x="fraction",
                y=metric,
                hue="model",
                dodge=True,
                width=0.8,
                gap=0.1,
            )

            # Plot single points
            sns.stripplot(
                data=df,
                x="fraction",
                y=metric,
                hue="model",
                dodge=True,
                jitter=False,
                alpha=0.5,
                marker="o",
                linewidth=0.5,
            )
        elif type == "violin":
            sns.violinplot(
                x="fraction",
                y=metric,
                hue="model",
                data=df,
                dodge=True,
                split=True,
                inner="quart",
            )
        elif type == "line":
            sns.lineplot(
                data=df,
                x="fraction",
                y=metric,
                hue="model",
                err_style="bars",
                errorbar=("se", 2),
            )
        else:
            raise NotImplementedError

        # Add axis labels
        plt.xlabel("Percentage of Training Data")
        plt.ylabel(metric)

        # Add legend
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles[: len(labels)], labels=labels, title="Model")

        plt.show()


def plot_results_fractional_multirow(
    results,
    config,
    metrics,
    type="box",
    title_appendix=None,
    fractions=None,
    nrow=2,
    ncol=None,
):
    models = config.model["models"]
    labels = [model for model in models.keys()]

    data = {
        "fraction": [
            f"{int(100*p)}%"
            for p in fractions
            for _ in range(config.cross_validation["outer_cv"].n_splits)
        ]
        * len(models),
        "model": [
            model
            for model in models.keys()
            for _ in range(
                config.cross_validation["outer_cv"].n_splits * len(fractions)
            )
        ],
    }

    for metric in metrics:
        data[metric] = [
            result[p][metric]["values"][i]
            for _, result in results.items()
            for p in fractions
            for i in range(config.cross_validation["outer_cv"].n_splits)
        ]
        data[f"{metric}_mean"] = [
            result[p][metric]["mean"]
            for _, result in results.items()
            for p in fractions
            for _ in range(config.cross_validation["outer_cv"].n_splits)
        ]

    df = pd.DataFrame(data)

    if ncol is None:
        ncol = int(len(metrics) / nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 4))

    for n, metric in enumerate(metrics):

        i = n // ncol
        j = n % ncol

        if type == "box":
            sns.boxplot(
                data=df,
                x="fraction",
                y=metric,
                hue="model",
                dodge=True,
                width=0.8,
                gap=0.1,
                ax=axes[i, j],
                legend=False,
            )

            # Plot single points
            # sns.stripplot(data=df, x="fraction", y=metric, hue="model", dodge=True, jitter=False, alpha=0.5, marker="o", linewidth=0.5, ax=axes[i], legend=False)
        elif type == "violin":
            sns.violinplot(
                x="fraction",
                y=metric,
                hue="model",
                data=df,
                dodge=True,
                split=True,
                inner="quart",
            )
        elif type == "line":
            sns.lineplot(
                data=df,
                x="fraction",
                y=metric,
                hue="model",
                err_style="bars",
                errorbar=("se", 2),
            )
        else:
            raise NotImplementedError

        # Add axis labels
        axes[i, j].set_xlabel("")
        axes[i, j].set_ylabel(metric, fontsize=12)

        # Add legend
        if i == 0:
            handles, _ = axes[i, j].get_legend_handles_labels()
            handles = handles[: len(labels)]

    fig.text(0.5, 0.04, "Percentage of Training Data", ha="center", fontsize=10.5)
    fig.subplots_adjust(top=0.93)

    legend = [
        {
            "label": "SVM",
            "color": "tab:blue",
        },
        {
            "label": "TabPFN",
            "color": "tab:orange",
        },
    ]

    handles = []
    for line in legend:
        handles.append(
            mpatches.Patch(
                facecolor=line["color"], edgecolor="black", alpha=1, label=line["label"]
            )
        )

    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=12)


def plot_shap_beeswarm(
    config,
    shap_result: dict,
    model_type: str,
    top_n: int = 20,
) -> None:
    """Beeswarm (dot) SHAP summary plot for one model type.

    Parameters
    ----------
    config : AdonisConfig
    shap_result : dict returned by compute_shap_fold0 / run_shap_analysis
    model_type : e.g. "xgb", "tabpfn"
    top_n : number of top features to display (ranked by mean |SHAP|)
    """
    import shap as shap_lib
    from adonis_ml.machine_learning.utils import format_feature_name

    figures_path = get_figures_path(config)
    model_figures_path = os.path.join(figures_path, model_type)
    os.makedirs(model_figures_path, exist_ok=True)

    formatted_names = [format_feature_name(n) for n in shap_result["feature_names"]]

    shap_lib.summary_plot(
        shap_result["shap_values"],
        shap_result["X_display"],
        feature_names=formatted_names,
        max_display=top_n,
        show=False,
        plot_type="dot",
    )

    fig = plt.gcf()
    fig.set_size_inches(12, max(5, top_n * 0.4 + 2))

    # Shrink colorbar font
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=10)
        cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=10)

    out_path = os.path.join(model_figures_path, "shap_beeswarm.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_scatter(
    config,
    shap_result: dict,
    model_type: str,
    top_n: int = 20,
) -> None:
    """One scatter plot per top-N feature: feature value (x) vs SHAP value (y)
    with marginal distributions and a LOWESS trend line with 95% bootstrap CI.

    Parameters
    ----------
    config : AdonisConfig
    shap_result : dict returned by compute_shap_fold0 / run_shap_analysis
    model_type : e.g. "xgb", "tabpfn"
    top_n : number of top features (ranked by mean |SHAP|) to plot
    """
    from adonis_ml.machine_learning.utils import format_feature_name

    figures_path = get_figures_path(config)
    scatter_path = os.path.join(figures_path, model_type, "shap_scatter")
    os.makedirs(scatter_path, exist_ok=True)

    shap_values   = shap_result["shap_values"]                          # (n_samples, n_features)
    X_raw         = shap_result.get("X_raw", shap_result["X_display"])  # raw unscaled values
    feature_names = shap_result["feature_names"]

    # Rank by mean |SHAP| and pick top_n
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    for rank, feat_idx in enumerate(top_idx):
        raw_name       = feature_names[feat_idx]
        formatted_name = format_feature_name(raw_name)
        feat_vals      = np.asarray(X_raw[:, feat_idx], dtype=float)
        shap_vals      = np.asarray(shap_values[:, feat_idx], dtype=float)

        # JointGrid: x = feature value, y = SHAP value
        g = sns.JointGrid(height=5, ratio=4)

        g.ax_joint.scatter(
            feat_vals, shap_vals,
            color="#2196F3", alpha=0.4, s=8, linewidths=0,
        )
        g.ax_joint.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

        # Binned mean + 5th/95th percentile of SHAP values
        mask = np.isfinite(feat_vals) & np.isfinite(shap_vals)
        if mask.sum() >= 10:
            fx, sy = feat_vals[mask], shap_vals[mask]
            bins = np.linspace(fx.min(), fx.max(), 21)
            bin_idx = np.digitize(fx, bins[1:-1])
            bin_centers, means, p5, p95 = [], [], [], []
            for b in np.unique(bin_idx):
                ys = sy[bin_idx == b]
                if len(ys) < 3:
                    continue
                bin_centers.append(np.median(fx[bin_idx == b]))
                means.append(np.mean(ys))
                p5.append(np.percentile(ys, 5))
                p95.append(np.percentile(ys, 95))
            if len(bin_centers) > 1:
                bc = np.array(bin_centers)
                order = np.argsort(bc)
                bc = bc[order]
                means = np.array(means)[order]
                p5 = np.array(p5)[order]
                p95 = np.array(p95)[order]
                g.ax_joint.plot(bc, means, color="#E53935", linewidth=1.8, label="Mean (90% CI)")
                g.ax_joint.fill_between(bc, p5, p95, color="#E53935", alpha=0.2)
                g.ax_joint.legend(fontsize=8, frameon=False, loc="best")

        g.ax_joint.set_xlabel(formatted_name, fontsize=10)
        g.ax_joint.set_ylabel("SHAP value", fontsize=10)

        # Marginal distributions
        sns.histplot(x=feat_vals, ax=g.ax_marg_x, bins=50, color="#555", alpha=0.7, linewidth=0)
        sns.histplot(y=shap_vals, ax=g.ax_marg_y, bins=50, color="#555", alpha=0.7, linewidth=0)

        safe_name = raw_name.replace("/", "_")
        out_path  = os.path.join(scatter_path, f"{rank + 1:02d}_{safe_name}.png")
        g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(g.figure)

    logger.info("SHAP scatter plots saved to %s", scatter_path)


def save_subgroups_markdown(config, results_subgroups):
    results_path = get_results_path(config)

    for model_type, subgroup_scores in results_subgroups.items():
        model_path = os.path.join(results_path, model_type)
        os.makedirs(model_path, exist_ok=True)

        for group_name, df in subgroup_scores.items():
            if not df.empty:
                pq_path = os.path.join(model_path, f"subgroups_{group_name}.parquet")
                df.to_parquet(pq_path, index=False)

        out_path = os.path.join(model_path, "subgroups.md")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Subgroup Analysis — {model_type}\n\n")

            for group_name, df in subgroup_scores.items():
                if df.empty:
                    continue

                label = config.subgroups.get(group_name, {}).get("label", group_name)
                f.write(f"## {label}\n\n")

                metric_cols = [c for c in df.columns if c not in ("fold", "subgroup", "n")]

                # Per-fold detail table
                header = ["Subgroup", "Fold", "N"] + metric_cols
                f.write("| " + " | ".join(header) + " |\n")
                f.write("|" + "|".join(["---"] * len(header)) + "|\n")
                for _, row in df.sort_values(["subgroup", "fold"]).iterrows():
                    vals = (
                        [str(row["subgroup"]), str(int(row["fold"])), str(int(row["n"]))]
                        + [f"{row[m]:.3f}" if pd.notna(row[m]) else "—" for m in metric_cols]
                    )
                    f.write("| " + " | ".join(vals) + " |\n")

                f.write("\n**Summary (mean ± std across folds)**\n\n")

                # Summary table
                summary_header = ["Subgroup", "Mean N"] + metric_cols
                f.write("| " + " | ".join(summary_header) + " |\n")
                f.write("|" + "|".join(["---"] * len(summary_header)) + "|\n")
                for subgroup_val, grp in df.groupby("subgroup", observed=True):
                    mean_n = f"{grp['n'].mean():.1f}"
                    metric_vals = [
                        f"{grp[m].mean():.3f}±{grp[m].std():.3f}" if pd.notna(grp[m].mean()) else "—"
                        for m in metric_cols
                    ]
                    f.write("| " + " | ".join([str(subgroup_val), mean_n] + metric_vals) + " |\n")

                f.write("\n---\n\n")

        logger.info("Subgroup markdown saved: %s", out_path)


def plot_metrics_over_time_windowed(
    config,
    model_metrics: dict,
    metrics: list | None = None,
) -> None:
    """Plot classification metrics vs relative window hour (1–12 = −12 to −1 h before sepsis).

    Parameters
    ----------
    config : AdonisConfig
    model_metrics : dict[model_type, pd.DataFrame]
        Keys are model type strings.  Each DataFrame has columns
        [fold, relative_hour, metric1, metric2, ...].
    metrics : list[str] | None
        Subset of metric columns to plot. If None, all metric columns are used.
    """
    if not model_metrics:
        logger.warning("plot_metrics_over_time_windowed: model_metrics is empty, nothing to plot.")
        return

    first_df = next(iter(model_metrics.values()))
    all_metric_cols = [c for c in first_df.columns if c not in ("fold", "relative_hour")]
    if metrics is not None:
        missing = [m for m in metrics if m not in all_metric_cols]
        if missing:
            logger.warning("plot_metrics_over_time_windowed: requested metrics not found in parquet: %s", missing)
        metrics = [m for m in metrics if m in all_metric_cols]
    else:
        metrics = all_metric_cols
    if not metrics:
        logger.warning("plot_metrics_over_time_windowed: no metric columns found.")
        return

    tick_labels = [str(h) for h in range(-12, 0)]
    out_dir = os.path.join(get_figures_path(config), "metrics_over_time")
    os.makedirs(out_dir, exist_ok=True)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))

        for mi, (model_type, df) in enumerate(model_metrics.items()):
            if metric not in df.columns:
                continue
            color = _LINE_COLORS[mi % len(_LINE_COLORS)]
            label = MODEL_LABELS.get(model_type, model_type)

            agg = (
                df.groupby("relative_hour")[metric]
                .agg(
                    mean="mean",
                    p5=lambda x: x.quantile(0.05),
                    p95=lambda x: x.quantile(0.95),
                )
                .reset_index()
                .sort_values("relative_hour")
            )

            style = _LINE_STYLES[min(mi, len(_LINE_STYLES) - 1)]
            ax.plot(
                agg["relative_hour"], agg["mean"],
                color=color, linewidth=2.0, label=label,
                linestyle=style["linestyle"],
                marker=style["marker"], markersize=6,
                markeredgewidth=0.8,
            )
            ax.fill_between(agg["relative_hour"], agg["p5"], agg["p95"], color=color, alpha=0.15)

        ax.set_xlim(1, 12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Hours before sepsis", fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ")), fontsize=11)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.set_title("Metrics over time — 12-h prediction window", fontsize=11, fontweight="bold")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"metrics_over_time_{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out_path)
