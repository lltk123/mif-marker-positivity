#!/usr/bin/env python3
"""Plot frozen TabM ablations from evaluation CSVs, with no model selection.

Usage: python scripts/plot_tabm.py --run-dir /path/to/run [--font-path font.ttf]
Every panel has source CSV, Chinese/English PNG, and editable-text SVG exports.
Seed variation and core-level variation are descriptive; neither is a patient CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TEST = "external_val_TMA_cores"
FAMILY_COLORS = {"TabM": "#507EA1", "HGB": "#B39072", "GMM": "#666666"}
MODES = ["fixed_0.5", "validation_selected"]
CONTEXTS = ("TMA1_CD8", "TMA2_CD8", "TMA3_CD8")
TEXT = {
    "en": {
        "fixed_0.5": "Fixed threshold 0.5",
        "validation_selected": "Validation-selected threshold",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "feature": "Feature set",
        "external": "External test",
        "seed": "Dots: 3 seeds; bars: mean ± descriptive SD",
        "core": "Tissue core",
        "delta_f1": "Paired change in F1",
        "weak": "Weak-positive recall",
        "contact": "Contact-bright proxy: false-positive rate",
        "background": "High local reference: false-positive rate",
        "risk_note": "Image-feature strata are proxies, not confirmed contamination",
        "missing": "Missing fraction",
        "feature_name": "Feature",
        "disagreement": "Mean member-score SD",
        "disagreement_note": "Submodel disagreement is not calibrated uncertainty",
        "training_WSIs": "Training",
        "internal_val_WSIs": "Development validation",
        "external_val_TMA_cores": "External test",
        "f1_gain": "C − A",
        "f1_add": "C − B",
    },
    "zh": {
        "fixed_0.5": "固定门限 0.5",
        "validation_selected": "验证集选择门限",
        "precision": "查准率",
        "recall": "查全率",
        "f1": "F1",
        "feature": "特征组合",
        "external": "外部测试",
        "seed": "点：3 个随机种子；误差线：均值 ± 描述性标准差",
        "core": "组织芯",
        "delta_f1": "配对 F1 变化",
        "weak": "弱阳性查全率",
        "contact": "接触侧亮信号代理：假阳性率",
        "background": "高局部参照：假阳性率",
        "risk_note": "图像特征分层是代理指标，不能确认邻居污染",
        "missing": "缺失比例",
        "feature_name": "特征",
        "disagreement": "子模型评分标准差的均值",
        "disagreement_note": "子模型分歧不等同于已校准的不确定性",
        "training_WSIs": "训练集",
        "internal_val_WSIs": "开发验证集",
        "external_val_TMA_cores": "外部测试集",
        "f1_gain": "C − A",
        "f1_add": "C − B",
    },
}


def configure_fonts(font_path: Path | None) -> None:
    if font_path:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        font_manager.fontManager.addfont(str(font_path))
    names = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "DengXian",
        "Source Han Sans SC",
    ]
    chosen = next((name for name in candidates if name in names), None)
    if chosen is None and font_path:
        chosen = font_manager.FontProperties(fname=str(font_path)).get_name()
    if chosen is None:
        raise RuntimeError(
            "Chinese font unavailable. Supply --font-path to a CJK font; no broken-glyph Chinese export will be generated."
        )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out: Path, name: str, lang: str) -> None:
    folder = out / lang
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def decorate(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def mean_seed_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    lang: str,
    baseline: pd.DataFrame | None = None,
) -> None:
    """Paired feature-set order, visible seed points, descriptive SD only."""
    for family, offset in (("TabM", -0.13), ("HGB", 0.13)):
        for j, fset in enumerate(("A", "B", "C")):
            values = (
                frame.loc[
                    frame.model_family.eq(family) & frame.feature_set.eq(fset), metric
                ]
                .dropna()
                .to_numpy(float)
            )
            if not len(values):
                continue
            x = j + offset
            sd = values.std(ddof=1) if len(values) > 1 else 0
            ax.errorbar(
                x,
                values.mean(),
                yerr=sd,
                fmt="o",
                markersize=4,
                color=FAMILY_COLORS[family],
                capsize=2,
                linewidth=0.9,
                label=family if j == 0 else None,
                zorder=4,
            )
            ax.scatter(
                x + np.linspace(-0.035, 0.035, len(values)),
                values,
                s=7,
                color=FAMILY_COLORS[family],
                alpha=0.6,
                zorder=5,
            )
    if baseline is not None and len(baseline) and baseline[metric].notna().any():
        ax.axhline(
            float(baseline[metric].dropna().iloc[0]),
            color=FAMILY_COLORS["GMM"],
            linestyle="--",
            linewidth=0.8,
            label="GMM",
            zorder=2,
        )
    ax.set_xticks(range(3), ["A", "B", "C"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel(TEXT[lang]["feature"])
    decorate(ax)


def performance(metrics: pd.DataFrame, out: Path) -> None:
    source = metrics.loc[metrics.split.eq(TEST) & metrics.group_type.eq("split")].copy()
    source["normalization_method"] = "E1_outer_P1_P99"
    if "model_family" in source:
        source.loc[
            source.model_family.eq("GMM"), "normalization_method"
        ] = "original_unsupervised_GMM"
    source.to_csv(out / "source_data" / "performance.csv", index=False)
    baseline = source.loc[source.model_family.eq("GMM")]
    for lang in ("zh", "en"):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True)
        for row, mode in enumerate(MODES):
            frame = source.loc[source.threshold_mode.eq(mode)]
            for col, metric in enumerate(("precision", "recall", "f1")):
                ax = axes[row, col]
                mean_seed_panel(ax, frame, metric, lang, baseline)
                ax.set_ylim(0, 1.03)
                ax.set_ylabel(TEXT[lang][metric])
                ax.set_title(TEXT[lang][mode])
                ax.text(
                    -0.2,
                    1.07,
                    chr(97 + row * 3 + col),
                    transform=ax.transAxes,
                    fontweight="bold",
                    fontsize=10,
                )
        axes[0, 0].legend(loc="lower left", fontsize=6)
        fig.suptitle(
            TEXT[lang]["external"] + " — " + TEXT[lang]["seed"], fontsize=8.5, y=1.01
        )
        fig.tight_layout(w_pad=1.8, h_pad=2)
        save_figure(fig, out, "performance_ablation", lang)


def paired_cores(paired: pd.DataFrame, out: Path) -> None:
    names = [
        f"{family}_{second}_minus_{first}"
        for family in ("TabM", "HGB")
        for second, first in (("C", "A"), ("C", "B"))
    ]
    source = paired.loc[
        paired.split.eq(TEST)
        & paired.group_type.eq("image")
        & paired.comparison.isin(names)
    ].copy()
    if set(source.context.unique()) != set(CONTEXTS):
        raise ValueError(
            f"Paired-core plot requires contexts {CONTEXTS}; observed {sorted(source.context.unique())}"
        )
    source["normalization_method"] = "E1_outer_P1_P99"
    if "model_family" in source:
        source.loc[
            source.model_family.eq("GMM"), "normalization_method"
        ] = "original_unsupervised_GMM"
    source.to_csv(out / "source_data" / "paired_cores.csv", index=False)
    for mode in MODES:
        subset = source.loc[source.threshold_mode.eq(mode)]
        for lang in ("zh", "en"):
            fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True)
            for row, (second, first) in enumerate((("C", "A"), ("C", "B"))):
                for col, context in enumerate(CONTEXTS):
                    ax = axes[row, col]
                    frame = subset.loc[subset.context.eq(context)]
                    if frame.empty:
                        raise ValueError(f"No paired-core rows for {mode}/{context}")
                    cores = sorted(frame.group_id.unique())
                    for family, offset in (("TabM", -0.15), ("HGB", 0.15)):
                        selected = frame.loc[
                            frame.comparison.eq(f"{family}_{second}_minus_{first}")
                        ]
                        if selected.empty or not selected.delta_f1.notna().any():
                            raise ValueError(
                                f"No finite paired-core data for {mode}/{context}/{family}/{second}-{first}"
                            )
                        means = (
                            selected.groupby("group_id").delta_f1.mean().reindex(cores)
                        )
                        ax.scatter(
                            np.arange(len(cores)) + offset,
                            means,
                            color=FAMILY_COLORS[family],
                            s=15,
                            label=family,
                            zorder=4,
                        )
                        for i, core in enumerate(cores):
                            values = (
                                selected.loc[selected.group_id.eq(core), "delta_f1"]
                                .dropna()
                                .to_numpy()
                            )
                            ax.scatter(
                                i + offset + np.linspace(-0.04, 0.04, len(values)),
                                values,
                                color=FAMILY_COLORS[family],
                                alpha=0.3,
                                s=5,
                                zorder=3,
                            )
                    ax.axhline(0, color="#666666", linewidth=0.65)
                    ax.set_title(
                        context.split("_", 1)[0] + " · " + second + " − " + first
                    )
                    ax.set_xticks(np.arange(len(cores)))
                    ax.set_xticklabels(
                        [str(core).split("/")[-1] for core in cores],
                        rotation=90,
                        fontsize=5.5,
                    )
                    ax.set_xlabel(TEXT[lang]["core"])
                    if col == 0:
                        ax.set_ylabel(TEXT[lang]["delta_f1"])
                    decorate(ax)
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(TEXT[lang][mode], fontsize=9, y=1.01)
            fig.tight_layout(w_pad=1.1, h_pad=2)
            save_figure(fig, out, "paired_core_changes_" + mode, lang)


def error_strata(risk: pd.DataFrame, out: Path) -> None:
    source = risk.loc[risk.split.eq(TEST) & risk.group_type.eq("split")].copy()
    source["false_positive_rate"] = source["fp"] / source["n_true_negative"].replace(
        0, np.nan
    )
    source["normalization_method"] = "E1_outer_P1_P99"
    if "model_family" in source:
        source.loc[
            source.model_family.eq("GMM"), "normalization_method"
        ] = "original_unsupervised_GMM"
    source.to_csv(out / "source_data" / "risk_strata.csv", index=False)
    panels = [
        ("weak_positive", "recall", "weak"),
        ("contact_bright_proxy_negative", "false_positive_rate", "contact"),
        ("high_local_reference_proxy_negative", "false_positive_rate", "background"),
    ]
    for lang in ("zh", "en"):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True)
        for row, mode in enumerate(MODES):
            for col, (stratum, metric, title) in enumerate(panels):
                ax = axes[row, col]
                part = source.loc[source.stratum.eq(stratum)]
                frame = part.loc[part.threshold_mode.eq(mode)]
                mean_seed_panel(
                    ax, frame, metric, lang, part.loc[part.model_family.eq("GMM")]
                )
                ax.set_ylim(0, 1.03)
                ax.set_title(TEXT[lang][title], fontsize=7)
                if col == 0:
                    ax.set_ylabel(TEXT[lang][mode])
        axes[0, 0].legend(fontsize=6, loc="lower left")
        fig.suptitle(
            TEXT[lang]["risk_note"] + "\n" + TEXT[lang]["seed"], fontsize=8, y=1.04
        )
        fig.tight_layout(w_pad=1.5, h_pad=2)
        save_figure(fig, out, "error_strata", lang)


def missingness(missing: pd.DataFrame, out: Path) -> None:
    # Weight by observed cell counts; do not average differently sized cores blindly.
    source = missing.groupby(["split", "feature"], as_index=False)[
        ["n_cells", "n_missing"]
    ].sum()
    source["missing_fraction"] = source.n_missing / source.n_cells
    source["normalization_method"] = "E1_outer_P1_P99"
    if "model_family" in source:
        source.loc[
            source.model_family.eq("GMM"), "normalization_method"
        ] = "original_unsupervised_GMM"
    source.to_csv(out / "source_data" / "missingness.csv", index=False)
    order = (
        source.groupby("feature").missing_fraction.max().sort_values(ascending=False)
    )
    order = order.loc[order > 0].head(20).index.tolist()
    if not order:
        return
    splits = ["training_WSIs", "internal_val_WSIs", TEST]
    matrix = source.pivot(
        index="feature", columns="split", values="missing_fraction"
    ).reindex(index=order, columns=splits)
    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(6.3, max(3.0, 0.16 * len(order) + 1.2)))
        im = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(order)), order, fontsize=6)
        ax.set_xticks(range(len(splits)), [TEXT[lang][s] for s in splits], fontsize=7)
        for i in range(len(order)):
            for j in range(len(splits)):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.1%}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > 0.55 else "black",
                    )
        fig.colorbar(im, ax=ax, shrink=0.7, label=TEXT[lang]["missing"])
        ax.set_title(
            TEXT[lang]["missing"]
            + " — "
            + (
                "最多 20 项非零缺失特征"
                if lang == "zh"
                else "Up to 20 features with missing values"
            )
        )
        fig.tight_layout()
        save_figure(fig, out, "feature_missingness", lang)


def disagreement(data: pd.DataFrame, out: Path) -> None:
    source = data.loc[
        data.split.eq(TEST)
        & data.group_type.eq("image")
        & data.true_class.astype(str).eq("all")
    ].copy()
    if set(source.context.unique()) != set(CONTEXTS):
        raise ValueError(
            f"Disagreement plot requires contexts {CONTEXTS}; observed {sorted(source.context.unique())}"
        )
    source["normalization_method"] = "E1_outer_P1_P99"
    if "model_family" in source:
        source.loc[
            source.model_family.eq("GMM"), "normalization_method"
        ] = "original_unsupervised_GMM"
    source.to_csv(out / "source_data" / "submodel_disagreement.csv", index=False)
    for lang in ("zh", "en"):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
        for ax, context in zip(axes, CONTEXTS):
            frame = source.loc[source.context.eq(context)]
            if frame.empty:
                raise ValueError(f"No disagreement rows for {context}")
            for j, fset in enumerate(("A", "B", "C")):
                values = (
                    frame.loc[frame.feature_set.eq(fset)]
                    .groupby("group_id")
                    .std_mean.mean()
                    .dropna()
                    .to_numpy()
                )
                if not len(values):
                    raise ValueError(
                        f"No finite disagreement data for {context}/{fset}"
                    )
                if len(values):
                    ax.scatter(
                        j + np.linspace(-0.15, 0.15, len(values)),
                        values,
                        s=12,
                        alpha=0.65,
                        color=FAMILY_COLORS["TabM"],
                    )
                    ax.plot(
                        [j - 0.18, j + 0.18],
                        [values.mean(), values.mean()],
                        color="#333333",
                        linewidth=1,
                    )
            ax.set_xticks(range(3), ["A", "B", "C"])
            ax.set_xlabel(TEXT[lang]["feature"])
            ax.set_title(context.split("_", 1)[0])
            decorate(ax)
        axes[0].set_ylabel(TEXT[lang]["disagreement"])
        fig.suptitle(TEXT[lang]["disagreement_note"], fontsize=8, y=1.02)
        fig.tight_layout(w_pad=1.2)
        save_figure(fig, out, "submodel_disagreement", lang)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not (root / "FROZEN_TEST_PROTOCOL.json").is_file():
        raise FileNotFoundError("Missing frozen test protocol")
    configure_fonts(args.font_path)
    out, results = root / "figures", root / "results"
    (out / "source_data").mkdir(parents=True, exist_ok=True)
    contract = {
        "claim": "Describe whether added morphology/spatial features and TabM versus matched HGB improve external classification, without presupposing a winner.",
        "archetype": "quantitative grid",
        "backend": "Python matplotlib",
        "panels": [
            "external precision/recall/F1",
            "paired core F1 changes",
            "weak-positive and neighbour/background proxy error strata",
            "raw feature missingness",
            "uncalibrated submodel disagreement",
        ],
        "exports": [
            "Chinese PNG",
            "Chinese editable SVG",
            "English PNG",
            "English editable SVG",
            "CSV source data",
        ],
        "error_bars": "between-seed sample SD (n=3), descriptive only",
        "review_risks": [
            "development-selected thresholds are not independent validation evidence",
            "cores are not assumed independent patients",
            "feature proxies do not establish contamination",
            "member disagreement is not calibrated uncertainty",
        ],
    }
    (out / "FIGURE_CONTRACT.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    performance(pd.read_csv(results / "metrics_all.csv"), out)
    paired_cores(pd.read_csv(results / "paired_comparisons.csv"), out)
    error_strata(pd.read_csv(results / "risk_strata_metrics.csv"), out)
    missingness(pd.read_csv(results / "feature_missingness_by_unit_class.csv"), out)
    disagreement(pd.read_csv(results / "submodel_disagreement.csv"), out)
    print(f"Wrote figures and plotting source tables to {out}", flush=True)


if __name__ == "__main__":
    main()
