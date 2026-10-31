"""Source-linked data-quality context for the robust CD8 comparison report.

Figure contract: cohort differences in pre-existing zero pixels and spatial-feature
availability must be visible without presenting them as classification outcomes.
The two-panel quantitative figure uses population descriptions, no inference tests
or uncertainty bars. It exports Chinese/English PNG and editable-text SVG at
a nominal 183 x 88 mm canvas with tight export. Pixel and cell denominators include unlabeled
instances. This module reads only two small CSVs, never images or Parquet.

The caller configures Matplotlib's font before generate(). Importing this module
produces no artifacts; final report generation controls when it is called.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = ["training_WSIs", "internal_val_WSIs", "external_val_TMA_cores"]
FEATURES = ["high_signal_edge_coverage", "edge_signal_enrichment"]
CLIPPING_TABLE = "DATA_QUALITY_CLIPPING.csv"
MISSINGNESS_TABLE = "DATA_QUALITY_MISSINGNESS.csv"
PLOT_TABLE = "DATA_QUALITY_PLOT_DATA.csv"
COLORS = ["#6D9DC5", "#B69CC4"]


def source_paths(shared: Path) -> list[Path]:
    """Return every data input for the caller's report-source hash manifest."""
    shared = Path(shared)
    return [
        shared / "calibration/clipping_summary.csv",
        shared / "feature_missingness_highlights.csv",
    ]


def _read_inputs(shared):
    clipping_path, missingness_path = source_paths(shared)
    clip_all = pd.read_csv(clipping_path, float_precision="round_trip")
    miss_all = pd.read_csv(missingness_path, float_precision="round_trip")
    clip = clip_all.loc[clip_all.aggregation.eq("split")].copy()
    missing = miss_all.loc[
        miss_all.aggregation.eq("split") & miss_all.feature.isin(FEATURES)
    ].copy()
    if (
        len(clip) != 3
        or clip.split.duplicated().any()
        or set(clip.split) != set(SPLITS)
    ):
        raise ValueError("Expected exactly one clipping row per predefined split")
    if (
        len(missing) != 6
        or missing.duplicated(["split", "feature"]).any()
        or set(zip(missing.split, missing.feature))
        != {(split, feature) for split in SPLITS for feature in FEATURES}
    ):
        raise ValueError("Expected both missingness features for every split")
    clip = clip.set_index("split").loc[SPLITS]
    missing = missing.set_index(["split", "feature"])
    for split in SPLITS:
        row = clip.loc[split]
        if row.finite_instance_pixels <= 0 or row.rows <= 0:
            raise ValueError("Empty population denominator")
        if not np.isclose(
            row.at_zero_percent,
            100 * row.at_zero_pixel_count / row.finite_instance_pixels,
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Zero-pixel fraction disagrees with its count")
        for feature in FEATURES:
            item = missing.loc[(split, feature)]
            if item.rows != row.rows:
                raise ValueError(
                    "Pixel/cell source tables describe different cell populations"
                )
            if (
                item.new_missing - item.old_missing
                != item.newly_missing - item.recovered
            ):
                raise ValueError("Paired missingness transitions do not reconcile")
            if not np.isclose(
                item.new_missing_percent,
                100 * item.new_missing / item.rows,
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Missingness fraction disagrees with its count")
    values = list(clip.at_zero_percent) + list(missing.new_missing_percent)
    if not all(np.isfinite(x) and 0 <= x <= 100 for x in values):
        raise ValueError("Invalid figure percentage")
    return clip, missing


def _plot_data(clip, missing):
    rows = []
    for split in SPLITS:
        item = clip.loc[split]
        rows.append(
            dict(
                panel="a",
                split=split,
                metric="zero_instance_pixels",
                value_percent=float(item.at_zero_percent),
                numerator=int(item.at_zero_pixel_count),
                denominator=int(item.finite_instance_pixels),
                denominator_type="finite instance pixels",
                source_csv=CLIPPING_TABLE,
                source_filter="aggregation=split; split=" + split,
                source_value_column="at_zero_percent",
            )
        )
        for feature in FEATURES:
            item = missing.loc[(split, feature)]
            rows.append(
                dict(
                    panel="b",
                    split=split,
                    metric=feature,
                    value_percent=float(item.new_missing_percent),
                    numerator=int(item.new_missing),
                    denominator=int(item.rows),
                    denominator_type="all instances including unlabeled",
                    source_csv=MISSINGNESS_TABLE,
                    source_filter="aggregation=split; split="
                    + split
                    + "; feature="
                    + feature,
                    source_value_column="new_missing_percent",
                )
            )
    return pd.DataFrame(rows)


def _draw(clip, missing, figure_directory, language):
    zh = language == "zh"
    split_labels = (
        ["训练 WSI", "内部 WSI", "外部 TMA"]
        if zh
        else ["Training WSI", "Internal WSI", "External TMA"]
    )
    feature_labels = (
        ["边缘高信号覆盖率", "面积校正的边缘富集"]
        if zh
        else ["High-signal edge coverage", "Area-corrected edge enrichment"]
    )
    x = np.arange(3)
    # Respect the caller's registered font and font size. Text remains editable.
    with mpl.rc_context(
        {
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
        bars = axes[0].bar(
            x, clip.at_zero_percent.to_numpy(), width=0.55, color=COLORS[0], linewidth=0
        )
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        width = 0.32
        for index, (feature, label) in enumerate(zip(FEATURES, feature_labels)):
            values = [
                float(missing.loc[(split, feature), "new_missing_percent"])
                for split in SPLITS
            ]
            bars = axes[1].bar(
                x + (index - 0.5) * width,
                values,
                width=width,
                color=COLORS[index],
                label=label,
                linewidth=0,
            )
            for bar in bars:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        for index, ax in enumerate(axes):
            ax.set_xticks(x, split_labels)
            ax.set_ylim(0, 110)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                -0.17,
                1.06,
                "ab"[index],
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=10,
                ha="left",
                va="baseline",
            )
        axes[0].set_title("已有零值像素" if zh else "Pre-existing zero-valued pixels", pad=12)
        axes[1].set_title("空间特征缺失" if zh else "Missing spatial features", pad=12)
        axes[0].set_ylabel("实例像素中零值比例（%）" if zh else "Zero-valued instance pixels (%)")
        axes[1].set_ylabel("缺失的细胞（%）" if zh else "Cells with missing features (%)")
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=False,
            fontsize=7,
            borderaxespad=0,
            labelspacing=0.5,
        )
        footer = (
            "全部实例，含未标注；a 按有效实例像素计，b 按细胞计。"
            if zh
            else "All instances, including unlabeled cells. a: finite instance pixels; b: cells."
        )
        caution = (
            "外部零值为原有像素；缺失不代表阴性。"
            if zh
            else "External zeros were already present; missingness does not imply negativity."
        )
        fig.text(0.08, 0.105, footer, ha="left", va="bottom", fontsize=7)
        fig.text(0.08, 0.052, caution, ha="left", va="bottom", fontsize=7)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.84, wspace=0.42)
        basename = figure_directory / ("data_quality_" + language)
        fig.savefig(
            str(basename) + ".png", dpi=300, bbox_inches="tight", facecolor="white"
        )
        fig.savefig(str(basename) + ".svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def generate(shared: Path, run: Path) -> list[str]:
    """Copy tiny evidence tables, plot both languages, and return Chinese MD blocks.

    The caller owns its completion gate, font setup and output hash manifest.
    Existing data-quality artifacts are never overwritten. No model fitting or
    additional data extraction takes place here.
    """
    shared, run = Path(shared), Path(run)
    clip, missing = _read_inputs(shared)
    external = clip.loc["external_val_TMA_cores"]
    zero_is_original = (
        str(external.all_L_equal_zero).lower() == "true"
        and int(external.below_L_pixel_count) == 0
    )
    if not zero_is_original:
        raise ValueError(
            "Pre-existing external-zero figure claim no longer matches source evidence"
        )
    tables, figures = run / "report_tables", run / "figures"
    targets = [tables / CLIPPING_TABLE, tables / MISSINGNESS_TABLE, tables / PLOT_TABLE]
    targets += [
        figures / f"data_quality_{lang}.{ext}"
        for lang in ["zh", "en"]
        for ext in ["png", "svg"]
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "Data-quality output exists; archive it before regeneration"
        )
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(source_paths(shared), targets[:2]):
        shutil.copyfile(source, destination)
    _plot_data(clip, missing).to_csv(tables / PLOT_TABLE, index=False)
    for language in ["zh", "en"]:
        _draw(clip, missing, figures, language)
    coverage = missing.loc[("external_val_TMA_cores", FEATURES[0])]
    enrichment = missing.loc[("external_val_TMA_cores", FEATURES[1])]
    mostly_existing = all(
        item.newly_missing < 0.01 * item.new_missing for item in [coverage, enrichment]
    )
    conclusion = (
        "新增缺失仅占很小部分，高缺失主要继承原数据及固定参照规则。" if mostly_existing else "新增缺失比例需与原有缺失分开解释。"
    )
    return [
        "![共同校准的数据质量背景](figures/data_quality_zh.png)",
        "图中统计全部实例，包含未标注细胞。图 a 的分母是去重且有效的实例掩码内有限像素；"
        "图 b 的分母是细胞数，均不同于正式分类指标的有标签细胞分母。"
        f"绘图数值、计数、分母及逐行筛选条件见 `report_tables/{PLOT_TABLE}`。",
        "训练、内部、外部的零像素比例依次为 "
        + "、".join(
            f'{float(clip.loc[split, "at_zero_percent"]):.4f}%' for split in SPLITS
        )
        + f"。来源：`report_tables/{CLIPPING_TABLE}`，筛选 `aggregation=split` 和对应 `split`，"
        "列 `at_zero_percent,finite_instance_pixels`。外部 L 均为 0，严格低于 L 的像素数为 0"
        "（同表外部行 `all_L_equal_zero,below_L_pixel_count`），所以高零值比例原本就存在；"
        "它不是本轮下界裁剪新造的零值，也不是细胞阴性率。",
        f"外部覆盖率缺失由 {coverage.old_missing_percent:.4f}% 变为 "
        f"{coverage.new_missing_percent:.4f}%（新增 {int(coverage.newly_missing)}，恢复 "
        f"{int(coverage.recovered)}）；富集缺失由 {enrichment.old_missing_percent:.4f}% "
        f"变为 {enrichment.new_missing_percent:.4f}%（新增 {int(enrichment.newly_missing)}，"
        f"恢复 {int(enrichment.recovered)}）。{conclusion}"
        f"来源：`report_tables/{MISSINGNESS_TABLE}`，筛选 `aggregation=split, "
        "split=external_val_TMA_cores` 及 `feature=high_signal_edge_coverage/edge_signal_enrichment`，"
        "列 `old_missing_percent,new_missing_percent,newly_missing,recovered`。",
        "在图 b 的两项特征中，参照 MAD=0（或不高于数值 epsilon）只直接限制高信号覆盖率；"
        "边缘富集不因 MAD=0 自动缺失，它还要求足够的参照、edge/inner 支持及校正总信号。"
        "NaN 不代表阴性，也不能单凭这些代理确认邻居污染。",
    ]
