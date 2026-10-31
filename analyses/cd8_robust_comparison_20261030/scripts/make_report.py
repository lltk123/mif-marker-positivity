"""Create source-linked Chinese results and bilingual figures after completed runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


METHODS = [
    ("TabM", "A"),
    ("TabM", "B"),
    ("TabM", "C"),
    ("Nimbus", "pretrained"),
    ("Nimbus", "finetuned"),
]
SPLITS = ["internal_val_WSIs", "external_val_TMA_cores"]
FAMILY_COLORS = {"TMA1_CD8": "#6D9DC5", "TMA2_CD8": "#B69CC4", "TMA3_CD8": "#D9AB7E"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(
        path, float_precision="round_trip", keep_default_na=False, na_values=[""]
    )


def label(family, feature, lang="en"):
    if family == "GMM":
        return "GMM"
    if family in ("TabM", "HGB"):
        return f"{family} {feature}"
    suffix = (
        {"pretrained": "预训练", "finetuned": "微调"}
        if lang == "zh"
        else {"pretrained": "pretrained", "finetuned": "finetuned"}
    )[feature]
    return "Nimbus " + suffix


def configure(font_path):
    family = "DejaVu Sans"
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "savefig.facecolor": "white",
        }
    )


def save(fig, path):
    fig.savefig(str(path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(path) + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_performance(summary, directory, lang):
    """Old/new points, seed SD, and the unchanged GMM reference on a shared scale."""
    selected = summary[
        (summary.level == "split") & (summary.threshold_mode == "fixed_0.5")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 5.5), sharex=True, sharey=True)
    names = [label(*m, lang) for m in METHODS]
    for row, split in enumerate(SPLITS):
        data = selected[selected.split == split]
        for col, metric in enumerate(["f1", "precision", "recall"]):
            ax = axes[row, col]
            for y, (family, feature) in enumerate(METHODS):
                points = data[
                    (data.model_family == family) & (data.feature_set == feature)
                ]
                positions = {}
                for variant, color, marker, offset in [
                    ("old", "#8B8B8B", "o", -0.09),
                    ("new", "#307FA2", "s", 0.09),
                ]:
                    item = points[points.variant == variant]
                    if len(item) != 1:
                        raise ValueError("Missing/duplicate plot row")
                    value = float(item.iloc[0][metric + "_mean"])
                    sd = float(item.iloc[0][metric + "_std"])
                    positions[variant] = value
                    if np.isfinite(value):
                        ax.errorbar(
                            value,
                            y + offset,
                            xerr=sd if np.isfinite(sd) else None,
                            color=color,
                            marker=marker,
                            ms=4,
                            lw=0.8,
                            capsize=2,
                        )
                if all(np.isfinite(x) for x in positions.values()):
                    ax.plot(
                        [positions["old"], positions["new"]],
                        [y - 0.09, y + 0.09],
                        color="#BBBBBB",
                        lw=0.7,
                        zorder=0,
                    )
            gmm = data[data.model_family == "GMM"]
            if len(gmm) != 1:
                raise ValueError("Expected one unchanged GMM plot reference")
            ax.axvline(gmm.iloc[0][metric + "_mean"], color="black", lw=0.85, ls="--")
            ax.set_xlim(0, 1)
            ax.set_yticks(range(len(names)), names)
            ax.set_ylim(len(names) - 0.5, -0.5)
            ax.grid(axis="x", color="#E8E8E8", lw=0.5)
            metric_name = {
                "f1": "F1",
                "precision": "查准率" if lang == "zh" else "Precision",
                "recall": "查全率" if lang == "zh" else "Recall",
            }[metric]
            split_name = (
                ("内部 WSI", "外部 TMA")
                if lang == "zh"
                else ("Internal WSI", "External TMA")
            )[row]
            ax.set_title(split_name + " · " + metric_name)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="#8B8B8B",
            ls="",
            label="原流程" if lang == "zh" else "Previous",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="#307FA2",
            ls="",
            label="外层 P1/P99" if lang == "zh" else "Outer P1/P99",
        ),
        Line2D(
            [],
            [],
            color="black",
            ls="--",
            label="原 GMM" if lang == "zh" else "Original GMM",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, directory / ("performance_fixed_0_5_" + lang))


def plot_cores(paired, directory, lang):
    data = paired[
        (paired.level == "image")
        & (paired.split == "external_val_TMA_cores")
        & (paired.threshold_mode == "fixed_0.5")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 6.0), sharex=True, sharey=True)
    for ax, (family, feature) in zip(axes.flat, METHODS):
        group = data[(data.model_family == family) & (data.feature_set == feature)]
        if len(group) != 42:
            raise ValueError("Each core plot requires all 42 matched cores")
        for context, sub in group.groupby("context"):
            ax.scatter(
                sub.f1_mean_old,
                sub.f1_mean_new,
                s=15,
                c=FAMILY_COLORS[context],
                label=context.replace("_CD8", ""),
                alpha=0.8,
            )
        valid = np.isfinite(group.f1_mean_old) & np.isfinite(group.f1_mean_new)
        support = (
            "有效配对" if lang == "zh" else "Valid pairs"
        ) + f": {int(valid.sum())}/42"
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=0.7)
        ax.set(
            xlim=(-0.025, 1.025),
            ylim=(-0.025, 1.025),
            title=label(family, feature, lang) + "\n" + support,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("原流程 F1" if lang == "zh" else "Previous F1")
        ax.set_ylabel("外层 P1/P99 F1" if lang == "zh" else "Outer P1/P99 F1")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.tight_layout()
    save(fig, directory / ("paired_cores_fixed_0_5_" + lang))


def plot_calibration(calibration, directory, lang):
    data = calibration.sort_values(["split", "context", "image_id"]).reset_index(
        drop=True
    )
    x = np.arange(len(data))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axes[0].plot(x, data.U, "o", ms=3, color="#307FA2", label="U = P99(P90)")
    axes[0].plot(
        x,
        data.raw_cell_p90_max,
        "o",
        ms=3,
        mfc="none",
        color="#999999",
        label="原 E1 U = max(P90)" if lang == "zh" else "Original E1 U = max(P90)",
    )
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].set_ylabel(
        "原始 CD8 强度（对称对数轴）" if lang == "zh" else "Raw CD8 intensity (symlog axis)"
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(
        x,
        100 * data.fraction_below_L,
        "o",
        ms=3,
        color="#9B8CAE",
        label="低于 L" if lang == "zh" else "Below L",
    )
    axes[1].plot(
        x,
        100 * data.fraction_above_U,
        "s",
        ms=3,
        color="#D09D6C",
        label="高于 U" if lang == "zh" else "Above U",
    )
    axes[1].set_ylabel("被截断的实例像素（%）" if lang == "zh" else "Clipped instance pixels (%)")
    axes[1].set_xlabel(
        "校准单位（顺序见绘图源数据）" if lang == "zh" else "Calibration unit (order in source data)"
    )
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, directory / ("calibration_clipping_" + lang))


def markdown_table(data, columns):
    """Small Markdown tables without an optional tabulate dependency."""
    out = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in data.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                value = f"{value:.4f}" if np.isfinite(value) else "NaN"
            values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    shared = args.project_root / "analyses/cd8_robust_calibration_20261030"
    from data_quality_report import source_paths, generate

    quality_sources = source_paths(shared)
    marker = json.loads((run / "results/COMPARISON_COMPLETE.json").read_text())
    for name, digest in marker["output_sha256"].items():
        if sha(run / "results" / name) != digest:
            raise RuntimeError("Comparison source changed: " + name)
    summary = read_csv(run / "results/SUMMARY_METRICS.csv")
    paired = read_csv(run / "results/PAIRED_CHANGES.csv")
    bounds = read_csv(shared / "calibration/unit_bounds.csv")
    clip = read_csv(shared / "calibration/clipping_audit.csv")
    calibration = bounds.merge(
        clip,
        on=["image_id", "split", "context"],
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    calibration = calibration.sort_values(["split", "context", "image_id"]).reset_index(
        drop=True
    )
    calibration.insert(0, "plot_unit_index", np.arange(len(calibration)))
    if len(calibration) != 48:
        raise ValueError("Expected 48 complete calibration units")
    inputs = {
        str(p): sha(p)
        for p in [
            run / "results/COMPARISON_COMPLETE.json",
            shared / "calibration/unit_bounds.csv",
            shared / "calibration/clipping_audit.csv",
            shared / "extraction_manifest.json",
            Path(__file__),
            Path(__file__).with_name("data_quality_report.py"),
            run / "RETRAIN_PROTOCOL.json",
            *quality_sources,
        ]
    }
    done = run / "REPORT_COMPLETE.json"
    if done.exists():
        previous = json.loads(done.read_text())
        if previous["input_sha256"] != inputs:
            raise RuntimeError(
                "Report sources changed; archive report output before regeneration"
            )
        for file, digest in previous["output_sha256"].items():
            if sha(run / file) != digest:
                raise RuntimeError("Report output changed: " + file)
        print("Compatible completed report retained")
        return
    tables, figures, source = (
        run / "report_tables",
        run / "figures",
        run / "figure_source",
    )
    for directory in [tables, figures, source]:
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise RuntimeError("Partial report output exists: " + str(directory))
    configure(args.font_path)
    contract = {
        "question": "Does outer-instance P1/P99 calibration improve the frozen TabM and Nimbus workflows?",
        "claim_policy": "Actual results determine direction; no improvement is presumed",
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "panels": [
            "old/new F1, precision, recall with GMM reference",
            "42 paired core F1 values per model",
            "calibration U and clipped pixel fractions",
            "pre-existing zero pixels and reference-dependent feature missingness",
        ],
        "error_bars": "sample SD across three TabM seeds; none for one-seed Nimbus or GMM",
        "export": "Chinese and English PNG 300 dpi; editable-text SVG; CSV source data",
        "statistics": "descriptive paired comparisons; no cell bootstrap or significance tests",
    }
    (run / "FIGURE_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    selected = summary[(summary.level == "split") & summary.split.isin(SPLITS)].copy()
    selected["display_model"] = [
        label(a, b, "zh") for a, b in zip(selected.model_family, selected.feature_set)
    ]
    for mode, filename in [
        ("fixed_0.5", "MAIN_FIXED_0_5.csv"),
        ("validation_selected", "VALIDATION_THRESHOLD.csv"),
    ]:
        selected[selected.threshold_mode == mode].to_csv(tables / filename, index=False)
    paired[(paired.level == "context") & paired.split.isin(SPLITS)].to_csv(
        tables / "PAIRED_CONTEXT_CHANGES.csv", index=False
    )
    calibration.to_csv(source / "calibration_clipping.csv", index=False)
    summary.to_csv(source / "performance.csv", index=False)
    paired.to_csv(source / "paired_cores.csv", index=False)
    for lang in ["zh", "en"]:
        plot_performance(summary, figures, lang)
        plot_cores(paired, figures, lang)
        plot_calibration(calibration, figures, lang)
    quality_blocks = generate(shared, run)

    lines = [
        "# CD8：外层百分位校准后的 TabM 与 Nimbus 重训练",
        "",
        "本报告仅记录本轮已完成的 PhenoBIC CD8 实验。原 GMM、原 TabM 和原 Nimbus 结果直接复用；Pan-M 未读取。",
        "",
        "## 方法与交付状态",
        "",
        "逐实例原始像素先取 P10/P90，再在每个原始 WSI（内部验证去除重复 FOV）或单个 TMA 组织芯内，分别取所有合格实例的 P1/P99 得到 L/U。所有像素共用 `clip((I-L)/(U-L),0,1)`；外层尾部实例仍保留。校准不读取标签。",
        "",
        "TabM A/B/C 分别使用 8/27/90 项特征；本轮没有因全缺失或常量而剔除输入列，实际输入顺序保存在各模型冻结配置中。正式结果为种子 0/1/2 的均值；图中误差线为这三个种子的样本标准差。Nimbus 包含官方 V1 在新尺度下直接推理，以及从 V1 重新微调两个结果，各使用一个种子。匹配 HGB 对照保留在完整 CSV 中。",
        "",
        "固定 `score > 0.5` 是主比较；验证集所选门限单列，未与固定门限混排。训练、验证、外部划分及细胞 ID 保持不变。扩张核 mask 的边缘仍是近似细胞边缘。",
        "内部 WSI 用于早停、学习率和门限选择，内部表展示开发表现；外部数据沿用官方分组作本轮比较。",
        "",
        "GMM 是按图拟合的无监督基线；TabM 和 Nimbus 微调使用原训练 WSI 的人工标签。逐芯若没有真阳性，Recall/AP 为 NaN，单一真值类别时 ROC-AUC 为 NaN；没有预测阳性时 Precision=0，F1 仅在其分母非零时计算。这与两套原评价函数一致。",
        "",
        "## 固定 0.5 门限的性能",
        "",
    ]
    main_table = selected[selected.threshold_mode == "fixed_0.5"]
    for split, title in zip(SPLITS, ["内部 WSI", "外部 TMA"]):
        data = main_table[
            (main_table.split == split)
            & ((main_table.variant == "new") | (main_table.model_family == "GMM"))
        ].copy()
        data = data[data.model_family != "HGB"].sort_values("f1_mean", ascending=False)
        eligible = data.all_seeds_complete.eq(True) & data.f1_mean.notna()
        data["rank"] = "不可评分"
        data.loc[eligible, "rank"] = (
            data.loc[eligible, "f1_mean"]
            .rank(ascending=False, method="min")
            .astype(int)
            .astype(str)
        )
        gmm = data[data.model_family == "GMM"].iloc[0]
        data["delta_f1_gmm"] = data.f1_mean - gmm.f1_mean
        data.to_csv(tables / ("RANK_" + split + ".csv"), index=False)
        lines += [
            "### " + title,
            "",
            markdown_table(
                data,
                [
                    ("rank", "排序"),
                    ("display_model", "模型"),
                    ("precision_mean", "Precision"),
                    ("recall_mean", "Recall"),
                    ("f1_mean", "F1"),
                    ("average_precision_mean", "AP"),
                    ("delta_f1_gmm", "F1 − GMM"),
                ],
            ),
            "",
            f"来源：[MAIN_FIXED_0_5.csv](report_tables/MAIN_FIXED_0_5.csv)，筛选 `split={split}`、`variant=new/baseline`、`model_family≠HGB`；列为 `precision_mean/recall_mean/f1_mean/average_precision_mean`。排序表为 [RANK_{split}.csv](report_tables/RANK_{split}.csv)。",
            "",
        ]
    lines += ["![固定门限性能](figures/performance_fixed_0_5_zh.png)", "", "## 新旧流程的配对变化", ""]
    delta = paired[
        (paired.level == "split")
        & (paired.split == "external_val_TMA_cores")
        & (paired.threshold_mode == "fixed_0.5")
        & (paired.model_family != "HGB")
    ].copy()
    delta["display_model"] = [
        label(a, b, "zh") for a, b in zip(delta.model_family, delta.feature_set)
    ]
    delta.to_csv(tables / "EXTERNAL_PAIRED_CHANGES.csv", index=False)
    lines += [
        markdown_table(
            delta,
            [
                ("display_model", "模型"),
                ("f1_mean_old", "原 F1"),
                ("f1_mean_new", "新 F1"),
                ("f1_delta", "ΔF1"),
                ("average_precision_delta", "ΔAP"),
                ("tp_delta", "ΔTP"),
                ("fp_delta", "ΔFP"),
                ("fn_delta", "ΔFN"),
            ],
        ),
        "",
        "来源：[EXTERNAL_PAIRED_CHANGES.csv](report_tables/EXTERNAL_PAIRED_CHANGES.csv)，外部整体、固定 0.5；正的 ΔF1/ΔAP 表示改善。TP/FP/FN 是种子间平均计数的变化，因此可能为小数，并非合并重复细胞。",
        "",
    ]
    for _, row in delta.iterrows():
        if not np.isfinite(row.f1_delta) or not np.isfinite(
            row.average_precision_delta
        ):
            lines += [
                f"- {row.display_model}：完整集合的 F1/AP 变化不可比较；请检查同表 `prediction_coverage_mean_new/prediction_coverage_mean_old` 及原始失败计数，不将失败子集退出解释为稳定或改善。"
            ]
            continue
        direction = "提高" if row.f1_delta > 0 else "降低" if row.f1_delta < 0 else "不变"
        lines += [
            f"- {row.display_model}：外部 F1 {direction} {abs(row.f1_delta):.4f}，AP 变化 {row.average_precision_delta:+.4f}；依据同表 `f1_delta/average_precision_delta`。"
        ]
    context_data = paired[
        (paired.level == "context")
        & (paired.split == "external_val_TMA_cores")
        & (paired.threshold_mode == "fixed_0.5")
        & (paired.model_family != "HGB")
    ].copy()
    context_data["display_model"] = [
        label(a, b, "zh")
        for a, b in zip(context_data.model_family, context_data.feature_set)
    ]
    lines += [
        "",
        "### 逐 TMA 与组织芯",
        "",
        markdown_table(
            context_data,
            [
                ("context", "TMA"),
                ("display_model", "模型"),
                ("f1_delta", "ΔF1"),
                ("average_precision_delta", "ΔAP"),
                ("recall_delta", "ΔRecall"),
                ("fp_delta", "ΔFP"),
                ("fn_delta", "ΔFN"),
            ],
        ),
        "",
        "[PAIRED_CONTEXT_CHANGES.csv](report_tables/PAIRED_CONTEXT_CHANGES.csv) 按 `context=TMA1_CD8/TMA2_CD8/TMA3_CD8` 和 `threshold_mode` 查看；`average_precision_delta/recall_delta/fp_delta/fn_delta` 分别对应可分性、召回、误报与漏检变化。完整组织芯数据见 [PAIRED_CHANGES.csv](results/PAIRED_CHANGES.csv)，筛选 `level=image`。",
        "",
        "![逐组织芯配对](figures/paired_cores_fixed_0_5_zh.png)",
        "",
        "## 校准与数据完整性",
        "",
    ]
    invalid = int(
        calibration.calibration_invalid.astype(str).str.lower().eq("true").sum()
    )
    pixels = calibration.finite_instance_pixels.to_numpy(float)
    high = float(np.average(calibration.fraction_above_U, weights=pixels))
    low = float(np.average(calibration.fraction_below_L, weights=pixels))
    lines += [
        f"校准共 {len(calibration)} 个单位，失效 {invalid} 个。按唯一实例的有限像素数加权，低于 L 的像素占 {100*low:.4f}%，高于 U 占 {100*high:.4f}%。分母不包含背景，不重复计算重叠 FOV 中的同一实例。来源：[calibration_clipping.csv](figure_source/calibration_clipping.csv)，计数列 `calibration_invalid/finite_instance_pixels`，比例列 `fraction_below_L/fraction_above_U`。",
        "",
        "此处限制的是参与尺度边界估计的极端细胞，而不是删除极端细胞。更稳健的尺度仍可能压缩很亮像素的差异；模型性能的实际方向由上面的配对表判断。",
        "",
        "![校准与截断](figures/calibration_clipping_zh.png)",
        "",
        "完整评分覆盖率见 [MODEL_COMPARISON.csv](results/MODEL_COMPARISON.csv) 的 `n_cells/n_score_finite/n_score_failed/prediction_coverage`；校准有效性另见 `n_calibration_invalid/calibration_coverage`。插补后得到有限分数不等于图像校准有效，两者分别保留。不完整评分组不进入完整集合 F1 排序。局部空间特征的 NaN 保留，并按各模型原定训练预处理处理。",
        "",
        "## 输入稀疏性与特征缺失",
        "",
        *[item for block in quality_blocks for item in (block, "")],
        "## 补充门限、错误类型与复现",
        "",
        "验证集门限结果：[VALIDATION_THRESHOLD.csv](report_tables/VALIDATION_THRESHOLD.csv)，筛选 `split` 和 `model_family/feature_set`；门限与性能列分别为 `threshold_mean`、`f1_mean/precision_mean/recall_mean/average_precision_mean`。各个种子的门限和结果保存在 [MODEL_COMPARISON.csv](results/MODEL_COMPARISON.csv)。",
        "",
        "弱阳性、邻近亮细胞、局部背景等预定错误分层、实例图和资源耗时见 [TabM 报告](../cd8_tabm_robust_20261030/ANALYSIS_REPORT.md) 与 [Nimbus 报告](../cd8_nimbus_robust_20261030/ANALYSIS_REPORT.md)。这些分层沿用原训练集定义。整体 FP 下降不能单独证明邻近污染已解决。",
        "",
        "复现命令与运行顺序见 [README](README.md)。源代码、参数与结果哈希分别由 [RETRAIN_PROTOCOL.json](RETRAIN_PROTOCOL.json)、各模型冻结协议和 `COMPARISON_COMPLETE.json/REPORT_COMPLETE.json` 绑定。",
        "",
        "全量文件保留在服务器：",
        "",
        f"- 归一化逐细胞特征：`{shared}/PhenoBIC_CD8_features.parquet`。",
        f"- TabM 模型、全部子模型预测及逐细胞结果：`{args.project_root}/analyses/cd8_tabm_robust_20261030`。",
        f"- Nimbus 权重及逐细胞结果：`{args.project_root}/analyses/cd8_nimbus_robust_20261030`。",
        "",
        "TabM 三种子和 Nimbus 单种子的计算条件不同，排序是本次实际配置的描述结果；未开展细胞级显著性检验。Nimbus 新尺度流程属于任务适配，不宣称复现整篇论文全部实验。",
    ]
    (run / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = [run / "ANALYSIS_REPORT.md", run / "FIGURE_CONTRACT.json"]
    outputs += [
        p
        for directory in [tables, figures, source]
        for p in directory.iterdir()
        if p.is_file()
    ]
    done.write_text(
        json.dumps(
            {
                "status": "complete",
                "input_sha256": inputs,
                "output_sha256": {str(p.relative_to(run)): sha(p) for p in outputs},
                "plot_languages": ["zh", "en"],
                "svg_text_editable": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("Report and bilingual figures complete")


if __name__ == "__main__":
    main()
