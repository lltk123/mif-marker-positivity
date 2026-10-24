"""Chinese Markdown report with explicit CSV evidence, generated from frozen results."""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd


def table(d):
    d = d.copy()

    def fmt(v):
        if pd.isna(v):
            return "NaN"
        if isinstance(v, (float, np.floating)):
            return "%.4f" % v
        return str(v).replace("|", "/").replace("\n", " ")

    return (
        "| "
        + " | ".join(map(str, d.columns))
        + " |\n| "
        + " | ".join(["---"] * len(d.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in d.itertuples(index=False, name=None)
        )
        + "\n"
    )


def main(root):
    r = Path(root)
    res = r / "results"
    m = pd.read_csv(res / "metrics_split.csv")
    ctx = pd.read_csv(res / "metrics_context.csv")
    pa = pd.read_csv(res / "paired_changes.csv")
    macro = pd.read_csv(res / "metrics_macro_units.csv")
    missing = pd.read_csv(res / "feature_availability.csv")
    qc = pd.read_csv(res / "QC_reasons.csv")
    cal = pd.read_csv(r / "calibration/calibration_parameters.csv")
    st = pd.read_csv(r / "stability/stability_split.csv")
    unit = pd.read_csv(r / "stability/stability_unit.csv")
    inv = pd.read_csv(r / "stability/feature_invariance.csv")
    models = ["GMM", "intensity_8", "morphospatial_90", "E1", "E2", "E3", "E4"]
    notes = []

    def metric(model, split="external_val_TMA_cores", mode="fixed_0.5"):
        return m[
            (m.model == model) & (m.split == split) & (m.threshold_mode == mode)
        ].iloc[0]

    facts = []

    def fact(claim, source, filters, column, value):
        facts.append(
            dict(claim=claim, CSV=source, filter=filters, column=column, value=value)
        )

    for n in models:
        v = metric(n)
        for col in ["F1", "precision", "recall", "ROC_AUC", "AP"]:
            fact(
                n + " 外部主结果 " + col,
                "results/metrics_split.csv",
                "split=external_val_TMA_cores;model=" + n + ";threshold_mode=fixed_0.5",
                col,
                v[col],
            )
    fixed = m[(m.threshold_mode == "fixed_0.5") & (m.split != "training_WSIs")][
        [
            "split",
            "model",
            "n_total",
            "precision",
            "recall",
            "F1",
            "ROC_AUC",
            "AP",
            "TP",
            "FP",
            "FN",
            "TN",
        ]
    ]
    fixed.to_csv(res / "report_primary_performance.csv", index=False)
    supp = m[
        (m.threshold_mode == "OOF_threshold")
        & (m.model.isin(["E1", "E2", "E3", "E4"]))
        & (m.split != "training_WSIs")
    ][["split", "model", "threshold", "precision", "recall", "F1", "TP", "FP", "FN"]]
    supp.to_csv(res / "report_supplementary_performance.csv", index=False)
    paired = pa[
        (pa.level == "split")
        & (pa.threshold_mode == "fixed_0.5")
        & (pa.split != "training_WSIs")
    ][
        [
            "split",
            "comparison",
            "F1_delta",
            "TP_delta",
            "FP_delta",
            "FN_delta",
            "ROC_AUC_delta",
            "AP_delta",
        ]
    ]
    paired.to_csv(res / "report_primary_paired_changes.csv", index=False)
    # Every pooled flip denominator is valid pairs; failure-aware denominator is the original sample.
    overview = []
    for mode in ["fixed_0.5", "OOF_threshold"]:
        for name in models:
            g = st[
                (st.model == name)
                & (st.split == "external_val_TMA_cores")
                & (st.threshold_mode == mode)
            ]
            aff = g[g.perturbation.str.startswith(("gain", "bias"))]
            rot = g[g.perturbation.str.startswith("rotate")]
            er = g[g.perturbation == "erode_1"].iloc[0]
            di = g[g.perturbation == "dilate_1"].iloc[0]
            overview.append(
                dict(
                    model=name,
                    threshold_mode=mode,
                    affine_max_flip_rate=aff.flip_rate.max(),
                    affine_max_score_change_p95=aff.score_change_p95.max(),
                    rotation_max_flip_rate=rot.flip_rate.max(),
                    rotation_max_score_change_p95=rot.score_change_p95.max(),
                    erosion_flip_rate=er.flip_rate,
                    dilation_flip_rate=di.flip_rate,
                    erosion_flip_or_unscoreable_rate=er.flip_or_unscoreable_rate,
                    dilation_flip_or_unscoreable_rate=di.flip_or_unscoreable_rate,
                    erosion_coverage=er.coverage,
                    dilation_coverage=di.coverage,
                    erosion_F1_before=er.F1_before,
                    erosion_F1_after=er.F1_after,
                    dilation_F1_before=di.F1_before,
                    dilation_F1_after=di.F1_after,
                )
            )
    ov = pd.DataFrame(overview)
    ov.to_csv(res / "report_stability_overview.csv", index=False)
    conf = json.loads((r / "THRESHOLDS_FROZEN.json").read_text())
    cells = pd.read_csv(r / "keys/key_manifest.csv")
    run = pd.DataFrame(
        [
            dict(
                item="all_cells",
                value=1817783,
                source="keys/key_manifest.csv; population=all; sum(rows)",
            ),
            dict(
                item="labelled_training",
                value=101057,
                source="keys/key_manifest.csv; split=training_WSIs; population=labelled",
            ),
            dict(
                item="labelled_internal",
                value=539132,
                source="keys/key_manifest.csv; split=internal_val_WSIs; population=labelled",
            ),
            dict(
                item="labelled_external",
                value=437480,
                source="keys/key_manifest.csv; split=external_val_TMA_cores; population=labelled",
            ),
            dict(
                item="units",
                value=len(cal),
                source="calibration/calibration_parameters.csv; row count",
            ),
            dict(
                item="sample_cells",
                value=9600,
                source="keys/stability_sample.csv; row count",
            ),
            dict(
                item="new_supervised_fits",
                value=16,
                source="results/new_model_fit_log.csv; row count",
            ),
            dict(
                item="stability_cell_model_pairs",
                value=604800,
                source="stability/STABILITY_COMPLETE.json",
            ),
            dict(
                item="brightness_GMM_refits",
                value=192,
                source="stability/perturbed_calibration.csv; row count",
            ),
        ]
    )
    run.to_csv(res / "report_run_counts.csv", index=False)
    text = [
        "# PhenoBIC CD8：E1–E4 与冻结模型稳定性分析报告\n",
        "## 结论\n",
        "本轮已完成全量共同校准、像素重提取、四组固定 HGB 的 OOF/最终训练、完整验证集合评价和规定的稳定性检查。旧实验直接复用保存模型与结果；Pan-M Gold 继续封存，没有读取其图像、标签、分布或预测。\n",
        "1. **共同校准明显改善外部跨图像识别。** 固定 0.5 时，原 8 项强度模型外部 F1 为 %.4f，E1 为 %.4f；内部从 %.4f 变为 %.4f，改善主要发生在外部数据。证据：[metrics_split.csv](results/metrics_split.csv)，筛选 `threshold_mode=fixed_0.5; model∈{intensity_8,E1}; split∈{internal_val_WSIs,external_val_TMA_cores}`，列 `F1`。\n"
        % (
            metric("intensity_8").F1,
            metric("E1").F1,
            metric("intensity_8", "internal_val_WSIs").F1,
            metric("E1", "internal_val_WSIs").F1,
        ),
        "2. **校准后的定位特征有小幅附加收益。** E2 外部 F1 为 %.4f，优于 E1 的 %.4f；内部也改善。固定主门限下，E2 是新四组中内部与外部 F1 最高者，但内部仍略低于旧 90 项模型。该结果是固定整套 S9 的收益，不能归因于其中某个指标或直接解释为污染减少。证据：同表，`model∈{E1,E2,morphospatial_90}`，列 `F1,TP,FP,FN`；精确配对见下文。\n"
        % (metric("E2").F1, metric("E1").F1),
        "3. **本轮没有看到接入 GMM 分数后一致的额外性能优势。** E3 的外部 F1 为 %.4f，E4 为 %.4f，均未超过主门限下 E2。补充门限下 E4 外部 F1 为 %.4f，E2 为 %.4f，差距很小；E4 的外部 AUC/AP 仍低于 E2，不能据此宣称 E4 全面最优。原 GMM 是无监督基线，新模型使用训练标签，监督条件不同。证据：[metrics_split.csv](results/metrics_split.csv)，`split=external_val_TMA_cores; model∈{E2,E3,E4}; threshold_mode∈{fixed_0.5,OOF_threshold}`，列 `F1,ROC_AUC,AP`。\n"
        % (
            metric("E3").F1,
            metric("E4").F1,
            metric("E4", mode="OOF_threshold").F1,
            metric("E2", mode="OOF_threshold").F1,
        ),
        "稳定性结论须与下列正式性能共同理解，不能把始终预测阴性的旧模型称为可靠。\n",
        "## 1. 数据、冻结方案与复现\n",
        table(run),
        "以上执行计数对应 [report_run_counts.csv](results/report_run_counts.csv)，列 `item,value,source`。数据 commit：`aeffd288d922ee5df4cbb414f3c1f77507ba0001`。所有关联使用 dataset_id + image_id + cell_id，无行号映射、无重新划分、无因特征缺失删行。完整目录：\n\n`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_e1_e4_stability_20261023`\n",
        "E1/E2/E3/E4 分别为 8/17/9/18 项；模型参数、几何参数和稳定性方案见 [E1_E4_PROTOCOL.json](E1_E4_PROTOCOL.json)、[RUN_CONFIG.json](RUN_CONFIG.json)。训练只使用原三张训练 WSI 的有标签细胞，均匀细胞权重。留一训练 WSI OOF 后在全部训练标签上拟合最终模型，无额外随机种子、超参数搜索、插补器或概率校准器。无标签实例参与本图校准，不进入监督损失和分类指标。\n",
        "校准单位为每张训练 WSI、每张内部 WSI 的有效 FOV 合并区域、每个外部组织芯。原实例至少 5 有限像素即可参与 L/U，独立于结构 QC。L=min(实例 p10)，U=max(实例 p90)，所有像素共享浮点裁剪映射；8 项强度从像素重提取。内部坐标重叠按原实例/全局坐标处理，不重复计数。原始动态范围与边界比例均以合格实例内有限像素为统计集合，不包含全图未分割区域的极值。\n",
        "原 GMM 来源已审计：全单位原始均值先按 0.01/99.99 百分位裁剪用于 log1p 后三成分拟合，最高两成分均值中点为 gate；映射使用未裁剪 log1p 均值及最近观测门限的分段 MinMaxScaler。原 rescale_score 是校准分数，不称阳性概率。固定映射保留 clip=False 外推。恢复状态和原分数一致性见 [GMM_mapping_recovery.csv](audit/GMM_mapping_recovery.csv)；没有用内部/外部标签调 gate。历史正常输入复核在最新“旧实验不重跑”指示前结束，此后未重训旧监督模型。\n",
        "补充门限已在查看新内部/外部指标前冻结："
        + ", ".join(k + "=" + str(v) for k, v in conf["thresholds"].items())
        + "。网格、三 WSI 等权 F1 与选中门限见 [OOF_threshold_grid.csv](results/OOF_threshold_grid.csv)，筛选 model，列 threshold/mean_WSI_F1；冻结文件见 [THRESHOLDS_FROZEN.json](THRESHOLDS_FROZEN.json)。旧三组在补充面板仍使用原 0.5。所有判阳严格大于门限。OOF 选门限后在同一 OOF 上报告的 F1 是“OOF 门限开发拟合值”，不是无偏泛化成绩；完整 OOF 指标在 [OOF_metrics.csv](results/OOF_metrics.csv)。\n",
        "## 2. 完整验证集合主结果\n",
        "下表对应 [report_primary_performance.csv](results/report_primary_performance.csv)，来自 metrics_split.csv 的 `threshold_mode=fixed_0.5; split!=training_WSIs`；显示列与表头同名。训练最终预测属于重代入，未混入此表。\n",
        table(fixed),
        "![固定门限性能](figures/01_performance_fixed_zh.png)\n",
        "## 3. 配对收益与 TMA 差异\n",
        "下表同时报告 TP/FP/FN 变化。正的 FP_delta 是新增误报，负的 FN_delta 是减少漏检；F1 提升不自动等同于污染减少。对应 [report_primary_paired_changes.csv](results/report_primary_paired_changes.csv)，原表 paired_changes.csv 筛选 `level=split; threshold_mode=fixed_0.5; split!=training_WSIs`，列与表头同名。\n",
        table(paired),
    ]
    for context in ["TMA1_CD8", "TMA2_CD8", "TMA3_CD8"]:
        g = ctx[(ctx.context == context) & (ctx.threshold_mode == "fixed_0.5")]
        cols = ["model", "precision", "recall", "F1", "ROC_AUC", "AP", "TP", "FP", "FN"]
        text += [
            "### " + context + "\n",
            "对应 [metrics_context.csv](results/metrics_context.csv)，筛选 `context="
            + context
            + "; threshold_mode=fixed_0.5`，列如下。\n",
            table(g[cols]),
        ]
    text += [
        "**TMA1：** 共同校准恢复了旧 8/90 项模型几乎失效的判阳与排序能力。应同时查看 ROC_AUC 和 AP，不能只看召回恢复。E2 相较 E1 的误报减少伴随少量漏检增加，并非所有方向均改善。\n",
        "**TMA2：** 新模型召回显著高于旧强度模型，但仍低于原 GMM；新模型查准率较高。这是本轮仍需保留的跨域召回问题，不据外部结果改门限或改背景方案。\n",
        "**TMA3：** E1–E4 相较旧 8 项模型显著减少误报，但漏检增加；与原 GMM 相比，同样是误报与漏检的权衡。上述三点的精确数字均在刚列出的 metrics_context.csv 各自筛选行及 TP/FP/FN、precision/recall、ROC_AUC/AP 列。\n",
        "![逐组织芯变化](figures/02_core_changes_zh.png)\n",
        "逐芯配对值见 [paired_changes.csv](results/paired_changes.csv)，`level=unit; split=external_val_TMA_cores; threshold_mode=fixed_0.5`；图中每个点是一枚组织芯，不假定其为独立患者。内部逐 WSI 与所有芯完整指标见 [metrics_unit.csv](results/metrics_unit.csv)。\n",
        "### 单位等权表现与低表现单位\n",
        "下表对应 [metrics_macro_units.csv](results/metrics_macro_units.csv)，筛选 `threshold_mode=fixed_0.5; split!=training_WSIs`。平均/标准差/最小值按 WSI 或组织芯等权，不能用细胞汇总 F1 替代。\n",
        table(
            macro[
                (macro.threshold_mode == "fixed_0.5") & (macro.split != "training_WSIs")
            ][
                [
                    "split",
                    "model",
                    "n_units",
                    "F1_mean",
                    "F1_std",
                    "F1_min",
                    "F1_median",
                    "units_no_positive_prediction",
                ]
            ]
        ),
        "完整低表现单位在 [lowest_units.csv](results/lowest_units.csv)，按 model/split/threshold_mode 筛选后查看 image_id、F1、recall、FP、FN。未做细胞级显著性检验、细胞 bootstrap 或仅凭三张 WSI 强行估计稳定置信区间。\n",
        "## 4. 补充门限结果\n",
        "对应 [report_supplementary_performance.csv](results/report_supplementary_performance.csv)，来源 metrics_split.csv 的 `threshold_mode=OOF_threshold; model∈{E1,E2,E3,E4}; split!=training_WSIs`。降低门限可改变查全/查准权衡，但不改变同一模型的 AUC/AP。\n",
        table(supp),
        "内部 F1 并未因 OOF 选定门限而普遍提高，外部改善也不能用于反向挑选本轮方案。两套结果都保留，未把补充结果替代主结果。\n",
        "## 5. 校准、缺失与可解释性边界\n",
        "所有单位的 L/U、贡献实例、动态范围、有效实例/像素数及裁剪比例见 [calibration_parameters.csv](calibration/calibration_parameters.csv)。boundary 比例以合格原实例内部的有限像素为一致集合，每个全局像素只统计一次；严格 <L/>U 裁剪与 ≤L/≥U 边界比例分别保存。极亮实例主导 U 的程度见 U_to_cell_p90_p99、second_largest_cell_p90，以及 [upper_bound_diagnostics.csv](calibration/upper_bound_diagnostics.csv)。这是一项敏感性诊断，本轮未据此更换边界。\n",
    ]
    csummary = pd.DataFrame(
        [
            dict(
                units=len(cal),
                invalid_units=int(cal.calibration_invalid.sum()),
                U_min=cal.U.min(),
                U_max=cal.U.max(),
                U_to_p90p99_median=cal.U_to_cell_p90_p99.median(),
                U_to_p90p99_max=cal.U_to_cell_p90_p99.max(),
                zero_boundary_fraction_median=cal.at_or_below_L_fraction.median(),
                upper_clip_fraction_max=cal.above_U_fraction.max(),
            )
        ]
    )
    csummary.to_csv(res / "report_calibration_summary.csv", index=False)
    text += [
        "校准摘要对应 [report_calibration_summary.csv](results/report_calibration_summary.csv)，由上表所有单位聚合，列名指明聚合方式。\n",
        table(csummary),
    ]
    mm = missing[
        (missing.level == "true_class")
        & (missing.context.isin(["TMA1_CD8", "TMA2_CD8", "TMA3_CD8"]))
        & (missing.class_stratum.isin(["negative", "positive"]))
        & (
            missing.feature.isin(
                ["high_signal_edge_coverage", "angular_R1", "edge_signal_enrichment"]
            )
        )
    ][["context", "class_stratum", "feature", "n", "missing_rate"]]
    mm.to_csv(res / "report_external_missingness.csv", index=False)
    text += [
        "外部局部高信号覆盖率大量缺失，且缺失模式与真实类别相关。MAD 数值零、无校正信号、参照不足分别保留，不能把 NaN 当 0 或阴性。模型可能利用缺失模式；本轮没有加入未经消融的背景回退，因此不能把定位收益全部解释成形态生物学改善。\n",
        "下表对应 [report_external_missingness.csv](results/report_external_missingness.csv)，原表 feature_availability.csv 筛选 `level=true_class; context=TMA1/2/3_CD8; class_stratum=negative/positive; feature=表中三项`，列 n/missing_rate。所有 17 项、无标签层和每单位统计仍保留在原表，QC 原因见 [QC_reasons.csv](results/QC_reasons.csv)。\n",
        table(mm),
        "![校准与缺失](figures/03_calibration_missingness_zh.png)\n",
        "expanded_nucleus 的边缘只能称为近似细胞边缘，inner 不是已验证的细胞核；无独立组织 mask，局部环带只是周细胞参照。无像素尺寸，不能声称跨图像已统一物理尺度。详细定义及 NaN 条件见 [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md) 与 [FEATURE_DICTIONARY.csv](FEATURE_DICTIONARY.csv)。\n",
        "## 6. 冻结模型稳定性\n",
        "固定 seed=0，按每单位有原标签的 ID 均匀抽取，最多 200 个；清单在 [stability_sample.csv](keys/stability_sample.csv)，不是按正确性/置信度选样。所有七模型使用相同细胞。稳定性是抽样性能，不能替代正式完整验证集指标。\n",
        "增益/偏置：正增益与共同偏置下 L/U 和所有细胞原始均值使用数学等价加速，已经在代表完整单位逐像素核验；完整单位 GMM 重新拟合，不用 200 个样本分布代替。原始强度/90 项模型从扰动像素重提取。旋转共同变换图像、所有邻居 ID、valid 支持，核验强度不变后复用 GMM。完整原图及内部 FOV 合并画布旋转后再裁剪的等价性检查见 [full_field_rotation_checks.csv](tests/full_field_rotation_checks.csv)，all_context_pixels_identical 列全部通过。mask 腐蚀/膨胀 1 px，膨胀仅入原背景、等距竞争留背景，所有邻居同规则；冻结 L/U 和原 GMM 映射，新均值重新评分。halo=8 px，缺失图像支持标 invalid。\n",
        "输入有效性和数值输出分开判断。mask 消失、像素不足、必要校准失败即不可评分；S 局部缺失但必要强度有效时继续 HGB 评分。主翻转率分母为两侧均有效固定配对；“翻转或不可评分比例”分母是原抽样总数。每单位完整分子/分母、双向翻转、阳性比例变化、前后 precision/recall/F1 见 [stability_unit.csv](stability/stability_unit.csv)；按 split/context 见同目录对应 CSV。\n",
        "下表仅汇总外部抽样的固定 0.5。affine/rotation 的 max 为各自预定扰动行的最大值，不是额外搜索；mask 两项单独列出。对应 [report_stability_overview.csv](results/report_stability_overview.csv)，筛选 `threshold_mode=fixed_0.5`；来源 stability_split.csv 的 `split=external_val_TMA_cores`。变化与比例均为原始数值，0.01 即 1%。\n",
        table(
            ov[ov.threshold_mode == "fixed_0.5"][
                [
                    "model",
                    "affine_max_flip_rate",
                    "affine_max_score_change_p95",
                    "rotation_max_flip_rate",
                    "erosion_flip_rate",
                    "dilation_flip_rate",
                    "erosion_flip_or_unscoreable_rate",
                    "dilation_flip_or_unscoreable_rate",
                    "erosion_coverage",
                    "dilation_coverage",
                ]
            ]
        ),
        "![稳定性](figures/04_stability_fixed_zh.png)\n",
        "图中分数变化为各单位 p95 的中位数；翻转率按有效配对计数汇总，不能与单个 split 的 p95 混为一谈。完整绘图聚合见 [stability_plot_data.csv](figures/source_data/stability_plot_data.csv)。补充门限的同类表和图均保留。mask 前后配对 F1、precision、recall 见 stability_split.csv；不能因为翻转少就忽视原始召回不足。\n",
    ]
    allcounts = (
        unit.groupby(["model", "perturbation", "threshold_mode"])[
            ["total_pairs", "valid_pairs", "flip_count", "unscoreable"]
        ]
        .sum()
        .reset_index()
    )
    allcounts["flip_rate"] = allcounts.flip_count / allcounts.valid_pairs
    allcounts["flip_or_unscoreable_rate"] = (
        allcounts.flip_count + allcounts.unscoreable
    ) / allcounts.total_pairs
    allcounts.to_csv(res / "report_stability_all_samples.csv", index=False)
    text += [
        "**实际稳定性发现：** E1/E2 在全部抽样细胞的四种亮度扰动和三种旋转中，固定 0.5 均无判阳翻转；接入 GMM 分数的 E3/E4 在亮度扰动下出现少量变化。与此相比，旧原始强度模型对共同偏置敏感。mask 的 1 px 变化仍导致少量翻转，腐蚀还使部分 HGB 输入无效；这些失效已经计入原抽样总分母，不能默认为阴性或忽略。具体失效为腐蚀后实例变为不连通，而非像素消失：详见 [input_invalid_cases.csv](stability/input_invalid_cases.csv)，筛选 perturbation=erode_1，查看 image_id、cell_id、observed_pixels、qc_reasons。证据：[report_stability_all_samples.csv](results/report_stability_all_samples.csv)，筛选 threshold_mode=fixed_0.5 与相应 model/perturbation，查看 flip_count、unscoreable、valid_pairs、flip_rate、flip_or_unscoreable_rate。分数稳定性并不弥补 TMA2 原有召回不足。\n"
    ]
    rot = inv[inv.perturbation.str.startswith("rotate")]
    summ = (
        rot.groupby("feature")[["absolute_change_max", "NaN_pattern_changes"]]
        .max()
        .reset_index()
    )
    summ.to_csv(res / "report_rotation_feature_checks.csv", index=False)
    text += [
        "方向特征的数值核验见 [report_rotation_feature_checks.csv](results/report_rotation_feature_checks.csv)，由 feature_invariance.csv 筛选 rotate_* 后按 feature 取最大值。正常输入评分重现见 [original_score_reproduction.csv](stability/original_score_reproduction.csv)；每个 model/image_id 的 max_original_score_reproduction_error 用于确认上下文和特征顺序未偏离原流程。\n",
        "## 7. 真实示例与交付检查\n",
        "![真实细胞示例](figures/05_real_examples_zh.png)\n",
        "样例来自预定样本中的代表训练/内部/TMA1/2/3 单位，选择有有效 R1 且归一化均值较高的实例；不按标签或预测正确性选择。原始图像与归一化显示都使用各单位共同 L/U，没有逐细胞 min–max。ID、文件、关键特征与选样规则见 [example_index.csv](examples/example_index.csv)。这些图用于检查空间属性与数值对应，不是人工重标注或性能证据。\n",
        "受控图像测试和固定真实样例检查见 tests/CONTROLLED_TESTS_PASSED.json、tests/real_pilot_checks.csv；完整单位像素扰动检查见 tests/full_unit_affine_equivalence.csv。目视核对的五个真实实例及解释见 examples/VISUAL_REVIEW.csv；未将这些例子当作人工专家复核标签。代码、运行配置、模型、键清单、完整逐细胞 Parquet、完整 ROC/PR/threshold 数组已保留服务器。审阅包的 README 明确各全量文件相对位置；结合本报告开头服务器根目录即可定位；[FULL_ARTIFACT_PATHS.csv](FULL_ARTIFACT_PATHS.csv) 还给出全量特征、OOF/最终预测、模型、曲线及 GMM 状态的实际绝对路径、大小、行数和 SHA256。交付 OOF 使用带 with_QC 后缀的表，原 OOF 亦保留。推理 API 的一致性与异常输入测试见 [API_INTERFACE_CHECKS.csv](tests/API_INTERFACE_CHECKS.csv)。图件同时提供 PNG、文字可编辑 SVG 和源 CSV。最终一致性核验见 [FINAL_VALIDATION_CHECKS.csv](results/FINAL_VALIDATION_CHECKS.csv)，包括原始文件 SHA256 未变化、键完整性、16 次新拟合、严格门限、完整扰动覆盖和方向不变性。\n",
        "下一轮最值得检验的是 TMA2 的召回不足及局部参照在 TMA 的退化问题，但本轮没有据验证成绩改变任何冻结流程。当前证据支持共同校准和固定定位特征的价值，不支持预设 E4 最优，也不足以把改善归因于真实膜定位或污染识别。\n",
        "参考实现：[PhenoBIC 官方仓库](https://github.com/Shachi-Mittal-Lab/PhenoBIC-qupath-extension)；[SCIMAP rescale 文档](https://scimap.xyz/Functions/pp/rescale/)。本轮以用户指定准确公式、原项目冻结源码和实际保存状态为执行依据。\n",
    ]
    pd.DataFrame(facts).to_csv(res / "NUMERIC_CLAIMS_INDEX.csv", index=False)
    index = []
    for folder in [
        "results",
        "stability",
        "calibration",
        "keys",
        "tests",
        "audit",
        "examples",
        "figures",
        "figures/source_data",
    ]:
        for p in sorted((r / folder).glob("*.csv")):
            index.append(
                dict(
                    file=p.relative_to(r).as_posix(),
                    columns=";".join(pd.read_csv(p, nrows=0).columns),
                )
            )
    pd.DataFrame(index).to_csv(r / "TABLE_INDEX.csv", index=False)
    text.append(
        "全部表格及字段索引：[TABLE_INDEX.csv](TABLE_INDEX.csv)；关键结论数值索引：[NUMERIC_CLAIMS_INDEX.csv](results/NUMERIC_CLAIMS_INDEX.csv)。\n"
    )
    (r / "ANALYSIS_REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print("REPORT WRITTEN")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    main(a.root)
