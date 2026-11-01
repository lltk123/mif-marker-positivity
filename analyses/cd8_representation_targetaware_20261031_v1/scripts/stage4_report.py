"""Generate an evidence-linked Chinese report only after all completion guards."""
import json, html, time
import numpy as np, pandas as pd
from experiment_common import *

GROUPS = [
    "image",
    "image_target",
    "image_target_other",
    "tabular_only",
    "image_target_other_tabular",
]


def mdtable(d):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, ".4f")
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
    )


def main():
    for p in [
        "results/EVALUATION_COMPLETE.json",
        "results/COMPOSITION_COMPLETE.json",
        "review/REVIEW_COMPLETE.json",
        "reports/FIGURES_COMPLETE.json",
    ]:
        assert (RUN / p).exists(), p
    m = pd.read_csv(RUN / "results/METRICS_BY_UNIT.csv")
    risk = pd.read_csv(RUN / "results/RISK_STRATA.csv")
    cal = pd.read_csv(RUN / "results/TARGET_LABEL_CALIBRATION_SIMULATION.csv")
    comp = pd.read_csv(RUN / "results/COMPOSITION_SUMMARY.csv")
    selected = json.loads((RUN / "models/MODEL_SELECTION_FROZEN.json").read_text())[
        "selected_model"
    ]
    review = json.loads((RUN / "review/REVIEW_COMPLETE.json").read_text())
    gpu = json.loads((RUN / "logs/GPU_BUDGET.json").read_text())

    def one(n, mode="fixed_0.5"):
        return m[
            (m.model == n)
            & (m.split == TEST)
            & (m.level == "split")
            & (m.threshold_mode == mode)
        ].iloc[0]

    def rr(n, stratum):
        return risk[
            (risk.model == n)
            & (risk.split == TEST)
            & (risk.level == "split")
            & (risk.stratum == stratum)
            & (risk.threshold_mode == "dev_recall_90")
        ].iloc[0]

    h0, h1, h2 = [one(n) for n in ["H0", "H1", "H2"]]
    expected = {
        "tp": 17293,
        "fp": 706,
        "fn": 5038,
        "tn": 414443,
        "F1": 0.8575750061988594,
        "AP": 0.9583571838791716,
    }
    check = {"expected": expected, "observed": {k: float(h0[k]) for k in expected}}
    check["passed"] = all(abs(expected[k] - h0[k]) <= 1e-12 for k in expected)
    savej(RUN / "results/EXTERNAL_BASELINE_REPRODUCTION.json", check)
    if not check["passed"]:
        raise ValueError("H0 external legacy baseline mismatch; delivery halted")
    image, aware = rr("ENS_image", "contact_bright_proxy_negative"), rr(
        "ENS_image_target_other", "contact_bright_proxy_negative"
    )
    weak0, weak1 = rr("ENS_image", "weak_positive"), rr(
        "ENS_image_target_other", "weak_positive"
    )
    relative = aware.FPR / image.FPR - 1 if image.FPR > 0 else np.nan
    weakdelta = weak1.recall - weak0.recall
    success = bool(np.isfinite(relative) and relative <= -0.2 and weakdelta >= -0.02)
    reltext = f"{relative:+.1%}" if np.isfinite(relative) else "无法计算（图像基线FPR为0）"
    names = (
        ["H0", "H1", "H2", "H3", "H4", "H5"] + ["ENS_" + g for g in GROUPS] + ["GMM"]
    )
    main = m[
        (m.split == TEST)
        & (m.level == "split")
        & (m.threshold_mode == "fixed_0.5")
        & m.model.isin(names)
    ][["model", "precision", "recall", "F1", "AP", "tp", "fp", "fn", "n_cells"]].copy()
    main["order"] = main.model.map({v: i for i, v in enumerate(names)})
    main = main.sort_values("order").drop(columns="order")
    source = m[
        (m.split == TEST)
        & (m.level == "context")
        & (m.threshold_mode == "dev_recall_90")
        & m.model.isin(list(dict.fromkeys(["H0", "H2", selected])))
    ][["model", "context", "threshold", "precision", "recall", "F1", "AP"]]
    cb = (
        cal[
            (cal.model == selected)
            & (cal.level == "split")
            & (cal.unit == TEST)
            & cal.stratum.isna()
        ]
        .groupby("budget_per_core")[["F1", "recall", "FPR", "Brier"]]
        .agg(["mean", "std"])
    )
    cb.columns = ["_".join(z) for z in cb.columns]
    cb = cb.reset_index()
    main.to_csv(RUN / "reports/CONCISE_EXTERNAL_RESULTS.csv", index=False)
    source.to_csv(RUN / "reports/SELECTED_SOURCE_RESULTS.csv", index=False)
    cb.to_csv(RUN / "reports/CONCISE_CALIBRATION_RESULTS.csv", index=False)
    md = f"""# CD8 marker阴阳性识别：三阶段受控实验报告

三阶段受控计算已完成。按冻结的**完整内部WSI等权AP与复杂度规则**选定候选为 **{selected}**。外部完整结果全部报告，没有据此重新挑模型、门限或追加搜索。真实病例审阅已导出 **{review['n_cases']}例**，状态仍是**待用户判读**；计算完成不等于机制问题已经解决。旧代码、分割、标签、权重与历史结果保留，Pan-M持续封存。

## 已完成与评价边界

固定3张训练WSI（101,057有标签细胞）、3张内部WSI（539,132）、42个外部组织芯（437,480）。患者映射缺失，42芯不是42独立患者。主结果保留完整有标签集合；目标标签模拟另列，不与零目标标签泛化混合。网络未观测上下文使用零哨兵，实际有效支持与目标截断单列于CROP_SUPPORT_QC.csv；有限预测不等于完整有效观测。

完成7个固定HGB、5组神经模型各3固定种子。神经模型同96px原尺度crop、同三通道名义结构、同初始化与样本计划、同增强和最大预算；名义参数均202,306，活动参数另列。图像通道统一为CD8、目标实例mask、其他实例mask；缺少的mask通道置零，邻居由已有分割ID产生，不按CD8标签筛选。精简特征与图像的融合为端到端联合训练，没有使用测试分数训练stacking。

120个真实开发细胞的17项E2特征复现、ID和crop检查通过；10项合成验证通过；五组网络的有限损失、梯度、二值mask与配对初始化检查通过。H0旧E2的开发评分和外部混淆矩阵/F1/AP均复现。

神经检查点使用固定、标签盲选的每内部WSI最多10,000实例；完整内部集用于最终候选与判定点开发。内部是开发成绩，本轮没有新OOF。PhenoBIC外部集合已有历史实验曝光，不是前瞻性未见队列。

## 第一步：逐项改变的HGB实验

H0：旧E2的8强度+9定位、原共同min/max边界、原生NaN。
H1：同17特征，只替换成外层P1/P99边界。
H2：同H1，保留其余7项定位，用13项无需MAD描述替换两个退化指标。
H3：H2做训练中位数插补，加缺失标记。
H4：H3再加仅训练拟合的正态分位数变换，标记不变。
H5：H2插补但不加标记；H6：仅标记诊断，不作为部署候选。

H1-H0外部F1变化 **{h1.F1-h0.F1:+.4f}**、AP变化 **{h1.AP-h0.AP:+.4f}**。H2-H1的F1变化 **{h2.F1-h1.F1:+.4f}**、AP变化 **{h2.AP-h1.AP:+.4f}**，FP变化 **{h2.fp-h1.fp:+.0f}**、FN变化 **{h2.fn-h1.fn:+.0f}**。H2检验的是整套修复表征，不把效果归因于单一指标。

新特征包含非零支持率、条件正值强度、边缘/局部参照差、含精确平局处理的序关系。非零支持是测量描述，不等同生物学阳性。无支持和无正值信号分别保留状态，条件均值没有信号时仍为NaN，没有用epsilon制造信息。缺失标记的移除/仅标记对照用于诊断，不能单凭分数证明来源捷径或结构机制。

![逐项改变与实际来源召回](figures/stage1_zh.png)

{mdtable(main)}

以上均为严格score>0.5的外部描述性结果。三种子集成与单种子结果分别保存；集成没有新增拟合。NEURAL_SEED_SUMMARY.csv记录种子间描述性均值/标准差，不是患者置信区间。输入状态的统计分母为相应有标签集合，详见INPUT_QUALITY_LABELLED_POPULATION.csv；不与旧438,236个外部全实例分母混用。

## 第二步：目标归属与误报/漏报权衡

冻结内部R90判定点下，图像+目标+其他实例的集成相对图像分支：接触侧亮阴性代理组FPR相对变化 **{reltext}**，弱阳性代理组召回变化 **{weakdelta:+.4f}**。预注册描述性标准“FPR至少下降20%，弱阳性召回损失不超过0.02”：**{'满足' if success else '未满足'}**。实际外部总体召回分别为 **{one('ENS_image','dev_recall_90').recall:.4f}**、**{one('ENS_image_target_other','dev_recall_90').recall:.4f}**，需同时解释。

![目标归属输入和错误权衡](figures/target_aware_zh.png)

测试PR曲线只作诊断，星号来自内部冻结门限。误报减少不能自动证明真实邻居污染被消除。当前mask为expanded_nucleus，边缘不是验证过的真实细胞膜；真实物理像素间距未知，crop和尺度条仅用像素。

## 第三步：判定点迁移与目标标签模拟

内部目标召回不保证外部达到同等召回。下表为冻结内部R90门限下的实际来源表现，完整逐芯表保留。

{mdtable(source)}

目标标签模拟按每芯20/50/100标签、3个固定哈希重复，抽样不使用标签且预算嵌套。每芯预留全部100实例，从所有预算及零预算基线的评价中排除；每次留出433,280实例。只拟合固定L2惩罚的logistic截距，单类校准子集回退原映射。它是监督的芯内适配模拟，不是零目标标签泛化或独立患者验证。

{mdtable(cb)}

![固定标注预算和相同留出集合](figures/calibration_zh.png)

误差条为3次标签抽样标准差，不是生物学置信区间。每次每芯20/50/100对应全42芯840/2,100/4,200标签。TEST_ONLY_THRESHOLD_DIAGNOSTIC.csv中的测试来源专用门限仅作诊断，没有回写部署门限或选模。

## 组成依赖诊断

300个训练/内部目标保持原像素和mask不变，仅改变参与边界估计的其他实例组成，产生3,600条评分记录。原像素哈希与完整组成条件下的评分复现均通过。这是测量过程的组成敏感性诊断，不是独立泛化评价，也不是重复旧9600细胞的亮度/旋转检查。

![目标像素不变的组成诊断](figures/composition_zh.png)

{mdtable(comp[['model','pool','n_targets','mean_abs_score_delta','max_abs_score_delta','prediction_change_fraction']])}

## 待用户判读与仍未知的问题

打开[真实病例审阅](../review/index_zh.html)，在CASE_REVIEW.csv填写user_decision、segmentation_issue、reviewer_confidence和reviewer_notes。病例是预定义平衡来源/错误代理的说明性抽样，不用于估计总体错误率。原CD8 crop、实例mask和有效支持保留在NPZ；采用单位共同U的线性显示范围，无局部选择性增强。自动代理、原参考标签和自行查看都不冒充新增专家复核。

发布版输入中的大量零值原已存在，其上游背景扣除、阈值化或量化来源仍未知。本轮不能给出物理成因。下一步优先用真实病例判读区分参考标签争议、分割归属异常、真实弱信号与亮邻居代理，再决定新增原始成像或独立来源验证。旧E2保留为强基线，不自动替换既有模型。

## 时间、资源与复现

神经训练/开发推理及外部推理累计保守墙钟预算计数 **{gpu['consumed_seconds']/3600:.2f}小时**，上限8小时；不是纯GPU算子计时。每模型/分块实际耗时、进程RSS和GPU峰值单独保存。

完整权重、真实crop缓存和逐细胞预测留服务器新实验目录；E盘轻量包包含代码、配置、汇总CSV、图和病例。REPRODUCE.md给出环境与命令。完成标志包括FEATURES_COMPLETE、HGB_COMPLETE、NEURAL_COMPLETE、MODEL_SELECTION_FROZEN、COMPOSITION_COMPLETE、EVALUATION_COMPLETE、REVIEW_COMPLETE。没有自动提交或推送。
"""
    (RUN / "reports/REPORT_zh.md").write_text(md, encoding="utf-8")
    sections = [
        f'<h2>结论与完成边界</h2><p>内部冻结规则选定候选：<strong>{html.escape(selected)}</strong>。三阶段计算已完成；真实病例{review["n_cases"]}例仍待用户判读。外部完整结果全部报告，未追加搜索。</p>',
        f"<p>归一化单项改变F1：{h1.F1-h0.F1:+.4f}；表征修复整套改变：{h2.F1-h1.F1:+.4f}。目标归属代理FPR变化{reltext}，弱阳性召回变化{weakdelta:+.4f}。</p>",
        "<h2>外部固定0.5结果</h2>"
        + main.to_html(index=False, float_format=lambda x: format(x, ".4f")),
        '<img src="figures/stage1_zh.png"><img src="figures/target_aware_zh.png">',
        "<h2>内部R90下实际逐来源表现</h2>"
        + source.to_html(index=False, float_format=lambda x: format(x, ".4f")),
        "<h2>严格隔离的目标标签模拟</h2><p>每芯预留100实例从所有预算和零预算评价中排除；同一重复内评价集合相同，留出433,280实例。三次抽样波动不是患者置信区间。</p>"
        + cb.to_html(index=False, float_format=lambda x: format(x, ".4f")),
        '<img src="figures/calibration_zh.png"><img src="figures/composition_zh.png">',
        '<h2>真实病例与完整材料</h2><p><a href="../review/index_zh.html">病例审阅</a> · <a href="../review/CASE_REVIEW.csv">可填写的判读清单</a> · <a href="REPORT_zh.md">完整中文报告</a> · <a href="REPRODUCE.md">复现命令</a> · <a href="CONCISE_EXTERNAL_RESULTS.csv">精简结果CSV</a></p><p>扩张核近似边界不是已知真实膜；像素间距未知。说明性病例不能估计总体错误率。既存零值的上游生成机制仍未知。</p>',
        "<details><summary>展开完整方法、限制和审计说明</summary><pre>"
        + html.escape(md)
        + "</pre></details>",
    ]
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CD8三阶段受控实验</title><style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#213343;background:#f4f6f8;margin:0}main{max-width:1160px;margin:auto;padding:32px;background:white}h1{font-size:30px}h2{margin-top:36px;font-size:22px;color:#355b70}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #dce3e8;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto;margin:20px 0}a{color:#276b8c}pre{white-space:pre-wrap;font:14px/1.6 sans-serif}.note{padding:16px;border-left:4px solid #6b9a9d;background:#eef4f4}</style><main><h1>CD8 marker阴阳性识别<br>三阶段受控实验</h1><p class="note">计算完成 · 旧实验保留 · 外部不参与选模 · Pan-M封存 · 用户病例判读待完成</p>'
        + "".join(sections)
        + "</main></html>"
    )
    (RUN / "reports/REPORT_zh.html").write_text(page, encoding="utf-8")
    savej(
        RUN / "reports/REPORT_COMPLETE.json",
        {
            "selected_model": selected,
            "real_case_review_pending": True,
            "predefined_proxy_criterion_met": success,
            "source_metrics_sha256": sha(RUN / "results/METRICS_BY_UNIT.csv"),
            "source_risk_sha256": sha(RUN / "results/RISK_STRATA.csv"),
            "code_sha256": sha(__file__),
            "no_new_experiments_during_reporting": True,
        },
    )
    print("REPORT_COMPLETE", selected, flush=True)


if __name__ == "__main__":
    main()
