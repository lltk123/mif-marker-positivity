"""Build the Chinese report strictly from completed evaluations, never from placeholders."""
from common import *


def mdtable(frame, columns):
    def fmt(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "/")

    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        + [
            "| " + " | ".join(fmt(row[c]) for c in columns) + " |"
            for _, row in frame.iterrows()
        ]
    )


def main():
    required = [
        "provenance/numeric_pipeline_complete.json",
        "provenance/prediction_integrity_passed.json",
        "tests/PILOT_PASSED.json",
        "models/finetuned_complete.json",
        "results/all_metrics.csv",
        "figures/FIGURES_MANIFEST.json",
        "development_review/development_cases.csv",
    ]
    for relative in required:
        if not (R / relative).is_file():
            raise FileNotFoundError(
                f"cannot report a completed experiment: missing {R / relative}"
            )
    for model in [MODEL_PRETRAINED, MODEL_FINETUNED, "GMM"]:
        marker = R / "provenance" / (model + "_evaluation_complete.json")
        state = json.loads(marker.read_text())
        for path, checksum in state["output_sha256"].items():
            assert sha(path) == checksum, f"evaluation output changed: {path}"
    table = pd.read_csv(R / "results/all_metrics.csv")
    if set(table.model) != {MODEL_PRETRAINED, MODEL_FINETUNED, "GMM"}:
        raise ValueError("incomplete model evaluation")
    old = pd.read_csv(PREVIOUS_NIMBUS / "results/all_metrics.csv")
    comparisons = []
    join = ["threshold_mode", "level", "split", "context", "image_id"]
    for new_name, old_name in [
        (MODEL_PRETRAINED, "Nimbus_pretrained"),
        (MODEL_FINETUNED, "Nimbus_finetuned"),
    ]:
        paired = table.query("model==@new_name").merge(
            old.query("model==@old_name"),
            on=join,
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )
        assert paired.n_new.eq(paired.n_old).all()
        for metric in ["TP", "FP", "FN", "precision", "recall", "F1", "AP", "ROC_AUC"]:
            paired["delta_" + metric] = (
                paired[metric + "_new"] - paired[metric + "_old"]
            )
        comparisons.append(paired)
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(R / "results/paired_vs_previous_nimbus.csv", index=False)
    rows = []
    fit = json.loads((R / "models/finetuned_complete.json").read_text())
    for model in [MODEL_PRETRAINED, MODEL_FINETUNED]:
        traces = [
            json.loads(p.read_text())
            for p in (R / "predictions" / model).glob("*.json")
        ]
        rows.append(
            dict(
                model=model,
                inference_seconds=sum(t["seconds"] for t in traces),
                inference_peak_gpu_allocated_gib=max(
                    t.get("peak_gpu_allocated_gib", 0) for t in traces
                ),
                inference_process_peak_rss_gib=max(
                    t.get("process_peak_rss_gib", 0) for t in traces
                ),
                fit_seconds=fit["seconds"] if model == MODEL_FINETUNED else 0,
                fit_peak_gpu_allocated_gib=fit["peak_gpu_allocated_gib"]
                if model == MODEL_FINETUNED
                else 0,
                fit_process_peak_rss_gib=fit["peak_rss_gib"]
                if model == MODEL_FINETUNED
                else 0,
            )
        )
    runtime = pd.DataFrame(rows)
    runtime.to_csv(R / "results/runtime_summary.csv", index=False)
    reportcols = [
        "model",
        "split",
        "precision",
        "recall",
        "F1",
        "AP",
        "ROC_AUC",
        "coverage",
        "TP",
        "FP",
        "FN",
    ]
    primary = table.query(
        "level=='split' and threshold_mode=='official_default' and split!='training_WSIs'"
    )
    supplement = table.query(
        "level=='split' and threshold_mode=='validation_selected' and split!='training_WSIs'"
    )
    slide = table.query(
        "level=='slide' and split=='external_val_TMA_cores' and threshold_mode=='official_default'"
    )
    main_delta = paired.query(
        "level=='split' and threshold_mode=='official_default' and split!='training_WSIs'"
    )
    for name, frame in [
        ("report_primary", primary),
        ("report_validation_threshold", supplement),
        ("report_external_TMA", slide),
        ("report_vs_previous", main_delta),
    ]:
        frame.to_csv(R / "results" / (name + ".csv"), index=False)
    bounds = pd.read_csv(R / "results/normalization_parameters.csv")
    p = protocol()
    checkpoint = json.loads((R / "models/finetuned_checkpoint.json").read_text())
    conclusions = []
    for _, row in main_delta.query("split=='external_val_TMA_cores'").iterrows():
        direction = "提高" if row.delta_F1 > 0 else ("下降" if row.delta_F1 < 0 else "持平")
        conclusions.append(
            f"{row.model_new} 外部总体 F1 相对旧 Nimbus {direction} {abs(row.delta_F1):.4f}；AP 变化 {row.delta_AP:+.4f}，TP/FP/FN 变化分别为 {int(row.delta_TP):+d}/{int(row.delta_FP):+d}/{int(row.delta_FN):+d}。"
        )
    n_failed = int(primary.n_failed.sum())
    text = f"""# Nimbus：outer P1/P99 共同校准实验

本次已完成原 V1 权重在新输入上的推理，以及从原 V1 开始的独立微调和全量评价。原 Nimbus 和原始数据未改写，Pan-M 保持封存。此为输入归一化的研究适配，不是作者论文全部实验复现。

## 主要结果

{' '.join(conclusions)} 对应表为 results/report_vs_previous.csv，筛选 level=split、threshold_mode=official_default、split=external_val_TMA_cores，查看 delta_F1、delta_AP、delta_TP、delta_FP、delta_FN。不能仅据 F1 变化推断邻居污染减少。

固定 score > 0.5 的主结果来自 results/report_primary.csv；两模型在同一完整标签集合比较，未按特征或预测成功情况删行。

{mdtable(primary, reportcols)}

## 验证集门限

补充门限仅在内部三个 WSI 的完整标签上，按等权平均 F1 选择 0.01–0.99 网格；并列优先距离0.5更近，再取较高者。微调检查点也由内部验证选择，因此内部为开发拟合成绩。本轮没有新 OOF 或额外参数搜索。GMM 仅保留原分数和固定0.5，不重新调参。

来源 results/report_validation_threshold.csv；实际门限和网格分别见 models/*_thresholds.json、results/*_validation_threshold_grid.csv。

{mdtable(supplement, reportcols)}

## 外部各 TMA 与组织芯

来源 results/report_external_TMA.csv，筛选 level=slide、split=external_val_TMA_cores、threshold_mode=official_default。

{mdtable(slide, ['model','context','precision','recall','F1','AP','TP','FP','FN'])}

逐组织芯完整指标在 results/all_metrics.csv，筛选 level=image；等权均值、中位数、最低F1及无阳性预测单位数在 results/unit_equal_weight_summary.csv。病例身份缺失，组织芯不等同独立患者。无真阳性时 recall/AP/ROC-AUC 按定义输出 NA，且无阳性预测时 precision=0；无真阳性且无预测阳性时 F1=NA。未进行细胞级显著性检验。

## 输入定义与训练

共享校准由 cd8_robust_calibration_20261030 提供：每个至少5个有限有效像素的原实例先计算P10/P90，再按完整原WSI或单个TMA芯的全部合格实例等权计算外层P1/P99。包含无标签实例；不读标签或模型预测估计边界。计算float64，裁剪到[0,1]后转float32网络输入；不再除以17.2643。校准表在 results/normalization_parameters.csv，按image_id查看L/U、n_eligible、calibration_invalid。当前校准单位数为 {len(bounds)}。原始图像、实例ID及掩码未改动。

官方源码固定 {p['official_inference_commit']}，V1权重revision {p['weight_revision']}、SHA256 {WEIGHT_SHA}。模型、既有0.5缩放、全图网格与完整原实例聚合保持不变。固定源码的实际TTA包含四次270°及上下/左右翻转；保持该行为并验证流式等价。模型只输入CD8与掩码内部二值图，不输入GMM、人工标签、形态表或身份编号。

微调seed=0，最多10epochs，patience=5，batch=8，学习率3e-6。初始化为原V1，不加载旧微调权重。选择记录见 models/finetuned_checkpoint.json；当前选定validation_call={checkpoint['validation_call']}。训练与验证的几何tile集合在无校准失败时强制与旧实验一致。患者信息缺失，保留官方WSI划分。掩码为expanded_nucleus，边缘是扩张核近似边缘，不能解释成验证过的真实细胞膜。

## 核验、耗时与错误审阅

tests/pilot_reference_comparison.csv 给出真实训练、内部及TMA1/2/3样例对官方推理的数值对照；tests/global_preparation_pixel_checks.csv 检查归一化和全图缩放。tests/offline_numeric_checks.csv 的人工bounds crop检查仅验证算术，不代表完整单位校准。results/prediction_integrity.csv 和 results/seam_cell_audit.csv 检查原ID、一行一细胞及接缝覆盖。

校准失败时保留全部ID、score=NaN，不能用网络对NaN的有限输出代替有效观测。主表失败计数合计为 {n_failed}（按模型/划分相加，不是去重细胞数）；具体n_failed/coverage查看主表。局部未观测像素在normalized_full中保留NaN及有效支持；仅网络张量使用零哨兵表示未观测上下文，不将其计为真实观测。所有目标实例像素必须有效。

资源来源 results/runtime_summary.csv；fit_seconds包含优化及验证，RSS为运行进程高水位，不能解释为纯模型权重内存。

{mdtable(runtime, list(runtime.columns))}

开发集病例索引、原始CD8图、掩码叠加与模型对照见 development_review/development_cases.csv、case_model_comparison.csv 和 figures/*/development_*.png/svg。弱表达、邻居接触侧亮、局部高参照等均为预定图像属性候选，不能单凭这些代理确认污染原因。图件的数值源表位于 figures/source_data。

## 复现与完整文件

运行入口：`python scripts/run_pipeline.py --stage all`。服务器目录：`{R}`。完整逐细胞结果在 predictions/{MODEL_PRETRAINED}/all_cells.parquet 和 predictions/{MODEL_FINETUNED}/all_cells.parquet；含原键、标签、split/context、连续评分、两套预测及QC。曲线和threshold数组在 results/curves；微调权重在 models/{MODEL_FINETUNED}.pt。旧结果比较表为 results/paired_vs_previous_nimbus.csv，GMM比较为 results/paired_vs_GMM.csv。代码/输入/校准/权重哈希变更拒绝复用缓存。

上述归一化是项目适配，文献没有证明双层P1/P99最优。极亮实例可是真阳性；新裁剪可能压缩强阳性差异或放大稀少阳性图的背景。当前结果仅支持此固定协议的观测变化。
"""
    (R / "ANALYSIS_REPORT.md").write_text(text, encoding="utf-8")
    dump(
        R / "provenance/delivery_complete.json",
        dict(
            status="complete" if n_failed == 0 else "complete_with_unscorable_cells",
            report_sha256=sha(R / "ANALYSIS_REPORT.md"),
            protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
            main_table_failed_observations=n_failed,
        ),
    )
    print("REPORT_COMPLETE", R / "ANALYSIS_REPORT.md")


if __name__ == "__main__":
    main()
