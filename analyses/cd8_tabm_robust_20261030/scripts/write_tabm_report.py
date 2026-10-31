"""Generate an evidence-linked Chinese report from completed frozen TabM results.

No fitting, threshold selection, external image reading or Pan-M access occurs.
The report fails closed when required evaluation/validation artifacts are absent.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN, VALID, TEST = "training_WSIs", "internal_val_WSIs", "external_val_TMA_cores"
MODES = ["fixed_0.5", "validation_selected"]
EXPECTED = {f"{f}_{g}_s{s}" for f in ["TabM", "HGB"] for g in "ABC" for s in range(3)}
GROUP = [
    "model_family",
    "feature_set",
    "threshold_mode",
    "split",
    "group_type",
    "group_id",
    "context",
]
MEASURES = [
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "average_precision",
    "tp",
    "fp",
    "fn",
    "tn",
    "true_positive_fraction",
    "predicted_positive_fraction",
    "positive_fraction_absolute_error",
    "prediction_coverage",
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    require(path.is_file(), "Missing required artifact: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path, columns=()):
    require(path.is_file(), "Missing required table: " + str(path))
    frame = pd.read_csv(path, float_precision="round_trip")
    require(
        set(columns).issubset(frame), f"{path.name}: missing {set(columns)-set(frame)}"
    )
    return frame


def aggregate(frame, keys, columns):
    rows = []
    for key, part in frame.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        require(
            not part.seed.duplicated().any(), "Duplicate seed in summary: " + str(key)
        )
        row = dict(zip(keys, key), n_seeds=part.seed.nunique())
        for col in columns:
            values = pd.to_numeric(part[col], errors="raise")
            values = values[np.isfinite(values)]
            row.update(
                {
                    col + "_mean": values.mean(),
                    col + "_std": values.std(ddof=1),
                    col + "_n_finite": len(values),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value, digits=4):
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def mean_sd(row, metric, digits=4):
    out = fmt(row[metric + "_mean"], digits)
    if pd.notna(row[metric + "_std"]):
        out += " ± " + fmt(row[metric + "_std"], digits)
    return out


def markdown(frame, columns):
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col, _ in columns:
            value = row[col]
            vals.append(
                ("NA" if pd.isna(value) else str(value))
                .replace("|", "\\|")
                .replace("\n", " ")
            )
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def model_label(row):
    return (
        "GMM（固定0.5）"
        if row.model_family == "GMM"
        else row.model_family + "_" + row.feature_set
    )


def display(frame, metrics):
    frame = frame.copy()
    frame["model"] = frame.apply(model_label, axis=1)
    for metric in metrics:
        frame[metric + "_display"] = frame.apply(
            lambda row: mean_sd(row, metric), axis=1
        )
    return frame


def build_report(root):
    paths = {
        "protocol": "TABM_PROTOCOL.json",
        "runtime": "RUN_CONFIG.json",
        "frozen": "FROZEN_TEST_PROTOCOL.json",
        "ready": "data/DATA_READY.json",
        "audit": "AUDIT_PROTOCOL.json",
        "evaluation": "results/EVALUATION_MANIFEST.json",
        "delivery": "DELIVERY_VALIDATION.json",
        "review": "development_review/REVIEW_MANIFEST.json",
        "features": "FEATURE_SETS.json",
        "upstream": "upstream/SOURCE_MANIFEST.json",
    }
    docs = {k: read_json(root / v) for k, v in paths.items()}
    protocol, runtime, frozen, ready = [
        docs[k] for k in ["protocol", "runtime", "frozen", "ready"]
    ]
    evaluation, delivery, review = [
        docs[k] for k in ["evaluation", "delivery", "review"]
    ]
    require(
        protocol.get("normalization_method")
        == ready.get("normalization_method")
        == frozen.get("normalization_method")
        == "E1_outer_P1_P99",
        "Calibration provenance mismatch",
    )
    require(
        frozen["signature"]["normalization"] == ready["normalization_signature"],
        "Shared extractor signature mismatch",
    )
    require(
        all(d.get("status") == "complete" for d in [evaluation, delivery, review]),
        "Incomplete evaluation/delivery/review",
    )
    require(
        delivery.get("PanM_accessed") is False, "Pan-M status is not explicitly sealed"
    )
    require(
        set(frozen["models"]) == EXPECTED and delivery["n_models"] == 18,
        "All 18 final models are required",
    )
    require(
        evaluation["n_models"] == 19, "Expected 18 supervised models and unchanged GMM"
    )
    require(
        evaluation.get("curves", "").startswith("full ROC/PR"),
        "Full curves have not been exported",
    )
    require(
        evaluation["protocol_sha256"] == digest(root / paths["frozen"]),
        "Evaluation/frozen protocol hash mismatch",
    )
    require(
        frozen["signature"]["data"] == ready["cells_sha256"],
        "Frozen data hash mismatch",
    )
    require(
        delivery["n_cells_per_model"] == ready["cells"] == evaluation["n_all_cells"],
        "Cell totals disagree",
    )
    require(
        protocol["official_commit"] == docs["upstream"]["commit"],
        "Official commit mismatch",
    )
    require(sorted(protocol["seeds"]) == [0, 1, 2], "Three formal seeds required")
    risk_hash = digest(root / "data/risk_strata.json")
    require(
        docs["audit"]["risk_strata_sha256"]
        == risk_hash
        == evaluation["risk_strata_sha256"],
        "Risk strata hash mismatch",
    )
    require(
        review["fingerprint"]["rules_sha256"] == risk_hash,
        "Development review uses different frozen risk strata",
    )
    require(
        docs["audit"]["frozen_unix_time"] < frozen["frozen_unix_time"],
        "Risk audit was not frozen before model/test protocol",
    )
    require(
        docs["audit"]["development_only_visual_review"] is True
        and docs["audit"]["causal_claim"] is False,
        "Unexpected audit policy",
    )
    schemas = {
        "metrics_all.csv": GROUP
        + ["model_id", "seed", "n_cells", "threshold"]
        + MEASURES,
        "metrics_seed_summary.csv": GROUP + ["metric", "mean", "std_between_seeds"],
        "paired_comparisons.csv": [
            "comparison",
            "seed",
            "threshold_mode",
            "split",
            "group_type",
            "group_id",
            "context",
            "delta_f1",
            "delta_tp",
            "delta_fp",
            "delta_fn",
        ],
        "risk_strata_metrics.csv": GROUP
        + ["seed", "stratum", "n_cells", "n_true_negative", "fp", "fn", "recall"],
        "unit_equal_summary.csv": [
            "model_id",
            "model_family",
            "feature_set",
            "seed",
            "threshold_mode",
            "split",
            "metric",
            "unit_equal_mean",
            "n_units",
            "n_finite_units",
            "n_units_no_positive_predictions",
        ],
        "feature_availability.csv": [
            "split",
            "feature",
            "n",
            "n_missing",
            "missing_rate",
        ],
        "feature_missingness_by_unit_class.csv": [
            "split",
            "unit_id",
            "true_class",
            "feature",
            "n_cells",
            "n_missing",
            "missing_fraction",
        ],
        "numeric_resolution_train.csv": [
            "feature",
            "is_hu_moment",
            "n_train_cells",
            "raw_nunique_train",
            "transformed_float32_nunique_train",
            "unique_retention_ratio",
            "resolution_note",
        ],
        "inference_resources.csv": [
            "model_id",
            "n",
            "inference_seconds",
            "wall_seconds_including_io",
            "peak_phase_process_rss_bytes",
            "phase_start_rss_bytes",
            "peak_rss_increase_bytes",
        ],
        "submodel_disagreement.csv": [
            "model_id",
            "model_family",
            "feature_set",
            "seed",
            "split",
            "group_type",
            "true_class",
            "std_mean",
            "std_p95",
        ],
        "curve_manifest.csv": ["status", "path"],
        "delivery_prediction_validation.csv": [
            "model_id",
            "n",
            "n_unique",
            "all_scores_finite",
            "api_max_absolute_difference",
        ],
    }
    tables = {n: read_csv(root / "results" / n, c) for n, c in schemas.items()}
    metrics, pairs, risk = [
        tables[n]
        for n in [
            "metrics_all.csv",
            "paired_comparisons.csv",
            "risk_strata_metrics.csv",
        ]
    ]
    require(
        set(metrics.model_id) == EXPECTED | {"GMM"}, "Performance models incomplete"
    )
    validation = tables["delivery_prediction_validation.csv"]
    require(
        set(validation.model_id) == EXPECTED and len(validation) == 18,
        "Prediction validation incomplete",
    )
    require(
        validation.all_scores_finite.eq(True).all(), "Finite prediction checks failed"
    )
    require(
        validation.n.eq(ready["cells"]).all()
        and validation.n_unique.eq(ready["cells"]).all(),
        "Prediction IDs incomplete",
    )
    for mid in EXPECTED:
        for split, count in ready["labeled_counts"].items():
            part = metrics.loc[
                metrics.model_id.eq(mid)
                & metrics.split.eq(split)
                & metrics.group_type.eq("split")
            ]
            require(
                len(part) == 2
                and set(part.threshold_mode) == set(MODES)
                and part.n_cells.eq(count).all(),
                f"Incomplete split metrics: {mid}/{split}",
            )
    for path in tables["curve_manifest.csv"].loc[
        lambda x: x.status.eq("complete"), "path"
    ]:
        require((root / path).is_file(), "Missing curve: " + str(path))
    cases = read_csv(root / "development_review/case_index.csv", ["split"])
    require(
        len(cases) == review["n_cases"] and cases.split.isin([TRAIN, VALID]).all(),
        "Wrong review case count/split",
    )
    if len(cases):
        require(
            "case_name" in cases and cases.case_name.notna().all(),
            "Missing names for exported review cases",
        )
    elif "case_name" not in cases:
        # The review selector intentionally writes an empty selection schema when
        # no eligible error is available; no case images should be fabricated.
        cases = cases.assign(case_name=pd.Series(dtype=str))
    stems = [
        "performance_ablation",
        "paired_core_changes_fixed_0.5",
        "paired_core_changes_validation_selected",
        "error_strata",
        "submodel_disagreement",
    ]
    # Match the plotting source: effective model features, not excluded raw columns.
    if tables["feature_missingness_by_unit_class.csv"].n_missing.gt(0).any():
        stems.append("feature_missingness")
    for language in ["zh", "en"]:
        for stem in stems:
            for ext in ["png", "svg"]:
                require(
                    (root / "figures" / language / f"{stem}.{ext}").is_file(),
                    "Missing figure: " + stem,
                )
        for stem in cases.case_name:
            for ext in ["png", "svg"]:
                require(
                    (
                        root / "development_review" / language / f"{stem}.{ext}"
                    ).is_file(),
                    "Missing case figure: " + stem,
                )

    artifacts, resources = [], []
    inference = tables["inference_resources.csv"]
    require(set(inference.model_id) == EXPECTED, "Inference timing models incomplete")
    for mid, meta in sorted(frozen["models"].items()):
        require(
            read_json(root / "models" / mid / "metadata.json") == meta,
            "Metadata changed after freeze: " + mid,
        )
        weight = root / (
            meta["weights_relative_path"]
            if meta["model_family"] == "TabM"
            else f"models/{mid}/model.joblib"
        )
        require(
            weight.is_file() and digest(weight) == meta["weights_sha256"],
            "Missing/changed weights: " + mid,
        )
        artifacts.append(
            dict(
                model_id=mid,
                family=meta["model_family"],
                feature_set=meta["feature_set"],
                seed=meta["seed"],
                effective_features=len(meta["features"]),
                threshold_validation=meta["threshold_validation"],
                weight_path=str(weight),
                weights_sha256=meta["weights_sha256"],
                best_epoch=meta.get("best_epoch"),
                epochs_run=meta.get("epochs_run"),
                epoch_cap_reached=meta.get("epoch_cap_reached"),
                learning_rate=meta.get("candidate", {}).get("learning_rate"),
            )
        )
        item = inference.loc[inference.model_id.eq(mid)]
        require(len(item) == 1, "Inference timing not unique: " + mid)
        item = item.iloc[0]
        opt = (
            meta["training_optimization_seconds"]
            if meta["model_family"] == "TabM"
            else meta["fit_seconds"]
        )
        val = (
            meta["validation_monitoring_seconds"]
            if meta["model_family"] == "TabM"
            else meta["validation_inference_seconds"]
        )
        resources.append(
            dict(
                model_id=mid,
                model_family=meta["model_family"],
                feature_set=meta["feature_set"],
                seed=meta["seed"],
                training_optimization_seconds=opt,
                validation_seconds=val,
                fit_wall_seconds=meta["fit_seconds"],
                inference_seconds=item.inference_seconds,
                inference_wall_seconds_with_io=item.wall_seconds_including_io,
                training_peak_phase_rss_GiB=meta["peak_phase_process_rss_bytes"]
                / 1024**3,
                training_baseline_rss_GiB=meta["phase_start_rss_bytes"] / 1024**3,
                training_rss_increase_GiB=meta["peak_rss_increase_bytes"] / 1024**3,
                inference_peak_phase_rss_GiB=item.peak_phase_process_rss_bytes
                / 1024**3,
                training_peak_gpu_allocated_GiB=meta.get("peak_gpu_allocated_bytes", 0)
                / 1024**3,
                inference_peak_gpu_allocated_GiB=item.get("peak_gpu_allocated_bytes", 0)
                / 1024**3,
            )
        )
    outputs, registry = {}, []

    def save_table(name, frame, source, selection, description):
        frame = frame.copy()
        if "normalization_method" not in frame:
            frame["normalization_method"] = "E1_outer_P1_P99"
            if "model_family" in frame:
                frame.loc[
                    frame.model_family.eq("GMM"), "normalization_method"
                ] = "original_unsupervised_GMM"
        rel = "results/report_" + name + ".csv"
        outputs[rel] = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        registry.append(
            dict(
                table_id=name,
                table_csv=rel,
                source=source,
                filter_and_aggregation=selection,
                metric_columns="|".join(frame.columns),
                description=description,
            )
        )

    inventory = pd.DataFrame(
        [
            dict(
                scope=s,
                n_labelled_cells=n,
                n_all_cells=np.nan,
                n_units=metrics.loc[
                    metrics.model_id.eq("GMM")
                    & metrics.split.eq(s)
                    & metrics.group_type.eq("image"),
                    "group_id",
                ].nunique(),
            )
            for s, n in ready["labeled_counts"].items()
        ]
        + [
            dict(
                scope="all_instances_including_unlabelled",
                n_labelled_cells=np.nan,
                n_all_cells=ready["cells"],
                n_units=np.nan,
            )
        ]
    )
    calibration_status = pd.DataFrame(
        [
            dict(
                n_cells=ready["cells"],
                n_calibration_invalid_cells=ready["calibration_invalid_cells"],
                n_calibration_invalid_units=ready["calibration_invalid_units"],
                calibration_coverage=1
                - ready["calibration_invalid_cells"] / ready["cells"],
            )
        ]
    )
    save_table(
        "calibration_status",
        calibration_status,
        "data/DATA_READY.json",
        "all original instances; no QC row deletion",
        "图像校准有效性",
    )
    save_table(
        "data_inventory",
        inventory,
        "data/DATA_READY.json;results/metrics_all.csv",
        "labelled_counts; GMM image-unit counts; all-instance total",
        "样本规模",
    )
    features = pd.DataFrame(
        [
            dict(
                feature_set=g,
                n_requested=len(docs["features"][g]["requested"]),
                n_effective=len(docs["features"][g]["effective"]),
                excluded=json.dumps(
                    docs["features"][g]["excluded"], ensure_ascii=False
                ),
            )
            for g in "ABC"
        ]
    )
    save_table(
        "feature_sets",
        features,
        "FEATURE_SETS.json",
        "requested versus effective",
        "特征集合",
    )
    artifacts = pd.DataFrame(artifacts)
    save_table(
        "model_artifacts",
        artifacts,
        "FROZEN_TEST_PROTOCOL.json;models/*/metadata.json",
        "18 canonical models; verified weight hashes",
        "模型、门限与运行状态",
    )
    resources = pd.DataFrame(resources)
    save_table(
        "resources",
        resources,
        "models/*/metadata.json;results/inference_resources.csv",
        "model_id join; bytes to GiB",
        "每模型耗时与资源",
    )
    perf = aggregate(metrics, GROUP, MEASURES + ["n_cells", "threshold"])
    for mode in MODES:
        save_table(
            "performance_" + mode,
            perf.loc[perf.threshold_mode.eq(mode) | perf.model_family.eq("GMM")],
            "results/metrics_all.csv;results/metrics_seed_summary.csv",
            f"{mode}; append fixed GMM; seed mean and sample SD",
            "性能汇总",
        )
    long = tables["metrics_seed_summary.csv"].loc[lambda x: x.metric.eq("f1")]
    check = perf.merge(long[GROUP + ["mean"]], on=GROUP, validate="one_to_one")
    require(
        len(check) == len(perf)
        and np.allclose(check.f1_mean, check["mean"], equal_nan=True),
        "Existing seed F1 summary disagrees",
    )
    paircols = [
        c
        for c in pairs
        if c.startswith("delta_") and pd.api.types.is_numeric_dtype(pairs[c])
    ]
    paired = aggregate(
        pairs,
        ["comparison", "threshold_mode", "split", "group_type", "group_id", "context"],
        paircols,
    )
    save_table(
        "paired_deltas",
        paired,
        "results/paired_comparisons.csv",
        "within-seed differences; seed mean/SD",
        "特征与架构配对差异",
    )
    risk = risk.copy()
    risk["false_positive_rate"] = risk.fp / risk.n_true_negative.replace(0, np.nan)
    risks = aggregate(
        risk,
        GROUP + ["stratum"],
        ["n_cells", "recall", "false_positive_rate", "fp", "fn"],
    )
    save_table(
        "risk_strata",
        risks,
        "results/risk_strata_metrics.csv;data/risk_strata.json",
        "FPR=fp/n_true_negative; seed mean/SD",
        "弱阳性和图像属性代理",
    )
    units = aggregate(
        tables["unit_equal_summary.csv"],
        ["model_family", "feature_set", "threshold_mode", "split", "metric"],
        [
            "unit_equal_mean",
            "n_units",
            "n_finite_units",
            "n_units_no_positive_predictions",
        ],
    )
    save_table(
        "unit_equal",
        units,
        "results/unit_equal_summary.csv",
        "unit equal weighting first; then seed mean/SD",
        "单位等权结果",
    )
    low = perf.loc[
        perf.split.eq(TEST)
        & perf.group_type.eq("image")
        & perf.threshold_mode.eq(MODES[0])
        & perf.model_family.ne("GMM")
    ]
    low = (
        low.sort_values("f1_mean")
        .groupby(["model_family", "feature_set"], group_keys=False)
        .head(3)
    )
    save_table(
        "lowest_cores",
        low,
        "results/metrics_all.csv",
        "external fixed0.5; lowest 3 mean-seed F1 cores per family/set",
        "低表现单位",
    )
    missing = tables["feature_availability.csv"].copy()
    missing[
        "population_definition"
    ] = "all instances including unlabelled, before imputation"
    save_table(
        "missingness",
        missing,
        "results/feature_availability.csv",
        "original split-level all-instance denominators",
        "原始缺失",
    )
    numeric = tables["numeric_resolution_train.csv"]
    save_table(
        "numeric_resolution",
        numeric,
        "results/numeric_resolution_train.csv",
        "labelled training cells only",
        "训练数值分辨率",
    )
    d = tables["submodel_disagreement.csv"]
    d = d.loc[d.group_type.eq("split") & d.true_class.astype(str).eq("all")]
    disagreement = aggregate(
        d, ["model_family", "feature_set", "split"], ["std_mean", "std_p95"]
    )
    save_table(
        "disagreement",
        disagreement,
        "results/submodel_disagreement.csv",
        "split/all true classes; member SD then seed summary",
        "未校准成员分歧",
    )
    provenance = pd.DataFrame(
        [
            dict(item=k, value=str(v), source=src)
            for k, v, src in [
                (
                    "official_commit",
                    protocol["official_commit"],
                    "upstream/SOURCE_MANIFEST.json",
                ),
                ("tabm_version", runtime["tabm_version"], "RUN_CONFIG.json"),
                (
                    "rtdl_num_embeddings",
                    runtime["rtdl_num_embeddings"],
                    "RUN_CONFIG.json",
                ),
                ("hardware", runtime["hardware"], "RUN_CONFIG.json"),
                (
                    "analysis_code_identity",
                    "results/code_sha256.csv plus REPORT_GENERATION_MANIFEST.json",
                    "DELIVERY_VALIDATION.json",
                ),
                (
                    "risk_frozen_unix_time",
                    docs["audit"]["frozen_unix_time"],
                    "AUDIT_PROTOCOL.json",
                ),
                (
                    "model_frozen_unix_time",
                    frozen["frozen_unix_time"],
                    "FROZEN_TEST_PROTOCOL.json",
                ),
            ]
        ]
    )
    save_table(
        "provenance",
        provenance,
        "official and run configuration files",
        "exact recorded values",
        "版本和冻结顺序",
    )
    training_config = pd.DataFrame(
        [
            dict(parameter=k, value=json.dumps(protocol[k], ensure_ascii=False))
            for k in [
                "arch_type",
                "k",
                "n_blocks",
                "d_block",
                "dropout",
                "embedding",
                "optimizer",
                "candidates",
                "batch_size",
                "max_epochs",
                "patience",
                "seeds",
                "training_loss",
                "inference",
                "precision",
                "selection_metric",
                "HGB",
            ]
        ]
    )
    save_table(
        "training_configuration",
        training_config,
        "TABM_PROTOCOL.json",
        "exact frozen parameters; no defaults inferred",
        "模型、嵌入和训练配置",
    )

    sections = [
        "# PhenoBIC CD8：TabM A/B/C 与匹配 HGB 的完整实验报告",
        "本报告仅依据已冻结的完整实验。所有数值表的来源、筛选条件、聚合方式及指标列见 "
        "[report_tables.csv](results/report_tables.csv)。均值 ± 标准差是种子间描述性波动，不是患者置信区间。",
    ]

    def add(text):
        sections.append(text)

    def note(name, selection, columns):
        add(
            f"数值依据：[report_{name}.csv](results/report_{name}.csv)；筛选：{selection}；指标列：`{columns}`。"
        )

    def perf_table(part):
        shown = display(
            part,
            [
                "precision",
                "recall",
                "f1",
                "average_precision",
                "positive_fraction_absolute_error",
            ],
        )
        return markdown(
            shown,
            [
                ("model", "模型"),
                ("precision_display", "Precision"),
                ("recall_display", "Recall"),
                ("f1_display", "F1"),
                ("average_precision_display", "AP"),
                ("positive_fraction_absolute_error_display", "阳性比例绝对误差"),
            ],
        )

    add("## 1. 主要结论与完成边界")
    external = perf.loc[
        perf.split.eq(TEST)
        & perf.group_type.eq("split")
        & perf.threshold_mode.eq(MODES[0])
    ]
    best = (
        external.loc[external.model_family.eq("TabM")]
        .sort_values("f1_mean", ascending=False)
        .iloc[0]
    )
    note(
        "performance_fixed_0.5",
        f"split={TEST},group_type=split,model_family=TabM",
        "f1_mean,f1_std",
    )
    add(
        f"固定 0.5 门限的完整外部细胞集合上，TabM A/B/C 中平均 F1 最高为 **{best.feature_set}**，"
        f'F1 为 **{mean_sd(best,"f1")}**。该排序是冻结后的描述，不用于重选模型、特征或门限。'
    )
    for comparison in ["TabM_B_minus_A", "TabM_C_minus_B", "TabM_C_minus_A"]:
        part = paired.loc[
            paired.comparison.eq(comparison)
            & paired.split.eq(TEST)
            & paired.group_type.eq("split")
            & paired.threshold_mode.eq(MODES[0])
        ]
        require(len(part) == 1, "Missing primary comparison: " + comparison)
        r = part.iloc[0]
        direction = (
            "提高" if r.delta_f1_mean > 0 else "降低" if r.delta_f1_mean < 0 else "不变"
        )
        add(
            f'- {comparison}：总体 F1 {direction}，ΔF1={mean_sd(r,"delta_f1")}；'
            f"平均 ΔTP={fmt(r.delta_tp_mean,1)}、ΔFP={fmt(r.delta_fp_mean,1)}、ΔFN={fmt(r.delta_fn_mean,1)}。"
        )
    note(
        "paired_deltas",
        f"split={TEST},group_type=split,threshold_mode=fixed_0.5,comparison=上述三项",
        "delta_f1_mean/std,delta_tp_mean,delta_fp_mean,delta_fn_mean",
    )
    add(
        "已完成官方 TabM 实现的逐细胞 CD8 任务适配、从头训练、三个随机种子和匹配 HGB 对照；"
        "不宣称复现 TabM 原论文全部数据集或指标。Pan-M 持续封存，本轮没有读取其图像、标签、分布或预测。"
    )

    add("## 2. 数据、划分与输入")
    note(
        "data_inventory",
        "scope=各官方划分及all_instances_including_unlabelled",
        "n_labelled_cells,n_all_cells,n_units",
    )
    add(
        markdown(
            inventory,
            [
                ("scope", "集合"),
                ("n_labelled_cells", "有标签细胞"),
                ("n_all_cells", "全部实例"),
                ("n_units", "WSI/组织芯单位数"),
            ],
        )
    )
    add(
        "统一键为 dataset_id + image_id + cell_id + marker，通过显式 ID 对齐。保留官方训练 WSI、内部 WSI 与外部 TMA 划分；"
        "不随机打散细胞。缺少一致患者映射时按原 WSI/组织芯分组，不将组织芯当作独立患者。"
        "仅原训练划分有标签细胞用于监督优化和拟合预处理；无标签实例保留预测，不进入分类指标。"
    )
    note(
        "calibration_status",
        "all original instances",
        "n_cells,n_calibration_invalid_cells,n_calibration_invalid_units,calibration_coverage",
    )
    add(
        f"校准失败实例 {ready['calibration_invalid_cells']} 个、单位 {ready['calibration_invalid_units']} 个；所有 ID 保留。"
        "正式指标依旧使用完整有标签集合；模型可通过原缺失处理产生分数，但有限分数不等于校准观测有效。"
        "逐 split/单位指标的 n_calibration_invalid 与 calibration_coverage 单独报告。"
    )
    note("feature_sets", "feature_set=A/B/C", "n_requested,n_effective,excluded")
    add(
        markdown(
            features,
            [
                ("feature_set", "组合"),
                ("n_requested", "请求维数"),
                ("n_effective", "有效维数"),
                ("excluded", "排除原因"),
            ],
        )
    )
    add(
        "A 为 E1_outer_P1_P99 像素校准后重新提取的 8 项 CD8 强度；B 加入原有 19 项形态；C 加入同一归一化像素重提的 63 项空间特征。"
        "完整名称和顺序见 FEATURE_SETS.json、results/feature_manifest.csv。标识、患者/批次、真值和 GMM 不进入模型。"
        "当前 mask 为 expanded_nucleus；其边缘是扩张核近似边缘，不是经过验证的真实膜。长度和面积保持像素尺度。"
    )
    add(
        "图像共同边界为 L=P1(实例内P10)、U=P99(实例内P90)，每个单位独立使用本图至少含 5 个有限像素的全部实例估计边界，不读取标签；所有有效像素使用共同浮点映射并 clip 到 [0,1]。中位数插补、带拟合噪声的正态分位数变换、PLE 分箱均仅训练拟合。HGB 使用相同有效特征和预处理，"
        "便于区分特征组合与学习器收益；它与历史未经该预处理的 HGB 是不同实验条件。"
    )

    add("## 3. 官方实现、有限调参与冻结")
    note(
        "provenance",
        "item=official_commit/tabm_version/rtdl_num_embeddings/hardware",
        "value,source",
    )
    add(
        f'官方 commit 为 `{protocol["official_commit"]}`；TabM {runtime["tabm_version"]}，'
        f'rtdl_num_embeddings {runtime["rtdl_num_embeddings"]}。采用 {protocol["variant"]}。'
        "解析后的结构、优化器、嵌入参数与随机种子见 TABM_PROTOCOL.json；下面模型表记录实际训练状态。"
    )
    note("training_configuration", "全部冻结参数", "parameter,value")
    add(markdown(training_config, [("parameter", "参数"), ("value", "冻结配置")]))
    add(
        "使用两个 logits；每个成员分别计算交叉熵再平均。推理先成员 softmax，再平均阳性概率，没有平均 logits。"
        "每组种子 0 比较预定两个学习率，其他结构固定；按内部 WSI 等权 AP 早停和选择配置，再完成种子 1、2。"
        "内部还用于全局门限开发，其成绩属于开发拟合表现。GMM 保持原固定 0.5，不重新拟合或调门限。"
    )
    note(
        "model_artifacts",
        "family=TabM，各feature_set和seed",
        "learning_rate,best_epoch,epochs_run,epoch_cap_reached,threshold_validation",
    )
    add(
        markdown(
            artifacts.loc[artifacts.family.eq("TabM")].round(4),
            [
                ("model_id", "模型"),
                ("learning_rate", "学习率"),
                ("best_epoch", "最佳epoch"),
                ("epochs_run", "运行epoch"),
                ("epoch_cap_reached", "触及预算上限"),
                ("threshold_validation", "验证门限"),
            ],
        )
    )
    add(
        "风险分层在最终模型/测试协议前冻结，哈希核验通过。原训练 WSI 用于训练，内部 WSI 用于早停、学习率与门限开发；外部 PhenoBIC 在本轮模型及门限冻结后评价。"
    )

    for mode, title in [(MODES[0], "4. 固定 0.5 主结果"), (MODES[1], "5. 验证门限补充结果")]:
        add("## " + title)
        note(
            "performance_" + mode,
            "group_type=split；内部开发验证与外部测试分别如下；GMM始终fixed_0.5",
            "precision/recall/f1/average_precision/positive_fraction_absolute_error_mean及_std",
        )
        for split, label in [(VALID, "内部开发验证"), (TEST, "外部测试")]:
            part = perf.loc[
                perf.split.eq(split)
                & perf.group_type.eq("split")
                & (perf.threshold_mode.eq(mode) | perf.model_family.eq("GMM"))
            ]
            add("### " + label)
            add(perf_table(part))
        add(
            "总体指标先在每个种子的完整有标签集合上计算，再汇总种子波动；它们是按细胞汇总结果。"
            "训练重代入成绩、AUC、specificity、TP/FP/FN/TN、真实/预测阳性比例与覆盖率都保存在对应 CSV。此次没有新 OOF；训练成绩不是 OOF，内部验证是明确的开发集。"
        )
        add("### 外部 TMA 分组")
        note(
            "performance_" + mode,
            f"split={TEST},group_type=context,context=TMA1_CD8/TMA2_CD8/TMA3_CD8",
            "f1/precision/recall/roc_auc/average_precision/fp/fn_mean及_std",
        )
        part = perf.loc[
            perf.split.eq(TEST)
            & perf.group_type.eq("context")
            & (perf.threshold_mode.eq(mode) | perf.model_family.eq("GMM"))
        ]
        part = display(
            part,
            ["f1", "precision", "recall", "roc_auc", "average_precision", "fp", "fn"],
        )
        add(
            markdown(
                part,
                [
                    ("context", "TMA"),
                    ("model", "模型"),
                    ("f1_display", "F1"),
                    ("precision_display", "Precision"),
                    ("recall_display", "Recall"),
                    ("average_precision_display", "AP"),
                    ("roc_auc_display", "AUC"),
                    ("fp_display", "FP"),
                    ("fn_display", "FN"),
                ],
            )
        )

    add("## 6. 配对消融、架构与逐组织芯差异")
    note(
        "paired_deltas",
        f"split={TEST},group_type=split；两套threshold_mode分别标记",
        "delta_f1/average_precision/tp/fp/fn_mean及_std",
    )
    shown = paired.loc[paired.split.eq(TEST) & paired.group_type.eq("split")].copy()
    for metric in ["delta_f1", "delta_tp", "delta_fp", "delta_fn"]:
        shown[metric + "_display"] = shown.apply(
            lambda row: mean_sd(row, metric, 4 if metric == "delta_f1" else 1), axis=1
        )
    add(
        markdown(
            shown,
            [
                ("comparison", "比较"),
                ("threshold_mode", "门限"),
                ("delta_f1_display", "ΔF1"),
                ("delta_tp_display", "ΔTP"),
                ("delta_fp_display", "ΔFP"),
                ("delta_fn_display", "ΔFN"),
            ],
        )
    )
    add(
        "同特征 TabM−HGB 辨别学习器差异；B−A 衡量形态组合，C−B 衡量空间组合。"
        "不能把 F1 改善直接归因于某一特征或污染消除。GMM 为无监督参考，新模型使用训练标签，监督条件不同。"
    )
    note(
        "unit_equal",
        f"split={TEST},metric=f1；先逐芯等权再种子汇总",
        "unit_equal_mean_mean/std,n_finite_units_mean,n_units_no_positive_predictions_mean",
    )
    part = units.loc[units.split.eq(TEST) & units.metric.eq("f1")].copy()
    part["model"] = part.apply(model_label, axis=1)
    part["macro_f1"] = part.apply(lambda row: mean_sd(row, "unit_equal_mean"), axis=1)
    add(
        markdown(
            part.round(3),
            [
                ("model", "模型"),
                ("threshold_mode", "门限"),
                ("macro_f1", "逐芯等权F1"),
                ("n_finite_units_mean", "有效芯数"),
                ("n_units_no_positive_predictions_mean", "无阳性预测芯数均值"),
            ],
        )
    )
    note(
        "lowest_cores",
        "external, fixed_0.5；每个组合按三种子平均F1最低三个组织芯",
        "group_id,context,f1/precision/recall_mean及_std",
    )
    add(
        markdown(
            display(low, ["f1", "precision", "recall"]),
            [
                ("model", "模型"),
                ("context", "TMA"),
                ("group_id", "组织芯"),
                ("f1_display", "F1"),
                ("precision_display", "Precision"),
                ("recall_display", "Recall"),
            ],
        )
    )
    add(
        "完整分布见 metrics_all.csv、unit_performance_ranked.csv、paired_comparisons.csv。"
        "无真实阳性时 recall/AP=NaN；无阳性预测时 precision=0；真值与预测都无阳性时 F1=NaN。"
        "宏平均只使用相应指标有限的单位并记录数量，不把 NaN 填零。未做细胞级 bootstrap 或将细胞作为独立样本的显著性检验。"
    )

    add("## 7. 弱阳性、接触侧亮信号代理与案例")
    note(
        "risk_strata",
        "group_type=split,stratum=weak_positive/contact_bright_proxy_negative；按来源和门限分别比较",
        "n_cells_mean,recall_mean/std,false_positive_rate_mean/std,fp_mean,fn_mean",
    )
    part = risks.loc[
        risks.group_type.eq("split")
        & risks.split.isin([VALID, TEST])
        & risks.model_family.eq("TabM")
        & risks.stratum.isin(["weak_positive", "contact_bright_proxy_negative"])
    ]
    add(
        markdown(
            display(part, ["recall", "false_positive_rate"]).round(2),
            [
                ("split", "来源"),
                ("threshold_mode", "门限"),
                ("model", "模型"),
                ("stratum", "分层"),
                ("n_cells_mean", "细胞数"),
                ("recall_display", "Recall"),
                ("false_positive_rate_display", "FPR"),
            ],
        )
    )
    for mode in MODES:
        base = risks.loc[
            risks.model_family.eq("TabM")
            & risks.split.eq(TEST)
            & risks.group_type.eq("split")
            & risks.threshold_mode.eq(mode)
        ]

        def difference(stratum, metric):
            b = base.loc[base.stratum.eq(stratum) & base.feature_set.eq("B")]
            c = base.loc[base.stratum.eq(stratum) & base.feature_set.eq("C")]
            require(len(b) == len(c) == 1, "Missing risk comparison: " + stratum)
            return c.iloc[0][metric + "_mean"] - b.iloc[0][metric + "_mean"]

        dfpr = difference("contact_bright_proxy_negative", "false_positive_rate")
        dr = difference("weak_positive", "recall")
        add(
            f"- 外部 {mode} 的 C−B：接触侧亮信号阴性代理组 ΔFPR={fmt(dfpr)}；弱阳性 ΔRecall={fmt(dr)}。"
            + (
                "该代理组平均假阳性率下降。"
                if np.isfinite(dfpr) and dfpr < 0
                else "未观察到该代理组平均假阳性率下降。"
                if np.isfinite(dfpr)
                else "该代理组没有有效分母，FPR变化不可评估。"
            )
            + (
                "同时出现弱阳性平均召回下降，应保留这一代价。"
                if np.isfinite(dr) and dr < 0
                else "还需结合逐种子、逐切片表判断稳定性。"
                if np.isfinite(dr)
                else "弱阳性召回变化不可评估。"
            )
        )
    add(
        "弱阳性与接触侧亮信号沿用旧实验原始强度列和原训练分位数，显式 audit_raw_* 映射保证新旧风险群组相同；这些审计列不进入模型。FPR=FP/真实阴性数。"
        "弱阳性依据 raw whole_mean <= 训练正类 q25，跨图像亮度尺度会影响落入该组的比例，不能等同统一生物学弱表达。"
        "这些是图像属性代理，不证明真实邻居污染，因此不能据 FPR 下降声称已确认减少“邻近亮细胞导致”的假阳性。"
    )
    add(
        f'实际导出 {review["n_cases"]} 个开发案例，数量与选择原因见 development_review/case_index.csv、case_availability.csv，邻居记录见 neighbor_context.csv。'
        "只使用训练/内部验证图像、种子0、固定门限，优先展示 B→C 纠正与新增错误，全部案例保留原始像素 NPZ。"
        "候选类别不等于污染机制；说明性抽样不能估算总体错误率，原人工标签未经过新增病理专家复核。"
    )

    add("## 8. 缺失、数值分辨率与成员分歧")
    note("missingness", "每个split全部实例，含无标签；下表各split缺失率最高五项", "n,n_missing,missing_rate")
    top = (
        missing.sort_values("missing_rate", ascending=False)
        .groupby("split", group_keys=False)
        .head(5)
    )
    add(
        markdown(
            top.round({"missing_rate": 5}),
            [
                ("split", "来源"),
                ("feature", "特征"),
                ("n", "全部实例"),
                ("n_missing", "缺失数"),
                ("missing_rate", "缺失率"),
            ],
        )
    )
    add(
        "上述分母不是有标签训练细胞数。逐单位/真值缺失与 QC 见 feature_missingness_by_unit_class.csv、qc_by_split_class.csv。"
        "插补后能够评分不表示原观测完整，未依据缺失直接赋阴性。"
    )
    note(
        "numeric_resolution",
        "is_hu_moment=True，仅有标签训练细胞",
        "raw_nunique_train,transformed_float32_nunique_train,unique_retention_ratio,resolution_note",
    )
    hu = numeric.loc[numeric.is_hu_moment.eq(True)]
    add(
        markdown(
            hu.round({"unique_retention_ratio": 5}),
            [
                ("feature", "Hu特征"),
                ("raw_nunique_train", "原始不同值数"),
                ("transformed_float32_nunique_train", "转换后不同值数"),
                ("unique_retention_ratio", "保留比例"),
                ("resolution_note", "诊断"),
            ],
        )
    )
    add(
        "官方 QT 固定绝对尺度拟合噪声可能压缩高阶 Hu moments 的分辨率；compute_bins 成功只说明仍有不同值。"
        "本轮仅记录 float32 诊断，没有因验证表现改归一化、伪造分箱或重选特征。"
    )
    note(
        "disagreement",
        f"split={TEST},group_type=split,true_class=all",
        "std_mean_mean/std,std_p95_mean/std",
    )
    part = disagreement.loc[disagreement.split.eq(TEST)].copy()
    part["model"] = part.apply(model_label, axis=1)
    part["mean_sd"] = part.apply(lambda row: mean_sd(row, "std_mean"), axis=1)
    part["p95_sd"] = part.apply(lambda row: mean_sd(row, "std_p95"), axis=1)
    add(
        markdown(
            part, [("model", "模型"), ("mean_sd", "逐细胞成员SD均值"), ("p95_sd", "逐细胞成员SD p95")]
        )
    )
    add("每个 TabM 的全部成员阳性概率随逐细胞表保留。分歧不是已校准不确定性，不据此自动拒判或删除细胞。")

    add("## 9. 资源、交付与复现")
    note(
        "resources",
        "逐canonical模型；下表为各组合三种子均值",
        "training_optimization_seconds,validation_seconds,inference_seconds,training_peak_phase_rss_GiB,training_rss_increase_GiB,training_peak_gpu_allocated_GiB",
    )
    cols = [
        "training_optimization_seconds",
        "validation_seconds",
        "inference_seconds",
        "training_peak_phase_rss_GiB",
        "training_rss_increase_GiB",
        "training_peak_gpu_allocated_GiB",
    ]
    part = (
        resources.groupby(["model_family", "feature_set"], as_index=False)[cols]
        .mean()
        .round(3)
    )
    add(
        markdown(
            part,
            [
                ("model_family", "模型"),
                ("feature_set", "组合"),
                ("training_optimization_seconds", "优化秒"),
                ("validation_seconds", "验证秒"),
                ("inference_seconds", "全量推理秒"),
                ("training_peak_phase_rss_GiB", "阶段RSS峰GiB"),
                ("training_rss_increase_GiB", "RSS增量GiB"),
                ("training_peak_gpu_allocated_GiB", "GPU分配峰GiB"),
            ],
        )
    )
    add(
        f'硬件记录：`{runtime["hardware"]}`。TabM 用 GPU，HGB 用同机 CPU，因此是实际配置的资源对照，不是等算力速度排名。'
        "TabM 验证时间包含每 epoch 监测，HGB 为验证推理。阶段 RSS 以 psutil 采样，含开始时驻留数据，故另报基线和增量；"
        "不以进程历史 ru_maxrss 充当当前模型峰值。GPU 指标是分配内存，不是显卡总占用。"
    )
    add(
        f"服务器运行根目录：`{root}`。权重绝对路径、SHA256 与门限见 report_model_artifacts.csv；"
        "预处理位于 models/preprocessor.joblib，完整逐细胞文件位于 predictions/，含统一键、原标签、分组、评分、两套预测和成员输出。"
        "版本见 RUN_CONFIG.json；上游 commit 见 report_provenance.csv；本项目源码以 results/code_sha256.csv 和报告 manifest 标识，"
        "不捏造未记录的本地 Git commit。"
    )
    add(
        "复现已有冻结流程（入口校验数据/代码/配置；新科学重复运行应使用新目录保留旧输出）：\n\n```bash\n"
        f'R="{root}"\nPY="{runtime["runtime_executable"]}"\n'
        'flock -n "$R/../.cd8_robust_gpu.lock" "$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage all --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc\n'
        '"$PY" "$R/scripts/audit_numeric_resolution.py" --run-dir "$R"\n'
        '"$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage validate\n'
        '"$PY" "$R/scripts/write_tabm_report.py" --run-dir "$R"\n```\n'
        "数值诊断已经存在时跳过对应命令，该脚本拒绝覆盖历史诊断。"
    )
    add(
        "新表推理调用示例（C 只是接口示例，不表示按测试结果推荐它）：\n\n```bash\n"
        'flock -n "$R/../.cd8_robust_gpu.lock" "$PY" "$R/scripts/predict_table.py" --run-dir "$R" --model-id TabM_C_s0 '
        "--input /path/to/keyed_features.parquet --output /path/to/new_predictions.parquet\n```\n"
        "输入必须具有显式键、normalization_method、calibration_valid，以及同一 E1_outer_P1_P99 协议下从归一化像素重新提取、尚未经过训练 QT 变换的特征；插补和 QT 由已保存的训练预处理器完成。已存在输出拒绝覆盖。"
    )
    add(
        "PNG 和文字可编辑 SVG 位于 figures/zh、figures/en；源数值位于 figures/source_data；"
        "各split及TMA分组的完整 ROC/PR 和 threshold 数组位于 curves/。曲线采用 sklearn 的 >= 门限约定，正式分类表使用预定严格 >。"
    )
    add("## 10. 限制")
    add(
        "内部 WSI 用于开发，外部 TMA 在模型和门限冻结后评价。WSI 数量有限，种子波动不是患者级不确定性。"
        "扩张核掩码、参照缺失和像素尺度限制膜定位与背景机制解释；弱阳性/接触侧代理仍需独立人工复核。"
        "C 的 63 项空间特征来自本轮归一化像素重提，19 项形态沿用原分割与原几何计算；空间特征含固定方位的十二扇区值，B/C 形态还含 Orientation。此次没有新增旋转稳定性检验，不能继承 E1–E4 的旋转不变性结论。"
        "本轮只评价 PhenoBIC CD8，不扩展 marker，也不声称封存 Pan-M 的泛化表现。"
    )
    commit = protocol["official_commit"]
    add(
        f"官方来源：[TabM](https://github.com/yandex-research/tabm/tree/{commit})；"
        f"[训练示例](https://github.com/yandex-research/tabm/blob/{commit}/example.ipynb)；"
        f"[论文实验说明](https://github.com/yandex-research/tabm/blob/{commit}/paper/README.md)。"
    )
    outputs["ANALYSIS_REPORT.md"] = ("\n\n".join(sections) + "\n").encode("utf-8")
    outputs["results/report_tables.csv"] = (
        pd.DataFrame(registry).to_csv(index=False, lineterminator="\n").encode("utf-8")
    )
    inputs = {rel: digest(root / rel) for rel in paths.values()}
    inputs.update(
        {"results/" + name: digest(root / "results" / name) for name in tables}
    )
    inputs["data/risk_strata.json"] = risk_hash
    inputs["development_review/case_index.csv"] = digest(
        root / "development_review/case_index.csv"
    )
    manifest = dict(
        status="complete",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__),
        inputs=inputs,
        outputs={
            name: hashlib.sha256(content).hexdigest()
            for name, content in outputs.items()
        },
        no_fitting_or_model_selection=True,
        PanM_accessed=False,
    )
    manifest_path = root / "REPORT_GENERATION_MANIFEST.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        require(
            old["inputs"] == inputs
            and old["outputs"] == manifest["outputs"]
            and old["script_sha256"] == manifest["script_sha256"],
            "Existing report manifest differs; preserve previous outputs and use a deliberate new run/path",
        )
    for rel, content in outputs.items():
        path = root / rel
        require(
            not path.exists() or path.read_bytes() == content,
            "Refusing overwrite of a different report artifact: " + str(path),
        )
    for rel, content in outputs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    if not manifest_path.exists():
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            dict(
                status="complete",
                report=str(root / "ANALYSIS_REPORT.md"),
                table_registry=str(root / "results/report_tables.csv"),
                n_report_tables=len(registry),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    build_report(parser.parse_args().run_dir.resolve())
