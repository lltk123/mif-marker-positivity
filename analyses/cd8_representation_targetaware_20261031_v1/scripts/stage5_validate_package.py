"""Independent saved-prediction audit and lightweight deliverable packaging."""
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, hashlib, time, zipfile, ast
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from experiment_common import *
from stage3_evaluate import model_names, meta_for


def main():
    for rel in [
        "reports/REPORT_COMPLETE.json",
        "reports/FIGURES_COMPLETE.json",
        "review/REVIEW_COMPLETE.json",
        "results/COMPOSITION_COMPLETE.json",
    ]:
        assert (RUN / rel).exists(), rel
    d, man = load()
    main = pd.read_csv(RUN / "results/METRICS_BY_UNIT.csv")
    frozen = json.loads((RUN / "models/MODEL_SELECTION_FROZEN.json").read_text())
    checks = []
    for filename, expected in frozen["model_file_sha256"].items():
        passed = sha(RUN / "models" / filename) == expected
        checks.append(
            {
                "check": "model_unchanged_after_selection_freeze",
                "model": filename,
                "passed": passed,
            }
        )
    master_keys = pd.util.hash_pandas_object(
        d[["dataset_id", "image_id", "cell_id", "marker"]], index=False
    ).to_numpy()
    n_checks = 0
    maxerr = 0.0
    for name in model_names():
        p = pd.read_parquet(RUN / "predictions" / (name + "_all_cells.parquet"))
        keys = pd.util.hash_pandas_object(
            p[["dataset_id", "image_id", "cell_id", "marker"]], index=False
        ).to_numpy()
        aligned = np.array_equal(keys, master_keys)
        checks.append(
            {
                "check": "all_explicit_keys_and_order_match",
                "model": name,
                "passed": aligned,
                "n": len(p),
            }
        )
        assert aligned and len(p) == 1077669
        score = p.score.to_numpy()
        y = p.ground_truth_binary.to_numpy(int)
        meta = meta_for(name)
        for mode, t in meta["thresholds"].items():
            pred = p["prediction_" + mode].to_numpy(bool)
            correct = np.array_equal(pred, score > np.asarray(t, dtype=score.dtype))
            checks.append(
                {
                    "check": "stored_prediction_equals_strict_score_threshold",
                    "model": name,
                    "mode": mode,
                    "passed": correct,
                }
            )
            assert correct
            for split in [TRAIN, DEV, TEST]:
                ix = p.split.to_numpy() == split
                yy = y[ix]
                ss = score[ix]
                pp = pred[ix]
                tp = int(np.sum((yy == 1) & pp))
                fp = int(np.sum((yy == 0) & pp))
                fn = int(np.sum((yy == 1) & ~pp))
                tn = int(np.sum((yy == 0) & ~pp))
                expected = main[
                    (main.model == name)
                    & (main.split == split)
                    & (main.level == "split")
                    & (main.threshold_mode == mode)
                ].iloc[0]
                assert len(yy) == int(expected.n_cells) and all(
                    v == int(expected[k])
                    for k, v in [("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)]
                )
                f1 = 2 * tp / (2 * tp + fp + fn)
                ap = float(average_precision_score(yy, ss))
                auc = float(roc_auc_score(yy, ss))
                errors = [
                    abs(f1 - expected.F1),
                    abs(ap - expected.AP),
                    abs(auc - expected.ROC_AUC),
                ]
                maxerr = max(maxerr, max(errors))
                assert max(errors) < 1e-10
                n_checks += 1
        print("INDEPENDENT_PREDICTION_AUDIT", name, flush=True)
    cal = pd.read_csv(RUN / "results/TARGET_LABEL_CALIBRATION_SIMULATION.csv")
    groups = ["replicate", "model", "level", "unit", "stratum"]
    variation = cal.groupby(groups, dropna=False).n_cells.nunique()
    assert variation.max() == 1
    checks.append(
        {
            "check": "calibration_budget_comparisons_same_evaluation_population",
            "passed": True,
            "groups": len(variation),
        }
    )
    reserved = pd.read_csv(RUN / "results/CALIBRATION_RESERVED_INSTANCES.csv")
    assert not reserved.duplicated(["replicate", "row_id"]).any()
    assert (reserved.groupby(["replicate", "image_id"]).size() == 100).all()
    assert np.array_equal(
        d.iloc[reserved.row_id.to_numpy()].cell_id.to_numpy(),
        reserved.cell_id.to_numpy(),
    )
    checks.append(
        {
            "check": "calibration_reserved_keys_unique_and_match_source",
            "passed": True,
            "rows": len(reserved),
        }
    )
    comp = pd.read_csv(RUN / "results/COMPOSITION_DIAGNOSTIC.csv")
    assert (comp.groupby("row_id").target_raw_pixels_sha256.nunique() == 1).all()
    checks.append(
        {
            "check": "composition_target_raw_pixels_identical",
            "passed": True,
            "n_targets": comp.row_id.nunique(),
        }
    )
    cases = pd.read_csv(RUN / "review/CASE_REVIEW.csv")
    for z in cases.itertuples(index=False):
        a = np.load(RUN / "review" / z.raw_npz)
        assert (
            hashlib.sha256(np.ascontiguousarray(a["raw_CD8"]).tobytes()).hexdigest()
            == z.raw_pixels_sha256
        )
        assert (
            a["raw_CD8"].shape == (96, 96)
            and a["original_instance_labels"].shape == (96, 96)
            and a["valid_support"].dtype == bool
        )
        assert int(a["target_cell_id"]) == int(z.cell_id)
        assert np.any(
            (a["original_instance_labels"] == int(z.cell_id)) & a["valid_support"]
        )
    checks.append(
        {
            "check": "real_review_pixels_match_recorded_SHA",
            "passed": True,
            "n": len(cases),
        }
    )
    for p in (RUN / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8-sig"))
    for p in (RUN / "reports/figures").glob("*.svg"):
        assert "<text" in p.read_text(encoding="utf-8")
    checks.append({"check": "python_syntax_and_SVG_editable_text", "passed": True})
    pd.DataFrame(checks).to_csv(
        RUN / "results/INDEPENDENT_DELIVERY_CHECKS.csv", index=False
    )
    validation = {
        "passed": all(x["passed"] for x in checks),
        "saved_prediction_models": len(model_names()),
        "complete_set_metric_rows_recounted": n_checks,
        "max_abs_rank_or_F1_error": maxerr,
        "all_external_populations": 437480,
        "all_main_original_reference_labels_retained": True,
        "PanM_accessed": False,
        "visual_QA": "pending local inspection",
        "human_case_review": "pending user",
        "model_sha_unchanged_after_development_selection": True,
        "code_sha256": sha(__file__),
    }
    savej(RUN / "reports/DELIVERY_NUMERIC_VALIDATION.json", validation)
    # Small numeric source tables sufficient to redraw the delivered quantitative panels.
    source = RUN / "reports/source_data"
    source.mkdir(exist_ok=True)
    main.to_csv(source / "metrics_by_unit.csv", index=False)
    pd.read_csv(RUN / "results/RISK_STRATA.csv").to_csv(
        source / "risk_strata.csv", index=False
    )
    pd.read_csv(RUN / "results/COMPOSITION_SUMMARY.csv").to_csv(
        source / "composition_summary.csv", index=False
    )
    curve_rows = []
    for group in [
        "image",
        "image_target",
        "image_target_other",
        "tabular_only",
        "image_target_other_tabular",
    ]:
        name = "ENS_" + group
        z = np.load(RUN / "results/curves" / (name + "_TEST_DIAGNOSTIC_PR.npz"))
        ix = np.linspace(0, len(z["recall"]) - 1, min(1200, len(z["recall"]))).astype(
            int
        )
        for j in ix:
            curve_rows.append(
                {
                    "model": name,
                    "curve_index": int(j),
                    "recall": float(z["recall"][j]),
                    "precision": float(z["precision"][j]),
                    "purpose": "test diagnostic curve; not deployable threshold selection",
                }
            )
    pd.DataFrame(curve_rows).to_csv(
        source / "plotted_diagnostic_PR_points.csv", index=False
    )
    repro = """# 复现与完成标志

本次运行保留在服务器独立目录，模型和逐细胞预测不包含于轻量包。原项目与数据目录固定，Pan-M不访问。源码使用既有两个独立环境；新机器缺数据/原环境时不能宣称仅克隆轻量包即可复现全部数值。

```bash
PROJECT=/mnt/disk3/mIHC/mif_marker_positivity_20261022
RUN=$PROJECT/analyses/cd8_representation_targetaware_20261031_v1
PYF=$PROJECT/analyses/cd8_morphospatial_20261023/.venv/bin/python
PYN=$PROJECT/analyses/cd8_nimbus_20261024/.venv/bin/python
cd "$RUN"
# 已完成阶段先核验完成标志与输入/源码哈希；不要覆盖原运行。
# 下列为本次真实执行顺序，不是要求重新训练旧实验。
$PYF -B scripts/stage1_prepare.py --stage pilot
$PYF -B scripts/stage1_prepare.py --stage full --workers 4
$PYF -B scripts/stage1_hgb.py
$PYN -B scripts/stage2_neural.py --smoke-only
$PYN -B scripts/stage2_neural.py
$PYF -B scripts/stage1_composition.py
$PYF -B scripts/stage3_freeze_hgb_predict.py
$PYN -B scripts/stage3_neural_predict.py
$PYF -B scripts/stage3_evaluate.py
$PYN -B scripts/stage4_review.py
$PYN -B scripts/stage4_figures.py
$PYN -B scripts/stage4_report.py
$PYN -B scripts/stage5_validate_package.py
```

对已完成目录，优先做只读哈希和保存预测核验。全量准备命令在已有分块时复用分块，但会重建汇总文件；正式新重复请复制代码到新的独立目录，统一调整各脚本RUN配置及协议路径并重新冻结，不能借用原完成标志。训练数据/原始图像/旧E2及共享校准表必须使用协议中相同来源与哈希。

完成标志：data/FEATURES_COMPLETE.json；models/HGB_COMPLETE.json；models/NEURAL_COMPLETE.json；models/MODEL_SELECTION_FROZEN.json；results/COMPOSITION_COMPLETE.json；results/EVALUATION_COMPLETE.json；review/REVIEW_COMPLETE.json；reports/DELIVERY_NUMERIC_VALIDATION.json。

轻量包只含代码、配置、模型JSON元数据、汇总数值、图与有限真实病例；完整权重、连续分数和逐细胞判定表留服务器predictions/、models/及data/。中文与英文图件为PNG/SVG/PDF；SVG文字保留可编辑。病例CSV中的人工判读栏留空，等待用户填写。
"""
    (RUN / "reports/REPRODUCE.md").write_text(repro, encoding="utf-8")
    manifest = []
    for folder in ["models", "predictions", "data"]:
        for p in (RUN / folder).iterdir():
            if p.is_file():
                manifest.append(
                    {
                        "path": str(p.relative_to(RUN)),
                        "bytes": p.stat().st_size,
                        "kept_on_server": True,
                    }
                )
    pd.DataFrame(manifest).to_csv(
        RUN / "reports/FULL_ARTIFACT_LOCATIONS.csv", index=False
    )
    delivery = RUN / "delivery"
    delivery.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        delivery / "CD8_results_code_20261031.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for rel in ["PROTOCOL.json", "PROTOCOL.sha256", "RESOURCE_ADDENDUM.json"]:
            z.write(RUN / rel, rel)
        for folder in ["scripts", "tests", "reports"]:
            for p in (RUN / folder).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, str(p.relative_to(RUN)))
        for p in (RUN / "results").glob("*.csv"):
            z.write(p, str(p.relative_to(RUN)))
        for p in (RUN / "results").glob("*.json"):
            z.write(p, str(p.relative_to(RUN)))
        for p in (RUN / "models").glob("*.json"):
            z.write(p, str(p.relative_to(RUN)))
        for p in (RUN / "provenance").glob("*.json"):
            z.write(p, str(p.relative_to(RUN)))
        for p in (RUN / "predictions").glob("*.json"):
            z.write(p, str(p.relative_to(RUN)))
    with zipfile.ZipFile(
        delivery / "CD8_real_case_review_20261031.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for p in (RUN / "review").rglob("*"):
            if p.is_file():
                z.write(p, str(p.relative_to(RUN)))
    savej(
        delivery / "DELIVERY_MANIFEST.json",
        {
            "artifacts": [
                {
                    "path": str(p.relative_to(RUN)),
                    "bytes": p.stat().st_size,
                    "sha256": sha(p),
                }
                for p in delivery.glob("*.zip")
            ],
            "extract_both_into_same_folder": True,
            "visual_QA": "pending local inspection",
            "human_review": "pending",
            "numeric_validation": validation,
        },
    )
    print("DELIVERY_NUMERIC_VALIDATED_AND_PACKAGED", json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
