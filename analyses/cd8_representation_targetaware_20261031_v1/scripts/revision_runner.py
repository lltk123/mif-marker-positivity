"""Post-training correctness revision. Never fits models or changes frozen thresholds."""
import os, sys, time, json, hashlib, shutil, subprocess, zipfile, difflib, traceback
from pathlib import Path
import numpy as np, pandas as pd
from runtime_paths import ROOT as P, RUN as R, FEATURE_PYTHON as PF, NEURAL_PYTHON as PN

V = R / "revisions/correctness_001"
sys.path.insert(0, str(R / "scripts"))
from execution_guards import *
from experiment_common import load, TRAIN, DEV, TEST
from stage2_neural import Network, seed_all, GROUPS
from stage3_evaluate import calibration, model_names
import torch


def step(script, python=PN):
    with (V / (script + ".log")).open("a") as log:
        z = subprocess.run(
            [str(python), "-B", "-u", str(R / "scripts" / script)],
            cwd=R,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if z.returncode:
        raise RuntimeError(script + " exited " + str(z.returncode))


def main():
    start = time.time()
    fpath = R / "models/MODEL_SELECTION_FROZEN.json"
    frozen = json.loads(fpath.read_text())
    before = file_sha(fpath)
    assert before == "288f2444df40fc1c0d82fa0105d044cffda74cd15bc39d922a8a207d869327a3"
    for n, h in frozen["model_file_sha256"].items():
        assert file_sha(R / "models" / n) == h, n
    d, man = load()
    assert file_sha(R / "data/features.parquet") == man["feature_sha256"]
    keys = canonical_keys_hash(d)
    audit = []
    for name in model_names():
        if name == "GMM":
            continue
        p = R / "predictions" / (name + "_all.npy")
        a = np.load(p, allow_pickle=False)
        assert a.shape == (len(d),) and np.isfinite(a).all()
        q = pd.read_parquet(R / "predictions" / (name + "_all_cells.parquet"))
        assert canonical_keys_hash(q) == keys and np.array_equal(q.score.to_numpy(), a)
        m = json.loads((R / "models" / (name + ".json")).read_text())
        for mode, t in m["thresholds"].items():
            assert np.array_equal(
                q["prediction_" + mode].to_numpy(), a > np.asarray(t, dtype=a.dtype)
            )
        mp = R / "predictions" / (name + "_external_complete.json")
        old = json.loads(mp.read_text()) if mp.exists() else {}
        h = file_sha(p)
        if old.get("score_sha256"):
            assert h == old["score_sha256"]
        sig = signature_for(d, man, fpath, name)
        save_score(p, mp, a, sig, old)
        assert file_sha(p) == h and validate_score(p, mp, len(d), sig)[0]
        audit.append(
            {
                "model": name,
                "rows": len(a),
                "four_keys_in_original_parquet_verified": True,
                "predictions_identical": True,
                "score_sha256": h,
            }
        )
        print("SCORE_VERIFIED", name, flush=True)
    pd.DataFrame(audit).to_csv(
        R / "results/CORRECTNESS_PREDICTION_AUDIT.csv", index=False
    )
    # Reconstruct provenance using the original deterministic algorithm, without any fit.
    init = []
    schedule = []
    groups = [
        g.index.to_numpy()
        for _, g in d[d.split == TRAIN].groupby("image_id", sort=True)
    ]
    for seed in [0, 1, 2]:
        for group in GROUPS:
            seed_all(seed)
            m = Network(56)
            digest = hashlib.sha256(
                b"".join(
                    v.detach().cpu().numpy().tobytes() for v in m.state_dict().values()
                )
            ).hexdigest()
            meta = json.loads(
                (R / "models" / ("N_" + group + "_s" + str(seed) + ".json")).read_text()
            )
            init.append(
                {
                    "seed": seed,
                    "group": group,
                    "hash": digest,
                    "nominal_parameters": sum(p.numel() for p in m.parameters()),
                    "reconstructed_after_training": True,
                    "contemporaneous_record": False,
                }
            )
            for epoch in range(meta["epochs_run"]):
                rng = np.random.default_rng(seed * 10000 + epoch)
                ix = np.concatenate(
                    [
                        rng.choice(g, 16667 if k < 2 else 16666, replace=len(g) < 16667)
                        for k, g in enumerate(groups)
                    ]
                )
                rng.shuffle(ix)
                assert len(ix) == 50000 and d.iloc[ix].split.eq(TRAIN).all()
                schedule.append(
                    {
                        "seed": seed,
                        "group": group,
                        "epoch_zero_based": epoch,
                        "rows": len(ix),
                        "unique_rows": len(np.unique(ix)),
                        "row_index_sha256": hashlib.sha256(
                            np.asarray(ix, dtype="<i8").tobytes()
                        ).hexdigest(),
                        "reconstructed_after_training": True,
                    }
                )
    assert all(
        len({x["hash"] for x in init if x["seed"] == seed}) == 1 for seed in [0, 1, 2]
    )
    smoke = json.loads((R / "results/NEURAL_SMOKE_COMPLETE.json").read_text())
    assert all(x["initial_state_sha256"] == init[0]["hash"] for x in smoke)
    val = np.load(R / "data/neural_checkpoint_validation_indices.npy")
    expected = []
    for uid, g in d[d.split == DEV].groupby("image_id", sort=True):
        h = [
            int.from_bytes(
                hashlib.blake2b(
                    (str(uid) + "|" + str(int(c))).encode(), digest_size=8
                ).digest(),
                "little",
            )
            for c in g.cell_id
        ]
        expected.extend(
            g.index.to_numpy()[np.argsort(np.asarray(h, np.uint64))[:10000]]
        )
    assert np.array_equal(val, np.sort(expected)) and d.iloc[val].split.eq(DEV).all()
    pd.DataFrame(init).to_csv(
        R / "results/POSTHOC_INITIALIZATION_HASHES.csv", index=False
    )
    pd.DataFrame(schedule).to_csv(
        R / "results/POSTHOC_TRAINING_INDEX_HASHES.csv", index=False
    )
    atomic_json(
        R / "results/POSTHOC_SAMPLING_AUDIT.json",
        {
            "reconstructed_after_training": True,
            "not_contemporaneous_training_evidence": True,
            "seed0_matches_contemporaneous_smoke": True,
            "validation_rows": len(val),
            "validation_sha256": hashlib.sha256(val.tobytes()).hexdigest(),
            "external_rows_in_samples": 0,
            "groups_identical_per_seed_epoch": True,
        },
    )
    # Original metric tables remain intact; only calibration is recalculated.
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(R / "scripts"))
    with (V / "calibration.log").open("a") as log:
        z = subprocess.run(
            [
                str(PF),
                "-B",
                "-u",
                "-c",
                "from stage3_evaluate import calibration;from experiment_common import load;d,_=load();calibration(d)",
            ],
            env=env,
            cwd=R,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if z.returncode:
        raise RuntimeError("Calibration failed")
    ca = pd.read_csv(R / "results/CALIBRATION_NATIVE_THRESHOLD_AUDIT.csv")
    old = pd.read_csv(V / "original_results/TARGET_LABEL_CALIBRATION_SIMULATION.csv")
    new = pd.read_csv(R / "results/TARGET_LABEL_CALIBRATION_SIMULATION.csv")
    join = ["replicate", "model", "budget_per_core", "level", "unit", "stratum"]
    merged = old.merge(new, on=join, suffixes=("_old", "_new"), validate="one_to_one")
    diffs = {
        k: int((merged[k + "_old"].fillna(-1) != merged[k + "_new"].fillna(-1)).sum())
        for k in ["tp", "fp", "fn", "tn"]
    }
    budget = json.loads((R / "logs/GPU_BUDGET.json").read_text())
    train_seconds = sum(
        json.loads(p.read_text())["fit_and_development_seconds"]
        for p in (R / "models").glob("N_*.json")
    )
    complexity = []
    for name in model_names():
        if name == "GMM":
            continue
        meta = json.loads((R / "models" / (name + ".json")).read_text())
        complexity.append(
            {
                "model": name,
                "complexity_value": meta["active_parameters"],
                "complexity_unit": "tree_nodes"
                if name.startswith("H")
                else "active_scalar_weights",
                "cross_family_counts_comparable": False,
            }
        )
    pd.DataFrame(complexity).to_csv(
        R / "results/MODEL_COMPLEXITY_UNITS.csv", index=False
    )
    audit = {
        "revision": "correctness_001",
        "frozen_selection_sha256": before,
        "selected_model": frozen["selected_model"],
        "models_unchanged": True,
        "predictions_unchanged": True,
        "frozen_threshold_values_unchanged": True,
        "native_dtype_threshold_applied": True,
        "old_zero_budget_decisions_changed_sum_across_repeats": int(
            ca[ca.budget == 0].old_implementation_changed_decisions.sum()
        ),
        "old_all_calibration_decisions_changed_sum_across_repeats_budgets": int(
            ca.old_implementation_changed_decisions.sum()
        ),
        "changed_metric_rows_by_count": diffs,
        "zero_budget_and_one_class_fallback_exact": bool(ca.native_mapping_exact.all()),
        "neural_train_development_seconds": train_seconds,
        "total_recorded_neural_seconds_including_external": budget["consumed_seconds"],
        "below_8h": budget["consumed_seconds"] < 28800,
        "partial_run_or_timeout_observed": False,
        "retrained": False,
        "PanM_accessed": False,
        "posthoc_provenance_explicit": True,
        "notes": "Original frozen selection retained. Tree nodes and neural scalar weights are not comparable. Guard changes apply prospectively, not retroactively to completed training.",
    }
    atomic_json(R / "results/CORRECTNESS_REVISION_AUDIT.json", audit)
    for script in ["stage4_figures.py", "stage4_report.py"]:
        step(script)
    appendix = f"""
## 正确性修订与保留限制

本次修订未重新训练，22个模型权重、全部逐细胞分数、冻结候选 **{frozen['selected_model']}** 和阈值数值均保持不变。27组预测数组已逐一核验长度、有限值、四键顺序、输入与分数哈希，并与原逐细胞表一致；写入现采用原子替换和独立完成元数据。

校准评价统一使用原分数dtype对应的有效阈值，零标注预算与单类别回退均逐实例核验为原严格“>”判定。旧实现在零预算下造成的判定差异计数为 **{audit['old_zero_budget_decisions_changed_sum_across_repeats']}**（三次抽样重复累加，不是独立细胞数）。所有校准预算、图和报告已重算；各预算与未校准基线共同排除每core预留100例。

训练及完整内部推理累计 **{train_seconds:.2f}秒**，加外部推理为 **{budget['consumed_seconds']:.2f}秒**，未触及8小时。新的执行保护区分partial、budget_exhausted、completed；这些保护未用于本次已完成训练，不倒称为当时已有。

正式seed初始化和每epoch 50,000训练索引哈希是**训练后按原确定性算法重建**，不是当时记录。seed 0与当时冒烟初始化哈希一致；固定30,000内部验证索引与保存文件完全一致。训练抽样各seed/epoch不同，同seed同epoch各组相同；最终阈值和评价使用完整内部/外部集合。

原内部筛选规则已冻结并保留，但复杂度排序将HGB树节点数与神经标量权重数直接比较，两者量纲不同，存在跨模型族偏向；因此候选仅代表该预定规则的结果，不应解释为公平的模型复杂度最优。外部结果未用于重选。Pan-M保持封存，真实病例仍待人工复核。
"""
    with (R / "reports/REPORT_zh.md").open("a", encoding="utf-8") as f:
        f.write(appendix)
    import html

    p = R / "reports/REPORT_zh.html"
    p.write_text(
        p.read_text().replace(
            "</main>",
            "<section><h2>正确性修订</h2><pre>"
            + html.escape(appendix)
            + "</pre></section></main>",
        ),
        encoding="utf-8",
    )
    assert file_sha(fpath) == before
    for n, h in frozen["model_file_sha256"].items():
        assert file_sha(R / "models" / n) == h
    step("stage5_validate_package.py")
    # Preserve original packages; export corrected versions distinctly.
    for name in ["CD8_results_code_20261031.zip", "CD8_real_case_review_20261031.zip"]:
        p = R / "delivery" / name
        shutil.copy2(p, R / "delivery" / (p.stem + "_correctness001.zip"))
    audit["seconds"] = time.time() - start
    atomic_json(V / "REVISION_COMPLETE.json", audit)
    print("REVISION_COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        atomic_json(
            V / "REVISION_FAILED.json",
            {"traceback": traceback.format_exc(), "time": time.time()},
        )
        raise
