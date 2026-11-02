from continuation_core import *
from calibration_common import aggregate, baseline_thresholds
import argparse, traceback

LEDGER = None


def status(stage, **kw):
    atomic_json(
        R / "STATUS_CONTINUATION.json", {"stage": stage, "time": time.time(), **kw}
    )
    print(stage, json.dumps(kw), flush=True)


def smoke(d, ledger, cutoffs):
    ledger.begin("smoke:new_continuation")
    idx = d.index[d.split == cc.TRAIN].to_numpy()[:8]
    state = torch.load(BASE / "models/A_s0.pt", map_location="cpu")["state_dict"]
    checks = []
    for v in VARIANTS:
        cc.seed_all(0)
        m = network(v, state).cuda()
        ds = ContinuationDataset(d, v, True, 0, cutoffs)
        batch = next(iter(cc.loader(ds, idx, batch=8, workers=0)))
        x, aug, a, y, rows, ok = batch
        out = m(x.cuda(), a.cuda(), "A")
        loss = nn.functional.cross_entropy(out, y.cuda())
        loss.backward()
        assert torch.isfinite(loss) and all(
            torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
        )
        checks.append(
            {
                "variant": v,
                "shape": list(x.shape),
                "finite_loss": float(loss),
                "nominal_parameters": sum(p.numel() for p in m.parameters()),
                "optimizer_steps": 0,
            }
        )
        del m
        torch.cuda.empty_cache()
    atomic_json(R / "results/CONTINUATION_SMOKE.json", checks)
    ledger.finish("running")


def cv(d, ledger):
    assert (R / "results/STAGE0_COMPLETE.json").exists() and (
        R / "models/ZERO_LABEL_SELECTION_FROZEN.json"
    ).exists()
    assert not (
        R / "models/CONTINUATION_CV_SELECTION.json"
    ).exists(), "CV already complete"
    baseline = pd.read_csv(R / "results/STAGE0_FULL_METRICS.csv")
    has_sensitivity = bool(
        (
            (baseline.model == "ENS_A")
            & (baseline.unit == "pooled")
            & ((baseline.delta_F1 < -0.003) | (baseline.delta_AP < -0.003))
        ).any()
    )
    decision = {
        "stage0_evidence_sensitivity": has_sensitivity,
        "remaining_seconds": ledger.remaining,
        "requires_remaining_at_least": 7200,
        "authorized_finite_new_continuation_not_old_refit": True,
        "plan_sha256": file_sha(R / "TRAINING_CONDITIONAL_PLAN.json"),
        "time": time.time(),
    }
    decision["run"] = has_sensitivity and ledger.remaining >= 7200
    atomic_json(R / "models/CONTINUATION_TRIGGER.json", decision)
    if not decision["run"]:
        atomic_json(
            R / "models/CONTINUATION_CV_SELECTION.json",
            {"selected": None, "reason": "trigger_not_met", "decision": decision},
        )
        return
    cutoffs = train_cutoffs(d)
    smoke(d, ledger, cutoffs)
    rows = []
    units = sorted(d.loc[d.split == cc.TRAIN, "image_id"].unique())
    for fold, unit in enumerate(units):
        fitidx = d.index[(d.split == cc.TRAIN) & (d.image_id != unit)].to_numpy()
        hold = d.index[(d.split == cc.TRAIN) & (d.image_id == unit)].to_numpy()
        for v in VARIANTS:
            name = "NEWCV_" + v + "_f" + str(fold)
            status("CV_continuation", model=name)
            fit(
                d,
                v,
                "CV_A_f" + str(fold),
                name,
                0,
                fitidx,
                20000,
                2,
                4,
                ledger,
                cutoffs,
            )
            s = predict_new(d, name, v, hold, ledger, "heldout")
            rows.append(
                {
                    "variant": v,
                    "fold": fold,
                    "heldout_training_WSI": unit,
                    **cc.metrics(d.loc[hold, "ground_truth_binary"], s[hold], 0.5),
                }
            )
            pd.DataFrame(rows).to_csv(R / "results/CONTINUATION_CV.csv", index=False)
    tab = pd.DataFrame(rows)
    control = tab[tab.variant == "control"].set_index("fold")
    decisions = {}
    for v in VARIANTS[1:]:
        a = tab[tab.variant == v].set_index("fold")
        da = a.AP - control.AP
        df = a.F1 - control.F1
        ok = bool(
            df.mean() >= -1e-12
            and da.mean() >= -0.003 - 1e-12
            and (da >= -0.003 - 1e-12).all()
        )
        decisions[v] = {
            "passed": ok,
            "mean_AP": float(a.AP.mean()),
            "mean_F1": float(a.F1.mean()),
            "delta_AP_vs_control": float(da.mean()),
            "delta_F1_vs_control": float(df.mean()),
            "per_fold_delta_AP": da.to_dict(),
        }
    passed = [v for v in VARIANTS[1:] if decisions[v]["passed"]]
    selected = None
    if passed:
        best = max(decisions[v]["mean_AP"] for v in passed)
        close = [v for v in passed if decisions[v]["mean_AP"] >= best - 0.001]
        selected = sorted(
            close, key=lambda v: (-decisions[v]["mean_F1"], VARIANTS.index(v))
        )[0]
    atomic_json(
        R / "models/CONTINUATION_CV_SELECTION.json",
        {
            "selected": selected,
            "decisions": decisions,
            "source_WSI_folds": 3,
            "independent_patients_unknown": True,
            "external_used": False,
            "internal_used": False,
            "plan_sha256": file_sha(R / "TRAINING_CONDITIONAL_PLAN.json"),
            "code_sha256": file_sha(__file__),
            "time": time.time(),
        },
    )
    status("CV_continuation_complete", selected=selected, decisions=decisions)


def final(d, ledger):
    select = json.loads((R / "models/CONTINUATION_CV_SELECTION.json").read_text())
    v = select["selected"]
    if v is None:
        atomic_json(
            R / "models/CONTINUATION_INTERNAL_GATE.json",
            {
                "passed": False,
                "reason": "no_CV_candidate_passed",
                "selected": None,
                "final_fits": 0,
            },
        )
        return
    train = d.index[d.split == cc.TRAIN].to_numpy()
    dev = d.index[d.split == cc.DEV].to_numpy()
    cutoffs = train_cutoffs(d)
    ss = {}
    rows = []
    thresholds = {}
    for seed in range(3):
        for variant in ["control", v]:
            name = "NEW_" + variant + "_s" + str(seed)
            status("final_continuation", model=name)
            fit(
                d,
                variant,
                "A_s" + str(seed),
                name,
                seed,
                train,
                50000,
                2,
                6,
                ledger,
                cutoffs,
            )
    for seed in range(3):
        for variant in ["control", v]:
            name = "NEW_" + variant + "_s" + str(seed)
            status("final_internal_prediction", model=name)
            s = predict_new(d, name, variant, dev, ledger, "internal")
            ss[name] = s
            t = cc.threshold_fpr(d, s, dev)
            thresholds[name] = {"fixed_0.5": 0.5, "dev_FPR01_each_WSI": t}
    for variant in ["control", v]:
        name = "NEW_ENS_" + variant
        s = np.mean(
            [ss["NEW_" + variant + "_s" + str(k)] for k in range(3)],
            axis=0,
            dtype=np.float32,
        )
        ss[name] = s
        path = R / "predictions" / (name + "__internal.npy")
        atomic_npy(path, s)
        atomic_json(
            path.with_suffix(".json"),
            {"sha256": file_sha(path), "seed_probability_mean": True, "n": len(dev)},
        )
        thresholds[name] = {
            "fixed_0.5": 0.5,
            "dev_FPR01_each_WSI": cc.threshold_fpr(d, s, dev),
        }
    for name, s in ss.items():
        for mode, t in thresholds[name].items():
            rows.extend(
                aggregate(
                    d,
                    dev,
                    s,
                    float(np.float32(t)),
                    {"model": name, "threshold_mode": mode},
                )
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(R / "results/CONTINUATION_INTERNAL_METRICS.csv", index=False)
    atomic_json(
        R / "models/CONTINUATION_THRESHOLDS_FROZEN.json",
        {
            "records": thresholds,
            "operator": ">",
            "max_per_internal_WSI_FPR": 0.01,
            "external_used": False,
            "time": time.time(),
            "selection_sha256": file_sha(R / "models/CONTINUATION_CV_SELECTION.json"),
        },
    )
    a = tab[(tab.level == "pooled") & (tab.threshold_mode == "dev_FPR01_each_WSI")]
    deltas = []
    for seed in range(3):
        x = a[a.model == "NEW_" + v + "_s" + str(seed)].iloc[0]
        b = a[a.model == "NEW_control_s" + str(seed)].iloc[0]
        deltas.append(
            {
                "seed": seed,
                "delta_F1": float(x.F1 - b.F1),
                "delta_AP": float(x.AP - b.AP),
            }
        )
    ensemble_new = a[a.model == "NEW_ENS_" + v].iloc[0]
    ensemble_control = a[a.model == "NEW_ENS_control"].iloc[0]
    ensemble_deltas = {
        "delta_F1": float(ensemble_new.F1 - ensemble_control.F1),
        "delta_AP": float(ensemble_new.AP - ensemble_control.AP),
    }
    passed = (
        all(x["delta_AP"] >= -0.003 - 1e-12 for x in deltas)
        and np.mean([x["delta_F1"] for x in deltas]) >= -1e-12
        and ensemble_deltas["delta_F1"] >= -1e-12
        and ensemble_deltas["delta_AP"] >= -0.003 - 1e-12
    )
    atomic_json(
        R / "models/CONTINUATION_INTERNAL_GATE.json",
        {
            "passed": bool(passed),
            "selected": v,
            "deltas": deltas,
            "ensemble_deltas": ensemble_deltas,
            "gate_amendment_sha256": file_sha(R / "TRAINING_GATES_AMENDMENT.json"),
            "final_fits": 6,
            "no_replacement_candidate_if_failed": True,
            "time": time.time(),
        },
    )
    status("continuation_internal_gate", passed=bool(passed), selected=v)


def external(d, ledger):
    gate = json.loads((R / "models/CONTINUATION_INTERNAL_GATE.json").read_text())
    v = gate["selected"]
    if not gate["passed"]:
        atomic_json(
            R / "results/CONTINUATION_EXTERNAL_SKIPPED.json",
            {"reason": "development_gate_failed", "gate": gate},
        )
        return
    thresholds = json.loads(
        (R / "models/CONTINUATION_THRESHOLDS_FROZEN.json").read_text()
    )["records"]
    dev = d.index[d.split == cc.DEV].to_numpy()
    test = d.index[d.split == cc.TEST].to_numpy()
    rows = []
    perturb = []
    for scenario in ["gain_0.5", "zero_0.95", "noise_high"]:
        for variant in ["control", v]:
            ss = []
            for seed in range(3):
                name = "NEW_" + variant + "_s" + str(seed)
                status("newmodel_internal_perturbation", model=name, scenario=scenario)
                s = predict_new(
                    d, name, variant, dev, ledger, "internal_" + scenario, scenario
                )
                ss.append(s)
                t = float(np.float32(thresholds[name]["dev_FPR01_each_WSI"]))
                perturb.extend(
                    aggregate(
                        d,
                        dev,
                        s,
                        t,
                        {
                            "model": name,
                            "scenario": scenario,
                            "threshold_mode": "dev_FPR01_each_WSI",
                        },
                    )
                )
            name = "NEW_ENS_" + variant
            s = np.mean(ss, axis=0, dtype=np.float32)
            perturb.extend(
                aggregate(
                    d,
                    dev,
                    s,
                    float(np.float32(thresholds[name]["dev_FPR01_each_WSI"])),
                    {
                        "model": name,
                        "scenario": scenario,
                        "threshold_mode": "dev_FPR01_each_WSI",
                    },
                )
            )
    pd.DataFrame(perturb).to_csv(
        R / "results/CONTINUATION_PERTURBATION_METRICS.csv", index=False
    )
    pt = pd.DataFrame(perturb)
    diffs = []
    for scenario in ["gain_0.5", "zero_0.95", "noise_high"]:
        for member in ["s0", "s1", "s2", "ENS"]:
            chosen = "NEW_ENS_" + v if member == "ENS" else "NEW_" + v + "_" + member
            control = "NEW_ENS_control" if member == "ENS" else "NEW_control_" + member
            aa = pt[
                (pt.model == chosen)
                & (pt.scenario == scenario)
                & (pt.level == "pooled")
            ].iloc[0]
            bb = pt[
                (pt.model == control)
                & (pt.scenario == scenario)
                & (pt.level == "pooled")
            ].iloc[0]
            diffs.append(
                {
                    "scenario": scenario,
                    "member": member,
                    "delta_AP_vs_matched_control": float(aa.AP - bb.AP),
                    "delta_F1_descriptive": float(aa.F1 - bb.F1),
                }
            )
    robust = all(q["delta_AP_vs_matched_control"] >= -0.003 - 1e-12 for q in diffs)
    atomic_json(
        R / "models/CONTINUATION_PERTURBATION_GATE.json",
        {
            "passed": robust,
            "selected": v,
            "delta_rows": diffs,
            "evaluated_before_new_model_external_prediction": True,
            "gate_amendment_sha256": file_sha(R / "TRAINING_GATES_AMENDMENT.json"),
            "time": time.time(),
        },
    )
    if not robust:
        atomic_json(
            R / "results/CONTINUATION_EXTERNAL_SKIPPED.json",
            {
                "reason": "predeclared_synthetic_AP_gate_failed",
                "selected": v,
                "diffs": diffs,
                "external_model_predictions": 0,
            },
        )
        status("continuation_stopped_at_synthetic_gate", selected=v)
        return

    for variant in ["control", v]:
        ss = []
        for seed in range(3):
            name = "NEW_" + variant + "_s" + str(seed)
            status("newmodel_external_prediction", model=name)
            s = predict_new(d, name, variant, test, ledger, "external")
            ss.append(s)
            for mode, t in thresholds[name].items():
                rows.extend(
                    aggregate(
                        d,
                        test,
                        s,
                        float(np.float32(t)),
                        {
                            "model": name,
                            "threshold_mode": mode,
                            "external_known_exploratory": True,
                        },
                    )
                )
        name = "NEW_ENS_" + variant
        s = np.mean(ss, axis=0, dtype=np.float32)
        path = R / "predictions" / (name + "__external.npy")
        atomic_npy(path, s)
        atomic_json(
            path.with_suffix(".json"),
            {"sha256": file_sha(path), "seed_probability_mean": True, "n": len(test)},
        )
        for mode, t in thresholds[name].items():
            rows.extend(
                aggregate(
                    d,
                    test,
                    s,
                    float(np.float32(t)),
                    {
                        "model": name,
                        "threshold_mode": mode,
                        "external_known_exploratory": True,
                    },
                )
            )
    pd.DataFrame(rows).to_csv(
        R / "results/CONTINUATION_EXTERNAL_METRICS.csv", index=False
    )
    atomic_json(
        R / "results/CONTINUATION_COMPLETE.json",
        {
            "selected": v,
            "external_reselection": False,
            "full_internal_and_external": True,
            "budget": ledger.data,
        },
    )
    status("continuation_complete")


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--phase", choices=["cv", "final", "external"], required=True)
        a = ap.parse_args()
        torch.set_num_threads(4)
        d, _ = cc.load_data()
        LEDGER = BudgetLedger(R / "logs/NEURAL_BUDGET.json", cap=28800)
        {"cv": cv, "final": final, "external": external}[a.phase](d, LEDGER)
    except BaseException as e:
        if LEDGER:
            LEDGER.abort(e)
        atomic_json(
            R / "logs/CONTINUATION_FAILURE.json",
            {"error": str(e), "traceback": traceback.format_exc()},
        )
        raise
