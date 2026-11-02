from physics_common import *
import argparse, traceback

LEDGER = None


def status(stage, **kw):
    atomic_json(R / "STATUS.json", {"stage": stage, "time": time.time(), **kw})
    print(stage, json.dumps(kw), flush=True)


def main():
    global LEDGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "full"], required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    for name in [
        "predictions",
        "models",
        "data",
        "results",
        "reports",
        "provenance",
        "logs",
    ]:
        (R / name).mkdir(exist_ok=True)
    d, _ = cc.load_data()
    dev = d.index[d.split == cc.DEV].to_numpy()
    assert len(dev) == 539132
    if (R / "data/ZERO_THRESHOLDS.json").exists():
        cutoffs = json.loads((R / "data/ZERO_THRESHOLDS.json").read_text())
    else:
        status("preparing_full_internal_instance_pixel_quantiles")
        cutoffs = prepare(d)
    LEDGER = BudgetLedger(R / "logs/NEURAL_BUDGET.json", cap=28800)
    if args.phase == "pilot":
        identity = []
        pilot = stable_indices(d, cc.DEV, 512, 4401)
        assert len(pilot) == 1536
        atomic_npy(R / "data/PILOT_INDICES.npy", pilot)
        for seed in range(3):
            status("identity_pilot", seed=seed)
            a, sec = model_score(d, pilot, "clean", seed, LEDGER, "pilot", cutoffs)
            base = np.load(BASE / "predictions" / ("A_s" + str(seed) + "_all.npy"))
            meta = json.loads(
                (BASE / "predictions" / ("A_s" + str(seed) + "_all.json")).read_text()
            )
            assert (
                file_sha(BASE / "predictions" / ("A_s" + str(seed) + "_all.npy"))
                == meta["sha256"]
            )
            delta = float(abs(a[pilot] - base[pilot]).max())
            assert delta <= 1e-6
            threshold = json.loads(
                (BASE / "models/THRESHOLDS_FROZEN.json").read_text()
            )["records"][seed * 4]["dev_FPR01_each_WSI"]
            diff = int(
                (
                    (a[pilot] > np.float32(threshold))
                    != (base[pilot] > np.float32(threshold))
                ).sum()
            )
            assert diff == 0
            identity.append(
                {
                    "seed": seed,
                    "max_probability_difference": delta,
                    "changed_decisions": diff,
                    "seconds": sec,
                }
            )
        probe = stable_indices(d, cc.DEV, 4000, 4401)
        assert len(probe) == 12000
        atomic_npy(R / "data/TIMING_PROBE_INDICES.npy", probe)
        timings = []
        for scenario in ["gain_1.5", "noise_high", "zero_0.95"]:
            status("timing_probe", scenario=scenario)
            a, sec = model_score(d, probe, scenario, 0, LEDGER, "timing_probe", cutoffs)
            timings.append({"scenario": scenario, "n": len(probe), "seconds": sec})
        estimate = max(x["seconds"] for x in timings) * 539132 / 12000 * 48 * 1.25
        record = {
            "identity": identity,
            "timing_probes": timings,
            "conservative_full_stage0_seconds": estimate,
            "remaining_seconds": LEDGER.remaining,
            "within_budget_gate": estimate <= LEDGER.remaining * 0.75,
            "crop_proxy_not_whole_slide_simulation": True,
            "full_stage0_not_started": True,
        }
        atomic_json(R / "results/PILOT_COMPLETE.json", record)
        status("pilot_complete", **record)
        return
    assert json.loads((R / "results/PILOT_COMPLETE.json").read_text())[
        "within_budget_gate"
    ]
    rows = []
    clean = {
        ("A_s" + str(seed)): np.load(
            BASE / "predictions" / ("A_s" + str(seed) + "_all.npy")
        )
        for seed in range(3)
    }
    clean["ENS_A"] = np.load(BASE / "predictions/ENS_A_all.npy")
    rows.extend(stage0_metrics(d, dev, "clean", clean))
    for scenario, param in SCENARIOS:
        scores = {}
        for seed in range(3):
            status("stage0_full_internal_inference", scenario=scenario, seed=seed)
            a, sec = model_score(
                d, dev, scenario, seed, LEDGER, "full_internal", cutoffs
            )
            scores["A_s" + str(seed)] = a
        ens = np.mean(list(scores.values()), axis=0, dtype=np.float32)
        scores["ENS_A"] = ens
        p = R / "predictions" / ("ENS_A__" + scenario + "__full_internal.npy")
        atomic_npy(p, ens)
        atomic_json(
            p.with_suffix(".json"),
            {
                "sha256": file_sha(p),
                "n_scored": len(dev),
                "definition": "arithmetic mean of3seed probabilities",
                "protocol_sha256": file_sha(R / "PROTOCOL.json"),
            },
        )
        rows.extend(stage0_metrics(d, dev, scenario, scores))
        pd.DataFrame(rows).to_csv(R / "results/STAGE0_FULL_METRICS.csv", index=False)
    atomic_json(
        R / "results/STAGE0_COMPLETE.json",
        {
            "changed_scenarios": 16,
            "clean_scores_reused": True,
            "internal_cells": len(dev),
            "models_per_scenario": 3,
            "probability_ensembles": 16,
            "n_metric_rows": len(rows),
            "budget": LEDGER.data,
            "proxy_scope": "normalized crop-level technical transforms, not physical acquisition reconstruction",
        },
    )
    status("stage0_complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        if LEDGER is not None:
            LEDGER.abort(e)
        atomic_json(
            R / "logs/STAGE0_FAILURE.json",
            {"error": str(e), "traceback": traceback.format_exc()},
        )
        raise
