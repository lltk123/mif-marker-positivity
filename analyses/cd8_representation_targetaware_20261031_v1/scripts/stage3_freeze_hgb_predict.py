"""Freeze development-only candidate selection, then score HGB external cells."""
import os

os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import json, time, joblib
import numpy as np, pandas as pd
from experiment_common import *
from stage1_hgb import transform
from execution_guards import signature_for, validate_score, save_score

GROUPS = [
    "image",
    "image_target",
    "image_target_other",
    "tabular_only",
    "image_target_other_tabular",
]


def main():
    assert (RUN / "models/HGB_COMPLETE.json").exists() and (
        RUN / "models/NEURAL_COMPLETE.json"
    ).exists()
    d, man = load()
    dev = (d.split == DEV).to_numpy()
    test = (d.split == TEST).to_numpy()
    selection = RUN / "models/MODEL_SELECTION_FROZEN.json"
    if not selection.exists():
        records = []
        for name in ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]:
            meta = json.loads((RUN / "models" / (name + ".json")).read_text())
            meta["inference_seconds_for_tie"] = meta["development_inference_seconds"]
            records.append(meta)
        for group in GROUPS:
            names = ["N_" + group + "_s" + str(seed) for seed in [0, 1, 2]]
            members = [
                json.loads((RUN / "models" / (name + ".json")).read_text())
                for name in names
            ]
            s = np.mean(
                np.stack(
                    [
                        np.load(RUN / "predictions" / (name + "_development.npy"))
                        for name in names
                    ]
                ),
                axis=0,
                dtype=np.float64,
            )
            name = "ENS_" + group
            np.save(RUN / "predictions" / (name + "_development.npy"), s)
            ts, grid = thresholds(d.loc[dev], s)
            pd.DataFrame(grid).to_csv(
                RUN / "results" / (name + "_development_threshold_grid.csv"),
                index=False,
            )
            meta = {
                "model": name,
                "family": "neural_ensemble",
                "group": group,
                "members": names,
                "n_seeds": 3,
                "development_macro_AP": macro_ap(d.loc[dev], s),
                "mean_individual_seed_development_macro_AP": float(
                    np.mean([x["development_macro_AP"] for x in members])
                ),
                "thresholds": ts,
                "active_parameters": sum(x["active_parameters"] for x in members),
                "nominal_parameters": sum(x["nominal_parameters"] for x in members),
                "inference_seconds_for_tie": sum(
                    x["development_inference_seconds"] for x in members
                ),
                "deployment_candidate": True,
                "extra_fits": 0,
                "member_model_sha256": {x["model"]: x["model_sha256"] for x in members},
            }
            savej(RUN / "models" / (name + ".json"), meta)
            records.append(meta)
        candidates = [x for x in records if x["deployment_candidate"]]
        best = max(x["development_macro_AP"] for x in candidates)
        eligible = [x for x in candidates if x["development_macro_AP"] >= best - 0.005]
        order = ["H0", "H1", "H2", "H3", "H4", "H5"] + ["ENS_" + g for g in GROUPS]
        chosen = sorted(
            eligible,
            key=lambda x: (
                x["active_parameters"],
                x["inference_seconds_for_tie"],
                order.index(x["model"]),
            ),
        )[0]
        result = {
            "selected_model": chosen["model"],
            "selection_population": DEV,
            "selection_rule": "within .005 of maximal internal equal-WSI AP, prefer fewer active parameters then lower measured inference time then fixed order",
            "candidates": [
                {
                    k: x[k]
                    for k in [
                        "model",
                        "development_macro_AP",
                        "active_parameters",
                        "inference_seconds_for_tie",
                        "deployment_candidate",
                    ]
                }
                for x in records
            ],
            "protocol_sha256": sha(RUN / "PROTOCOL.json"),
            "resource_addendum_sha256": sha(RUN / "RESOURCE_ADDENDUM.json"),
            "feature_sha256": man["feature_sha256"],
            "external_metrics_used": False,
            "frozen_time": time.time(),
            "model_file_sha256": {
                p.name: sha(p)
                for p in (RUN / "models").iterdir()
                if p.suffix in [".pt", ".joblib"]
            },
            "thresholds": {x["model"]: x["thresholds"] for x in records},
        }
        savej(selection, result)
        pd.DataFrame(result["candidates"]).to_csv(
            RUN / "results/DEVELOPMENT_CANDIDATE_SELECTION.csv", index=False
        )
        print(
            "SELECTION_FROZEN",
            json.dumps(
                {"selected_model": chosen["model"], "external_metrics_used": False}
            ),
            flush=True,
        )
    frozen = json.loads(selection.read_text())
    assert frozen["feature_sha256"] == man["feature_sha256"]
    for name in ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]:
        out = RUN / "predictions" / (name + "_all.npy")
        sig = signature_for(d, man, selection, name)
        metaout = RUN / "predictions" / (name + "_external_complete.json")
        if out.exists():
            valid, reason = validate_score(out, metaout, len(d), sig)
            if valid:
                continue
            raise RuntimeError(
                "Existing score requires audit before replacement: "
                + name
                + " "
                + reason
            )
        modelpath = RUN / "models" / (name + ".joblib")
        assert sha(modelpath) == frozen["model_file_sha256"][modelpath.name]
        b = joblib.load(modelpath)
        X = d.loc[test, b["features"]].to_numpy(float)
        A = transform(X, b["preprocessing"])
        s = np.load(RUN / "predictions" / (name + "_development.npy"))
        started = time.time()
        s[test] = b["estimator"].predict_proba(A)[:, 1]
        elapsed = time.time() - started
        assert np.isfinite(s).all()
        save_score(
            out,
            metaout,
            s,
            sig,
            {
                "rows": len(s),
                "external_rows": int(test.sum()),
                "inference_seconds": elapsed,
                "score_sha256": sha(out),
                "model_sha256": sha(modelpath),
                "selection_frozen_sha256": sha(selection),
            },
        )
        print("HGB_EXTERNAL_SCORED", name, flush=True)
    savej(
        RUN / "predictions/HGB_EXTERNAL_COMPLETE.json",
        {"n_models": 7, "selection_frozen_sha256": sha(selection)},
    )


if __name__ == "__main__":
    main()
