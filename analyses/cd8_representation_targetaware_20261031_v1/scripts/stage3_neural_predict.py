"""Full external neural inference only after development selection is locked."""
import json, time
import numpy as np, torch
from experiment_common import *
from execution_guards import signature_for, validate_score, save_score, BudgetLedger
from stage2_neural import Network, Crops, predict, tabular_matrix, GROUPS, seed_all

_LEDGER = None


def main():
    global _LEDGER
    seed_all(0)
    frozen = json.loads((RUN / "models/MODEL_SELECTION_FROZEN.json").read_text())
    d, man = load()
    assert man["feature_sha256"] == frozen["feature_sha256"]
    train = (d.split == TRAIN).to_numpy()
    test = (d.split == TEST).to_numpy()
    y = d.ground_truth_binary.to_numpy(np.int64)
    A, prep = tabular_matrix(d, man["feature_sets"]["H2"], train)
    assert prep == json.loads(
        (RUN / "data/NEURAL_TABULAR_PREPROCESSOR.json").read_text()
    )
    ledger = BudgetLedger(RUN / "logs/GPU_BUDGET.json")
    _LEDGER = ledger
    budget = ledger.data
    torch.set_num_threads(4)
    for seed in [0, 1, 2]:
        for group in GROUPS:
            name = "N_" + group + "_s" + str(seed)
            out = RUN / "predictions" / (name + "_all.npy")
            sig = signature_for(
                d, man, RUN / "models/MODEL_SELECTION_FROZEN.json", name
            )
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
            if budget["consumed_seconds"] >= budget["hard_cap_seconds"]:
                raise RuntimeError(
                    "Frozen GPU budget reached before complete external inference"
                )
            path = RUN / "models" / (name + ".pt")
            assert sha(path) == frozen["model_file_sha256"][path.name]
            b = torch.load(path, map_location="cpu")
            model = Network(A.shape[1]).cuda()
            model.load_state_dict(b["state_dict"])
            ds = Crops(A, y, group)
            torch.cuda.reset_peak_memory_stats()
            ledger.begin(name + "_external")
            part, elapsed = predict(
                model, ds, d.index[test].to_numpy(), group, ledger=ledger
            )
            scores = np.load(RUN / "predictions" / (name + "_development.npy"))
            scores[test] = part[test]
            assert np.isfinite(scores).all()
            save_score(
                out,
                metaout,
                scores,
                sig,
                {
                    "rows": len(scores),
                    "external_rows": int(test.sum()),
                    "inference_seconds": elapsed,
                    "score_sha256": sha(out),
                    "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "model_sha256": sha(path),
                },
            )
            ledger.finish("running")
            print("NEURAL_EXTERNAL_SCORED", name, "seconds", elapsed, flush=True)
            del model, ds
            torch.cuda.empty_cache()
    for group in GROUPS:
        name = "ENS_" + group
        s = np.mean(
            np.stack(
                [
                    np.load(
                        RUN
                        / "predictions"
                        / ("N_" + group + "_s" + str(seed) + "_all.npy")
                    )
                    for seed in [0, 1, 2]
                ]
            ),
            axis=0,
            dtype=np.float64,
        )
        dev = np.load(RUN / "predictions" / (name + "_development.npy"))
        assert np.array_equal(s[~test], dev[~test])
        save_score(
            RUN / "predictions" / (name + "_all.npy"),
            RUN / "predictions" / (name + "_external_complete.json"),
            s,
            signature_for(d, man, RUN / "models/MODEL_SELECTION_FROZEN.json", name),
        )
    ledger.data["status"] = "completed"
    ledger.persist()
    savej(
        RUN / "predictions/NEURAL_EXTERNAL_COMPLETE.json",
        {
            "individual_models": 15,
            "ensembles": 5,
            "rows": len(d),
            "budget": budget,
            "all_finite": True,
            "selection_frozen_sha256": sha(RUN / "models/MODEL_SELECTION_FROZEN.json"),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _LEDGER is not None:
            _LEDGER.abort(error)
        raise
