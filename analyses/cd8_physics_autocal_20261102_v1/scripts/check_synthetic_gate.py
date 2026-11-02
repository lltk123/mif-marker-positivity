"""Integration guard: bad synthetic AP must prevent any external prediction."""
from pathlib import Path
import tempfile, json
from unittest.mock import patch
import pandas as pd, numpy as np
import continuation_run as cr
from execution_guards import atomic_json, file_sha


def main():
    original = cr.R
    labels = np.tile([0, 1], 24)
    d = pd.DataFrame(
        {
            "split": [cr.cc.DEV] * 36 + [cr.cc.TEST] * 12,
            "ground_truth_binary": labels,
            "image_id": ["u1"] * 12 + ["u2"] * 12 + ["u3"] * 12 + ["external"] * 12,
        }
    )
    calls = []

    def predict(d, name, variant, idx, ledger, tag, scenario="clean"):
        calls.append(tag)
        if tag == "external":
            raise AssertionError("ExternalpredictioncalledafterbadsyntheticAP")
        score = np.where(d.ground_truth_binary.to_numpy() == 1, 0.9, 0.1).astype(
            np.float32
        )
        return 1 - score if variant == "physics" else score

    def aggregate(d, idx, s, t, extra):
        return [
            {
                **extra,
                "level": "pooled",
                "unit": "all",
                **cr.cc.metrics(
                    d.loc[idx, "ground_truth_binary"].to_numpy(), s[idx], t
                ),
            }
        ]

    with tempfile.TemporaryDirectory(prefix="cd8_gate_check_") as temp:
        root = Path(temp)
        for n in ["models", "results", "logs"]:
            (root / n).mkdir()
        atomic_json(
            root / "models/CONTINUATION_INTERNAL_GATE.json",
            {"passed": True, "selected": "physics"},
        )
        records = {}
        for v in ["control", "physics"]:
            for seed in range(3):
                records["NEW_" + v + "_s" + str(seed)] = {
                    "fixed_0.5": 0.5,
                    "dev_FPR01_each_WSI": 0.5,
                }
            records["NEW_ENS_" + v] = {"fixed_0.5": 0.5, "dev_FPR01_each_WSI": 0.5}
        atomic_json(
            root / "models/CONTINUATION_THRESHOLDS_FROZEN.json", {"records": records}
        )
        (root / "TRAINING_GATES_AMENDMENT.json").write_bytes(
            (original / "TRAINING_GATES_AMENDMENT.json").read_bytes()
        )
        with patch.object(cr, "R", root), patch.object(
            cr, "predict_new", predict
        ), patch.object(cr, "aggregate", aggregate):
            cr.external(d, None)
        decision = json.loads(
            (root / "models/CONTINUATION_PERTURBATION_GATE.json").read_text()
        )
        assert not decision["passed"]
        assert len(calls) == 18 and all(t.startswith("internal_") for t in calls)
        assert (root / "results/CONTINUATION_EXTERNAL_SKIPPED.json").exists()
    print(
        json.dumps(
            {
                "passed": True,
                "synthetic_internal_predictions_mocked": 18,
                "external_prediction_calls": 0,
                "new_model_fits": 0,
                "controller_sha256": file_sha(original / "scripts/continuation_run.py"),
                "test_sha256": file_sha(__file__),
            }
        ),
        flush=True,
    )
    print("SYNTHETIC_GATE_INTEGRATION_PASS_NO_EXTERNAL_CALLS", flush=True)


if __name__ == "__main__":
    main()
