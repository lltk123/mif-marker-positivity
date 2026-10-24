"""Synthetic evaluation fixtures; these are tests, not experimental model performance."""
from common import *
import evaluate as ev


def main():
    rows = []

    def check(name, ok):
        rows.append(dict(test=name, passed=bool(ok), synthetic=True))
        assert ok, name

    a = ev.metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], 0.5)
    check(
        "known_confusion_and_ranking",
        all(a[k] == 1 for k in ["TP", "FP", "FN", "TN"])
        and a["F1"] == 0.5
        and a["ROC_AUC"] == 0.75
        and abs(a["AP"] - 5 / 6) < 1e-12,
    )
    a = ev.metrics([1, 0], [0.5, 0.5], 0.5)
    check("strict_greater_than_gate", a["TP"] == 0 and a["FN"] == 1 and a["FP"] == 0)
    a = ev.metrics([0, 0], [0.1, 0.2], 0.5)
    check(
        "zero_positive_unit_NA",
        np.isnan(a["F1"])
        and np.isnan(a["recall"])
        and np.isnan(a["AP"])
        and a["precision"] == 0,
    )
    a = ev.metrics([0, 1], [0.2, np.nan], 0.5)
    check(
        "failed_observation_not_success_subset",
        a["coverage"] == 0.5 and np.isnan(a["F1"]) and a["n_failed"] == 1,
    )
    sandbox = R / "tests/metric_sandbox"
    (sandbox / "results").mkdir(parents=True, exist_ok=True)
    ev.R = sandbox
    rows2 = []
    for unit in ["A", "B", "C"]:
        rows2.extend(
            dict(image_id=unit, ground_truth_binary=y, score=s)
            for y, s in zip([1, 1, 1, 0, 0, 0], [0.46, 0.8, 0.9, 0.51, 0.55, 0.95])
        )
    threshold, score = ev.choose_threshold(pd.DataFrame(rows2), "synthetic_tie_test")
    check(
        "symmetric_threshold_tie_chooses_higher",
        threshold == 0.55 and abs(score - 2 / 3) < 1e-12,
    )
    pd.DataFrame(rows).to_csv(R / "tests/evaluation_contract_checks.csv", index=False)
    print("EVALUATION_CONTRACT_PASSED", len(rows), flush=True)


if __name__ == "__main__":
    main()
