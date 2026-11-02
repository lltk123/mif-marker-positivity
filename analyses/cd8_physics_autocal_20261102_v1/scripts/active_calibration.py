"""Only new random5/10 and active5/10/20. Historical random20/50/100 read unchanged."""
from calibration_common import *


def main():
    complete = R / "results/FEWSHOT_COMPLETE.json"
    if complete.exists():
        q = json.loads(complete.read_text())
        assert q["code_sha256"] == file_sha(__file__)
        assert all(file_sha(R / name) == h for name, h in q["output_sha256"].items())
        assert file_sha(BASE / "predictions/ENS_A_all.npy") == q["native_score_sha256"]
        assert (
            file_sha(BASE / "results/CALIBRATION_RESERVED_INSTANCES.csv")
            == q["old_reserved_sha256"]
        )
        print("FEWSHOT_ALREADY_COMPLETE_VERIFIED_NO_REFIT", flush=True)
        return
    started = time.time()
    d, _ = cc.load_data()
    native = verify_scores(d, "ENS_A")
    z = logits(native)
    y = d.ground_truth_binary.to_numpy(int)
    t = float(np.float32(baseline_thresholds()["ENS_A"]["dev_FPR01_each_WSI"]))
    pools = reserved(d)
    rows = []
    parameters = []
    choices = []
    integrity = []
    for rep, (reserve, hold) in pools.items():
        for strategy, budgets in [
            ("random", [5, 10]),
            ("active_boundary", [5, 10, 20]),
        ]:
            for budget in budgets:
                adjusted = native.astype(float).copy()
                for uid, ix in reserve.items():
                    # ix already ordered by frozen BLAKE2b random rank, tie-break inherits that order.
                    ordered = (
                        ix
                        if strategy == "random"
                        else ix[np.argsort(abs(z[ix] - logit(t)), kind="stable")]
                    )
                    cal = ordered[:budget]
                    b, fallback = intercept(z[cal], y[cal])
                    ii = d.index[(d.split == cc.TEST) & (d.image_id == uid)].to_numpy()
                    adjusted[ii] = shifted(native[ii], b)
                    assert not hold[cal].any()
                    if b == 0:
                        assert np.array_equal(
                            adjusted[ii] > t, native[ii] > np.asarray(t, native.dtype)
                        )
                    parameters.append(
                        {
                            "replicate": rep,
                            "strategy": strategy,
                            "budget": budget,
                            "image_id": uid,
                            "intercept": b,
                            "one_class_fallback": fallback,
                            "selected_positive_n": int(y[cal].sum()),
                            "reserved_excluded": 100,
                        }
                    )
                    for rank, i in enumerate(cal, 1):
                        choices.append(
                            {
                                "replicate": rep,
                                "strategy": strategy,
                                "budget": budget,
                                "image_id": uid,
                                "row_id": int(i),
                                "cell_id": int(d.loc[i, "cell_id"]),
                                "selection_rank": rank,
                                "sampling_used_labels": False,
                            }
                        )
                extra = {
                    "model": "ENS_A",
                    "replicate": rep,
                    "strategy": strategy,
                    "budget": budget,
                    "threshold_mode": "core_intercept_from_frozen_dev_FPR01",
                    "selection_bias": "deterministic_boundary_sample; noIPW; calibration_assumption_not_population_unbiased",
                }
                idx = np.flatnonzero(hold)
                rows.extend(aggregate(d, idx, adjusted, t, extra))
                integrity.append(
                    {
                        "replicate": rep,
                        "strategy": strategy,
                        "budget": budget,
                        "evaluation_n": len(idx),
                        "excluded_n": 4200,
                        "no_selected_in_evaluation": True,
                        "native_fallback_decisions_exact": True,
                    }
                )
                # Small case comparison later uses these params, no giant prediction duplicates.
                print("ACTIVE_DONE", rep, strategy, budget, flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(R / "results/NEW_FEWSHOT_METRICS.csv", index=False)
    pd.DataFrame(parameters).to_csv(
        R / "results/NEW_FEWSHOT_PARAMETERS.csv", index=False
    )
    pd.DataFrame(choices).to_csv(
        R / "results/NEW_FEWSHOT_SELECTED_IDS.csv", index=False
    )
    pd.DataFrame(integrity).to_csv(R / "results/NEW_FEWSHOT_ISOLATION.csv", index=False)
    # Historical rows are copied exactly, never fitted again.
    old = pd.read_csv(BASE / "results/TARGET_LABEL_CALIBRATION.csv")
    old[(old.model == "ENS_A") & (old.budget_per_core.isin([0, 20, 50, 100]))].to_csv(
        R / "results/REUSED_RANDOM_CALIBRATION_ROWS.csv", index=False
    )
    metrics = [
        "F1",
        "AP",
        "P",
        "R",
        "FPR",
        "weak_recall",
        "core_F1_median",
        "core_F1_IQR",
        "core_prevalence_MAE",
    ]
    summary = (
        tab[tab.level == "pooled"]
        .groupby(["strategy", "budget"])[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(x) for x in summary.columns]
    summary.reset_index().to_csv(R / "results/NEW_FEWSHOT_SUMMARY.csv", index=False)
    atomic_json(
        R / "results/FEWSHOT_COMPLETE.json",
        {
            "new_strategy_budget_replicate_groups": 15,
            "old_random_fits_repeated": False,
            "evaluation_n_per_replicate": 433280,
            "reserved_excluded_per_replicate": 4200,
            "model": "frozen3seedprobabilityENS_A_not_seed_metric_mean",
            "seconds_cpu_wall": time.time() - started,
            "old_reserved_sha256": file_sha(
                BASE / "results/CALIBRATION_RESERVED_INSTANCES.csv"
            ),
            "old_calibration_sha256": file_sha(
                BASE / "results/TARGET_LABEL_CALIBRATION.csv"
            ),
            "output_sha256": {
                str(p.relative_to(R)): file_sha(p)
                for p in [
                    R / "results" / n
                    for n in [
                        "NEW_FEWSHOT_METRICS.csv",
                        "NEW_FEWSHOT_PARAMETERS.csv",
                        "NEW_FEWSHOT_SELECTED_IDS.csv",
                        "NEW_FEWSHOT_ISOLATION.csv",
                        "REUSED_RANDOM_CALIBRATION_ROWS.csv",
                        "NEW_FEWSHOT_SUMMARY.csv",
                    ]
                ]
            },
            "native_score_sha256": file_sha(BASE / "predictions/ENS_A_all.npy"),
            "code_sha256": file_sha(__file__),
            "protocol_sha256": file_sha(R / "PROTOCOL.json"),
        },
    )


if __name__ == "__main__":
    main()
