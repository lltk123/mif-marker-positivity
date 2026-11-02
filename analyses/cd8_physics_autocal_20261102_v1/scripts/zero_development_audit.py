"""Independent replay of frozen zero-label development decisions, without refitting."""
from calibration_common import *
from final_audit import independent_metrics


def main():
    started = time.time()
    d, _ = cc.load_data()
    idx = d.index[d.split == cc.DEV].to_numpy()
    y = d.ground_truth_binary.to_numpy(int)
    folds = json.loads((R / "provenance/ZERO_LABEL_SOURCE_FOLDS.json").read_text())
    tab = pd.read_csv(
        R / "results/ZERO_LABEL_DEV_METRICS.csv", float_precision="round_trip"
    )
    params = pd.read_csv(
        R / "results/ZERO_LABEL_DEV_CORRECTIONS.csv", float_precision="round_trip"
    )
    selection = json.loads((R / "models/ZERO_LABEL_SELECTION_FROZEN.json").read_text())
    checks = []
    observed = {}
    units = set(d.loc[idx, "image_id"])
    source_checks = []
    for name in ["A_s0", "A_s1", "A_s2", "ENS_A"]:
        thresholds = np.full(len(d), np.nan)
        ff = [f for f in folds if f["model"] == name]
        for f in ff:
            uid = f["heldout_WSI"]
            assert set(f["fit_units"]) == units - {uid}
            assert all(x["fit_unit"] in f["fit_units"] for x in f["regression_fit"])
            ii = d.index[(d.split == cc.DEV) & (d.image_id == uid)].to_numpy()
            thresholds[ii] = f["clean_threshold_from_fit_sources_only"]
            source_checks.append(
                {
                    "model": name,
                    "heldout": uid,
                    "fit_units": f["fit_units"],
                    "derived_perturbations_excluded": True,
                }
            )
        for method in ["spatial", "mixture", "ridge"]:
            failures = []
            for row in tab[
                (tab.model == name)
                & (tab.method == method)
                & (tab.unit == "pooled")
                & (tab.threshold_mode == "source_LOO_FPR01")
            ].itertuples():
                if row.scenario == "clean":
                    native = verify_scores(d, name)
                else:
                    p = (
                        R
                        / "predictions"
                        / (name + "__" + row.scenario + "__full_internal.npy")
                    )
                    assert (
                        file_sha(p)
                        == json.loads(p.with_suffix(".json").read_text())["sha256"]
                    )
                    native = np.load(p)
                adjusted = native.astype(np.float64).copy()
                for par in params[
                    (params.model == name)
                    & (params.method == method)
                    & (params.scenario == row.scenario)
                ].itertuples():
                    ii = d.index[
                        (d.split == cc.DEV) & (d.image_id == par.unit)
                    ].to_numpy()
                    if par.b != 0:
                        p = np.clip(
                            native[ii].astype(np.float64),
                            np.finfo(np.float64).eps,
                            1 - np.finfo(np.float64).eps,
                        )
                        z = np.log(p) - np.log1p(-p)
                        adjusted[ii] = 1 / (1 + np.exp(-(z + par.b)))
                a = independent_metrics(y[idx], adjusted[idx], thresholds[idx])
                base = independent_metrics(
                    y[idx], native[idx].astype(np.float64), thresholds[idx]
                )
                err = {k: abs(a[k] - getattr(row, k)) for k in a}
                err["delta_AP"] = abs(a["AP"] - base["AP"] - row.delta_AP)
                err["delta_F1"] = abs(a["F1"] - base["F1"] - row.delta_F1)
                assert max(err.values()) <= 1e-12
                if row.scenario == "clean" and a["F1"] - base["F1"] < -1e-12:
                    failures.append("clean_F1_degradation")
                if a["AP"] - base["AP"] < -0.003 - 1e-12:
                    failures.append(row.scenario + ":AP_loss_gt_.003")
                checks.append(
                    {
                        "model": name,
                        "method": method,
                        "scenario": row.scenario,
                        "n": len(idx),
                        **err,
                    }
                )
            observed[method, name] = not failures
            assert (
                observed[method, name]
                == selection["decisions"][method]["models"][name]["passed"]
            )
    for method in ["spatial", "mixture", "ridge"]:
        assert (
            all(observed[method, n] for n in ["A_s0", "A_s1", "A_s2", "ENS_A"])
            == selection["decisions"][method]["passed_all3seeds_and_ENS"]
        )
    pd.DataFrame(checks).to_csv(
        R / "results/ZERO_DEVELOPMENT_INDEPENDENT_AUDIT.csv", index=False
    )
    atomic_json(
        R / "results/ZERO_DEVELOPMENT_AUDIT_COMPLETE.json",
        {
            "metric_groups": len(checks),
            "max_absolute_error": max(
                max(q[k] for k in ["F1", "AP", "FPR", "delta_F1", "delta_AP"])
                for q in checks
            ),
            "source_fold_checks": len(source_checks),
            "all_decisions_reproduced": True,
            "no_refit": True,
            "seconds_cpu_wall": time.time() - started,
        },
    )
    print("ZERO_DEVELOPMENT_AUDIT_PASS", len(checks), flush=True)


if __name__ == "__main__":
    main()
