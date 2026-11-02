"""Full-population, frozen-threshold evaluation and separate target-label calibration."""
from condition_core import *
from execution_guards import native_threshold
from scipy.special import expit, logit
from scipy.optimize import brentq

NUM = [
    "P",
    "R",
    "F1",
    "AP",
    "ROC_AUC",
    "FPR",
    "positive_fraction_absolute_error",
    "Brier",
]


def strata(d):
    p = json.loads((RUN / "provenance/STRATA_FROZEN.json").read_text())
    o = p["legacy_raw_proxies"]
    y = d.ground_truth_binary.to_numpy(int)
    signal = d.nonzero_rate_whole.to_numpy() * d.positive_mean_whole.to_numpy()
    signal = np.where(
        d.repair_whole_support_valid & d.repair_whole_no_nonzero_signal, 0, signal
    )
    return {
        "weak_observed_positive": (y == 1)
        & (signal <= p["normalized_observed_target_mean_q25_train_positive"]),
        "legacy_raw_weak_positive": (y == 1)
        & (d.audit_raw_whole_mean.to_numpy() <= o["weak_positive_whole_mean_q25"]),
        "crowded_proxy": d.audit_raw_neighbor_edge_area_fraction.to_numpy()
        > p["crowding_neighbor_area_q75_train"],
        "bright_neighbor_negative_proxy": (y == 0)
        & (d.audit_raw_neighbor_edge_area_fraction.to_numpy() > 0)
        & (
            d.audit_raw_neighbor_edge_mean_diff.to_numpy()
            > o["contact_bright_diff_q75"]
        ),
        "target_zero": d.repair_whole_no_nonzero_signal.to_numpy(bool),
        "reference_zero": d.repair_reference_no_nonzero_signal.to_numpy(bool),
        "reference_support_invalid": ~d.repair_reference_support_valid.to_numpy(bool),
        "reference_MAD_zero": d.qc_reference_mad_zero.to_numpy(bool),
        "boundary_proxy": d.qc_truncated.to_numpy(bool)
        | d.qc_context_clipped.to_numpy(bool)
        | (d.crop_target_fraction.to_numpy() < 1),
    }


def groups(d):
    for split, sg in d.groupby("split", sort=True):
        yield split, "split", split, "all", sg.index.to_numpy()
        for context, g in sg.groupby("context", sort=True):
            yield split, "context", context, context, g.index.to_numpy()
        for uid, g in sg.groupby("image_id", sort=True):
            yield split, "image", uid, g.context.iloc[0], g.index.to_numpy()


def verified(d):
    done = json.loads(
        (RUN / "models/TRAINING_AND_PREDICTION_COMPLETE.json").read_text()
    )
    sig = done["signatures"]
    assert canonical_keys_hash(d) == sig["four_keys_sha256"]
    assert file_sha(RUN / "PROTOCOL.json") == sig["protocol_sha256"]
    for n, h in sig["code_inventory"].items():
        assert file_sha(RUN / "scripts" / n) == h
    scores = {}
    ts = {}
    checks = []
    for rec in json.loads((RUN / "models/THRESHOLDS_FROZEN.json").read_text())[
        "records"
    ]:
        n = rec["model"]
        p = RUN / "predictions" / (n + "_all.npy")
        m = json.loads(p.with_suffix(".json").read_text())
        a = np.load(p)
        assert (
            file_sha(p) == m["sha256"]
            and m["signatures"] == sig
            and a.shape == (len(d),)
            and np.isfinite(a).all()
        )
        if n.startswith("ENS"):
            for fn, h in m["components"].items():
                assert file_sha(RUN / "predictions" / fn) == h
            assert np.array_equal(
                a,
                np.mean(
                    [
                        np.load(
                            RUN
                            / "predictions"
                            / (n[4:] + "_s" + str(seed) + "_all.npy")
                        )
                        for seed in [0, 1, 2]
                    ],
                    axis=0,
                    dtype=np.float32,
                ),
            )
        else:
            assert file_sha(RUN / "models" / (n + ".pt")) == m["model_sha256"]
        scores[n] = a
        ts[n] = {k: rec[k] for k in ["fixed_0.5", "dev_FPR01_each_WSI"]}
        checks.append(
            {
                "model": n,
                "sha256": m["sha256"],
                "rows": len(a),
                "dtype": str(a.dtype),
                "finite": True,
                "inputs_keys_models_verified": True,
            }
        )
    dev = d.index[d.split == DEV].to_numpy()
    for n in ["H0", "H1", "ENS_image_target_other"]:
        p = OLD / "predictions" / (n + "_all.npy")
        m = json.loads(
            (OLD / "predictions" / (n + "_external_complete.json")).read_text()
        )
        a = np.load(p)
        assert (
            file_sha(p) == m["score_sha256"]
            and a.shape == (len(d),)
            and np.isfinite(a).all()
        )
        name = "OLD_" + n
        scores[name] = a
        ts[name] = {"fixed_0.5": 0.5, "dev_FPR01_each_WSI": threshold_fpr(d, a, dev)}
        checks.append(
            {
                "model": name,
                "sha256": m["score_sha256"],
                "rows": len(a),
                "dtype": str(a.dtype),
                "finite": True,
                "inputs_keys_models_verified": True,
            }
        )
    atomic_json(
        RUN / "models/COMPARISON_THRESHOLDS_FROZEN.json",
        {
            "thresholds": ts,
            "external_used": False,
            "baseline_threshold_source": DEV,
            "strict_operator": ">",
            "selection_sha256": file_sha(RUN / "models/GROUP_SELECTION_FROZEN.json"),
        },
    )
    pd.DataFrame(checks).to_csv(RUN / "results/PREDICTION_INTEGRITY.csv", index=False)
    return scores, ts


def provenance():
    checks = []
    for phase, epochs in [("CV", 4), ("final", 6)]:
        for k in range(3):
            names = [
                ("CV_" + g + "_f" + str(k)) if phase == "CV" else g + "_s" + str(k)
                for g in GROUPS
            ]
            hs = [
                json.loads(
                    (RUN / "provenance" / (n + "_initialization.json")).read_text()
                )["hash"]
                for n in names
            ]
            assert len(set(hs)) == 1
            for epoch in range(epochs):
                rs = [
                    json.loads(
                        (
                            RUN / "provenance" / (n + "_epoch" + str(epoch) + ".json")
                        ).read_text()
                    )
                    for n in names
                ]
                assert len({v["sha256"] for v in rs}) == 1 and all(
                    v["recorded_during_training"] for v in rs
                )
                checks.append(
                    {
                        "phase": phase,
                        "fold_or_seed": k,
                        "epoch": epoch,
                        "initial_hash": hs[0],
                        "sample_hash": rs[0]["sha256"],
                        "rows": rs[0]["rows"],
                        "paired": True,
                        "recorded_during_training": True,
                    }
                )
    pd.DataFrame(checks).to_csv(
        RUN / "results/PAIRED_RUNTIME_PROVENANCE.csv", index=False
    )


def evaluate(d, scores, ts):
    y = d.ground_truth_binary.to_numpy(int)
    risk = strata(d)
    out = []
    rr = []
    for n, s in scores.items():
        for split, level, unit, context, ix in groups(d):
            for mode, t in ts[n].items():
                rec = {
                    "model": n,
                    "split": split,
                    "level": level,
                    "unit": unit,
                    "context": context,
                    "threshold_mode": mode,
                    "threshold": t,
                }
                out.append({**rec, **metrics(y[ix], s[ix], t)})
                if level != "image":
                    for label, mask in risk.items():
                        j = ix[mask[ix]]
                        if len(j):
                            rr.append(
                                {**rec, "stratum": label, **metrics(y[j], s[j], t)}
                            )
        print("EVALUATED", n, flush=True)
    tab = pd.DataFrame(out)
    st = pd.DataFrame(rr)
    tab.to_csv(RUN / "results/METRICS_BY_UNIT.csv", index=False)
    st.to_csv(RUN / "results/STRATA_METRICS.csv", index=False)
    for filename, level in [
        ("EXTERNAL_MAIN", "split"),
        ("EXTERNAL_BY_SOURCE", "context"),
        ("EXTERNAL_BY_CORE", "image"),
    ]:
        tab[(tab.split == TEST) & (tab.level == level)].to_csv(
            RUN / "results" / (filename + ".csv"), index=False
        )
    keys = ["group", "split", "level", "unit", "context", "threshold_mode"]
    summaries = []
    for frame, filename, extra in [
        (tab, "SEED_SUMMARY", []),
        (st, "STRATA_SEED_SUMMARY", ["stratum"]),
    ]:
        seed = frame[frame.model.str.match(r"^[ABCD]_s[012]$")].copy()
        seed["group"] = seed.model.str[0]
        summ = seed.groupby(keys + extra)[NUM].agg(["mean", "std", "count"])
        summ.columns = ["_".join(c) for c in summ.columns]
        summ = summ.reset_index()
        summ.to_csv(RUN / "results" / (filename + ".csv"), index=False)
        summaries.append(summ)
    core = (
        tab[(tab.level == "image") & (tab.split == TEST)]
        .groupby(["model", "threshold_mode", "context"])
        .agg(
            n_cores=("unit", "size"),
            mean_core_positive_fraction_MAE=(
                "positive_fraction_absolute_error",
                "mean",
            ),
            median_core_positive_fraction_MAE=(
                "positive_fraction_absolute_error",
                "median",
            ),
            max_core_positive_fraction_MAE=("positive_fraction_absolute_error", "max"),
            mean_core_AP=("AP", "mean"),
        )
        .reset_index()
    )
    core.to_csv(RUN / "results/CORE_MACRO_SUMMARY.csv", index=False)
    paired = []
    join = ["split", "level", "unit", "context", "threshold_mode"]
    for g in ["B", "C", "D"]:
        for seed in range(3):
            z = tab[tab.model == g + "_s" + str(seed)].merge(
                tab[tab.model == "A_s" + str(seed)],
                on=join,
                validate="one_to_one",
                suffixes=("_new", "_A"),
            )
            for row in z.to_dict("records"):
                paired.append(
                    {
                        **{k: row[k] for k in join},
                        "group": g,
                        "seed": seed,
                        **{"delta_" + k: row[k + "_new"] - row[k + "_A"] for k in NUM},
                    }
                )
    pd.DataFrame(paired).to_csv(RUN / "results/PAIRED_VS_A.csv", index=False)
    selected = json.loads((RUN / "models/GROUP_SELECTION_FROZEN.json").read_text())[
        "selected_group"
    ]
    s = summaries[0]
    s = s[
        (s.split == TEST)
        & (s.level == "split")
        & (s.threshold_mode == "dev_FPR01_each_WSI")
    ].set_index("group")
    w = summaries[1]
    w = w[
        (w.split == TEST)
        & (w.level == "split")
        & (w.threshold_mode == "dev_FPR01_each_WSI")
        & (w.stratum == "weak_observed_positive")
    ].set_index("group")
    da = float(s.loc[selected, "AP_mean"] - s.loc["A", "AP_mean"])
    dr = float(w.loc[selected, "R_mean"] - w.loc["A", "R_mean"])
    df = float(s.loc[selected, "FPR_mean"] - s.loc["A", "FPR_mean"])
    atomic_json(
        RUN / "results/PREDECLARED_SUCCESS_CHECK.json",
        {
            "selected_group_from_training_WSI_CV": selected,
            "delta_seed_mean_AP_vs_A": da,
            "delta_seed_mean_weak_observed_recall_vs_A": dr,
            "delta_seed_mean_FPR_vs_A": df,
            "joint_descriptive_target_met": bool(
                da >= 0.005 and dr >= 0.02 and df <= 0.002
            ),
            "selected_external_seed_mean_FPR": float(s.loc[selected, "FPR_mean"]),
            "external_reselection_performed": False,
        },
    )
    return tab, st


def calibration(d, scores, ts):
    y = d.ground_truth_binary.to_numpy(int)
    test = d[d.split == TEST]
    selected = json.loads((RUN / "models/GROUP_SELECTION_FROZEN.json").read_text())[
        "selected_group"
    ]
    names = list(dict.fromkeys(["ENS_A", "ENS_" + selected, "OLD_H0", "OLD_H1"]))
    records = []
    params = []
    indices = []
    checks = []
    risk = strata(d)
    for rep in [101, 102, 103]:
        reserve = {}
        hold = np.zeros(len(d), bool)
        hold[test.index] = True
        for uid, g in test.groupby("image_id", sort=True):
            keys = [
                int.from_bytes(
                    hashlib.blake2b(
                        (str(rep) + "|" + str(uid) + "|" + str(int(c))).encode(),
                        digest_size=8,
                    ).digest(),
                    "little",
                )
                for c in g.cell_id
            ]
            chosen = g.index.to_numpy()[np.argsort(np.asarray(keys, np.uint64))[:100]]
            assert len(chosen) == 100
            reserve[uid] = chosen
            hold[chosen] = False
            for rank, i in enumerate(chosen, 1):
                indices.append(
                    {
                        "replicate": rep,
                        "image_id": uid,
                        "row_id": int(i),
                        "cell_id": int(d.loc[i, "cell_id"]),
                        "sampling_rank": rank,
                        "label_used_for_sampling": False,
                    }
                )
        assert hold.sum() == 433280
        for n in names:
            native = scores[n]
            s = native.astype(float)
            eps = np.finfo(float).eps
            z = logit(np.clip(s, eps, 1 - eps))
            t = native_threshold(ts[n]["dev_FPR01_each_WSI"], native.dtype)
            for budget in [0, 20, 50, 100]:
                adjusted = s.copy()
                unchanged = np.zeros(len(d), bool)
                for uid, g in test.groupby("image_id", sort=True):
                    ix = g.index.to_numpy()
                    cal = reserve[uid][:budget]
                    fallback = budget > 0 and len(np.unique(y[cal])) < 2
                    b = 0.0
                    if budget and not fallback:
                        b = float(
                            brentq(
                                lambda b: float(np.sum(expit(z[cal] + b) - y[cal])) + b,
                                -64,
                                64,
                            )
                        )
                        adjusted[ix] = expit(z[ix] + b)
                    else:
                        unchanged[ix] = True
                    params.append(
                        {
                            "model": n,
                            "replicate": rep,
                            "budget_per_core": budget,
                            "image_id": uid,
                            "intercept": b,
                            "n_calibration_positive": int(y[cal].sum()),
                            "one_class_fallback": fallback,
                            "n_reserved_excluded": 100,
                        }
                    )
                j = hold & unchanged
                assert np.array_equal(
                    adjusted[j] > t, native[j] > np.asarray(t, native.dtype)
                )
                checks.append(
                    {
                        "model": n,
                        "replicate": rep,
                        "budget": budget,
                        "unchanged_mapping_n": int(j.sum()),
                        "changed_native_decisions": 0,
                        "evaluation_n": int(hold.sum()),
                    }
                )
                for split, level, unit, context, ix in groups(d):
                    if split != TEST:
                        continue
                    ix = ix[hold[ix]]
                    rec = {
                        "model": n,
                        "replicate": rep,
                        "budget_per_core": budget,
                        "level": level,
                        "unit": unit,
                        "context": context,
                        "threshold": t,
                        "stratum": "all",
                        "deployment": "supervised target-core intercept calibration",
                    }
                    records.append({**rec, **metrics(y[ix], adjusted[ix], t)})
                    if level != "image":
                        for label in [
                            "weak_observed_positive",
                            "bright_neighbor_negative_proxy",
                            "target_zero",
                        ]:
                            k = ix[risk[label][ix]]
                            if len(k):
                                records.append(
                                    {
                                        **rec,
                                        "stratum": label,
                                        **metrics(y[k], adjusted[k], t),
                                    }
                                )
        print("CALIBRATION", rep, flush=True)
    for rows, fn in [
        (records, "TARGET_LABEL_CALIBRATION"),
        (params, "CALIBRATION_PARAMETERS"),
        (indices, "CALIBRATION_RESERVED_INSTANCES"),
        (checks, "CALIBRATION_DTYPE_AUDIT"),
    ]:
        pd.DataFrame(rows).to_csv(RUN / "results" / (fn + ".csv"), index=False)
    atomic_json(
        RUN / "results/CALIBRATION_COMPLETE.json",
        {
            "models": names,
            "replicates": [101, 102, 103],
            "budgets": [0, 20, 50, 100],
            "reserved_per_core": 100,
            "same_433280_evaluation_instances_for_all_models_and_budgets_per_replicate": True,
            "all_unchanged_mappings_exact": True,
        },
    )


def main():
    start = time.time()
    d, _ = load_data()
    scores, ts = verified(d)
    provenance()
    tab, st = evaluate(d, scores, ts)
    calibration(d, scores, ts)
    atomic_json(
        RUN / "results/EVALUATION_COMPLETE.json",
        {
            "seconds": time.time() - start,
            "n_models": len(scores),
            "metric_rows": len(tab),
            "stratum_rows": len(st),
            "train_n": 101057,
            "internal_n": 539132,
            "external_n": 437480,
            "full_population": True,
            "external_exploratory_not_blind": True,
            "PanM_accessed": False,
            "code_sha256": file_sha(__file__),
        },
    )
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
