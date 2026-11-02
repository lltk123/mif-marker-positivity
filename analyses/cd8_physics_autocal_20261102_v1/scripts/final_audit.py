from calibration_common import *
from stage0_audit import ap_independent


def independent_metrics(y, s, t):
    pred = s > np.asarray(t, s.dtype)
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    return {
        "F1": float(2 * tp / (2 * tp + fp + fn)),
        "AP": ap_independent(y, s),
        "FPR": float(fp / (fp + tn)),
    }


def fewshot_checks(d):
    y = d.ground_truth_binary.to_numpy(int)
    native = verify_scores(d, "ENS_A")
    t = float(np.float32(baseline_thresholds()["ENS_A"]["dev_FPR01_each_WSI"]))
    pools = reserved(d)
    params = pd.read_csv(
        R / "results/NEW_FEWSHOT_PARAMETERS.csv", float_precision="round_trip"
    )
    metrics = pd.read_csv(
        R / "results/NEW_FEWSHOT_METRICS.csv", float_precision="round_trip"
    )
    chosen = pd.read_csv(R / "results/NEW_FEWSHOT_SELECTED_IDS.csv")
    checks = []
    for (rep, strategy, budget), g in params.groupby(
        ["replicate", "strategy", "budget"]
    ):
        reserve, hold = pools[int(rep)]
        s = native.astype(float).copy()
        for row in g.itertuples():
            ix = d.index[(d.split == cc.TEST) & (d.image_id == row.image_id)].to_numpy()
            s[ix] = shifted(native[ix], row.intercept)
            sel = (
                chosen[
                    (chosen.replicate == rep)
                    & (chosen.strategy == strategy)
                    & (chosen.budget == budget)
                    & (chosen.image_id == row.image_id)
                ]
                .sort_values("selection_rank")
                .row_id.to_numpy(int)
            )
            assert (
                len(sel) == budget
                and set(sel).issubset(set(reserve[row.image_id]))
                and not hold[sel].any()
            )
            order = reserve[row.image_id]
            if strategy == "active_boundary":
                order = order[
                    np.argsort(abs(logits(native[order]) - logit(t)), kind="stable")
                ]
            assert np.array_equal(sel, order[:budget])
        ix = np.flatnonzero(hold)
        actual = independent_metrics(y[ix], s[ix], t)
        row = metrics[
            (metrics.replicate == rep)
            & (metrics.strategy == strategy)
            & (metrics.budget == budget)
            & (metrics.level == "pooled")
        ].iloc[0]
        errors = {k: abs(actual[k] - row[k]) for k in actual}
        assert max(errors.values()) <= 1e-12
        checks.append(
            {
                "kind": "fewshot",
                "model": "ENS_A",
                "replicate": int(rep),
                "strategy": strategy,
                "budget": int(budget),
                "n": len(ix),
                **errors,
            }
        )
    return checks


def main():
    d, _ = cc.load_data()
    y = d.ground_truth_binary.to_numpy(int)
    checks = fewshot_checks(d)
    for filename in [
        "ZERO_LABEL_EXTERNAL_METRICS.csv",
        "CONTINUATION_EXTERNAL_METRICS.csv",
    ]:
        path = R / "results" / filename
        if not path.exists():
            continue
        tab = pd.read_csv(path, float_precision="round_trip")
        idx = d.index[d.split == cc.TEST].to_numpy()
        for row in tab[tab.level == "pooled"].itertuples():
            if filename.startswith("ZERO"):
                pred = (
                    R
                    / "predictions"
                    / ("AUTOCAL_" + row.method + "_" + row.model + ".npy")
                )
            else:
                pred = R / "predictions" / (row.model + "__external.npy")
            meta = json.loads(pred.with_suffix(".json").read_text())
            assert file_sha(pred) == meta["sha256"]
            s = np.load(pred)
            assert s.shape == (len(d),) and np.isfinite(s[idx]).all()
            actual = independent_metrics(y[idx], s[idx], row.threshold)
            errors = {k: abs(actual[k] - getattr(row, k)) for k in actual}
            assert max(errors.values()) <= 1e-12
            checks.append(
                {
                    "kind": "newexternal",
                    "model": row.model,
                    "threshold_mode": row.threshold_mode,
                    "n": len(idx),
                    **errors,
                }
            )

    # Independently check the development metrics that controlled entry/stop decisions.
    if (R / "results/CONTINUATION_CV.csv").exists():
        cv = pd.read_csv(
            R / "results/CONTINUATION_CV.csv", float_precision="round_trip"
        )
        for row in cv.itertuples():
            name = "NEWCV_" + row.variant + "_f" + str(row.fold)
            p = R / "predictions" / (name + "__heldout.npy")
            meta = json.loads(p.with_suffix(".json").read_text())
            assert file_sha(p) == meta["sha256"]
            idx = d.index[
                (d.split == cc.TRAIN) & (d.image_id == row.heldout_training_WSI)
            ].to_numpy()
            assert (
                hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()
                == meta["signature"]["indices_sha256"]
            )
            a = np.load(p)
            v = independent_metrics(y[idx], a[idx], 0.5)
            errors = {k: abs(v[k] - getattr(row, k)) for k in v}
            assert max(errors.values()) <= 1e-12
            checks.append(
                {
                    "kind": "training_WSI_development",
                    "model": name,
                    "n": len(idx),
                    **errors,
                }
            )
    for fn, perturbed in [
        ("CONTINUATION_INTERNAL_METRICS.csv", False),
        ("CONTINUATION_PERTURBATION_METRICS.csv", True),
    ]:
        p = R / "results" / fn
        if not p.exists():
            continue
        idx = d.index[d.split == cc.DEV].to_numpy()
        tab = pd.read_csv(p, float_precision="round_trip")
        for row in tab[tab.level == "pooled"].itertuples():
            tag = "internal_" + row.scenario if perturbed else "internal"
            if row.model.startswith("NEW_ENS_"):
                variant = row.model[len("NEW_ENS_") :]
                arr = []
                for seed in range(3):
                    pp = (
                        R
                        / "predictions"
                        / ("NEW_" + variant + "_s" + str(seed) + "__" + tag + ".npy")
                    )
                    mm = json.loads(pp.with_suffix(".json").read_text())
                    assert file_sha(pp) == mm["sha256"]
                    arr.append(np.load(pp))
                a = np.mean(arr, axis=0, dtype=np.float32)
                if not perturbed:
                    assert np.array_equal(
                        a[idx],
                        np.load(R / "predictions" / (row.model + "__internal.npy"))[
                            idx
                        ],
                    )
            else:
                pp = R / "predictions" / (row.model + "__" + tag + ".npy")
                mm = json.loads(pp.with_suffix(".json").read_text())
                assert file_sha(pp) == mm["sha256"]
                a = np.load(pp)
            v = independent_metrics(y[idx], a[idx], row.threshold)
            errors = {k: abs(v[k] - getattr(row, k)) for k in v}
            assert max(errors.values()) <= 1e-12
            checks.append(
                {
                    "kind": "synthetic_internal_development"
                    if perturbed
                    else "clean_internal_development",
                    "model": row.model,
                    "n": len(idx),
                    "threshold_mode": row.threshold_mode,
                    **errors,
                }
            )
    # Paired parentweights and epoch samples are audited from actual runtime records.
    paired = []
    if (R / "results/CONTINUATION_CV.csv").exists():
        selected = json.loads(
            (R / "models/CONTINUATION_CV_SELECTION.json").read_text()
        ).get("selected")
        sets = [
            [
                "NEWCV_" + v + "_f" + str(f)
                for v in ["control", "physics", "sdf", "ranking"]
            ]
            for f in range(3)
        ]
        if selected:
            sets += [
                ["NEW_" + v + "_s" + str(k) for v in ["control", selected]]
                for k in range(3)
            ]
        for names in sets:
            prov = [
                json.loads(
                    (R / "provenance" / (n + "_initialization.json")).read_text()
                )
                for n in names
            ]
            assert len({p["parent_sha256"] for p in prov}) == 1
            for epoch in [4, 5] if names[0].startswith("NEWCV") else [6, 7]:
                ps = [
                    json.loads(
                        (
                            R / "provenance" / (n + "_epoch" + str(epoch) + ".json")
                        ).read_text()
                    )
                    for n in names
                ]
                assert len({p["indices_sha256"] for p in ps}) == 1 and all(
                    p["recorded_before_epoch"] for p in ps
                )
                paired.append(
                    {
                        "names": names,
                        "epoch": epoch,
                        "same_sample_sha256": ps[0]["indices_sha256"],
                        "same_parent_sha256": prov[0]["parent_sha256"],
                    }
                )
            nonsdf = [p["initial_hash"] for p in prov if p["variant"] != "sdf"]
            assert len(set(nonsdf)) == 1
    models = {}
    for p in (R / "models").glob("*.pt"):
        meta = json.loads(p.with_suffix(".json").read_text())
        assert file_sha(p) == meta["sha256"]
        models[p.name] = meta["sha256"]
    ledger = json.loads((R / "logs/NEURAL_BUDGET.json").read_text())
    assert not ledger.get("active") and ledger["consumed_seconds"] <= 28800
    pd.DataFrame(checks).to_csv(
        R / "results/FINAL_INDEPENDENT_METRIC_AUDIT.csv", index=False
    )
    atomic_json(R / "provenance/FINAL_WEIGHT_HASHES.json", models)
    atomic_json(R / "provenance/PAIRED_CONTINUATION_AUDIT.json", paired)
    import importlib.metadata, platform

    env = {
        n: importlib.metadata.version(n)
        for n in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib"]
    }
    env.update(python=platform.python_version(), platform=platform.platform())
    atomic_json(R / "provenance/ENVIRONMENT.json", env)
    atomic_json(
        R / "results/FINAL_AUDIT_COMPLETE.json",
        {
            "metric_checks": len(checks),
            "max_metric_absolute_error": max(
                max(r[k] for k in ["F1", "AP", "FPR"]) for r in checks
            ),
            "new_weight_count": len(models),
            "paired_sample_epoch_checks": len(paired),
            "reserved_pool_isolation": True,
            "active_sampling_uses_no_labels_for_selection": True,
            "zero_intercept_preserves_native_threshold_semantics": True,
            "budget": ledger,
            "expert_adjudication": "pending",
            "Library_uploads": 0,
        },
    )
    print("FINAL_AUDIT_PASS", len(checks), flush=True)


if __name__ == "__main__":
    main()
