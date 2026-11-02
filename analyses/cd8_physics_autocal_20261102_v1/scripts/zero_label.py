"""Three predeclared label-free intercept hypotheses; source-WSI grouped development."""
from calibration_common import *
import argparse

NAMES = ["A_s0", "A_s1", "A_s2", "ENS_A"]
METHODS = ["spatial", "mixture", "ridge"]
MODERATE = [
    "clean",
    "gain_0.5",
    "gain_0.75",
    "gain_1.5",
    "gain_2.0",
    "gamma_0.7",
    "gamma_1.3",
    "zero_0.7",
    "zero_0.85",
    "noise_low",
]


def load_stage0(d):
    assert (R / "results/STAGE0_COMPLETE.json").exists()
    manifest = json.loads(
        (R / "provenance/STAGE0_VALIDATED_ARTIFACT_HASHES.json").read_text()
    )
    assert all(file_sha(p) == h for p, h in manifest.items())
    scenarios = ["clean"] + [x[0] for x in SCENARIOS]
    scores = {}
    features = {}
    for scenario in scenarios:
        features[scenario] = np.load(
            R
            / "data"
            / (
                "CLEAN_INTERNAL_CELL_PROXIES.npy"
                if scenario == "clean"
                else "CELL_PROXIES__" + scenario + "__full_internal.npy"
            )
        )
        scores[scenario] = {}
        for n in NAMES:
            if scenario == "clean":
                a = verify_scores(d, n)
            else:
                p = R / "predictions" / (n + "__" + scenario + "__full_internal.npy")
                m = json.loads(p.with_suffix(".json").read_text())
                assert file_sha(p) == m["sha256"]
                a = np.load(p)
            scores[scenario][n] = a
    return scenarios, scores, features


def fit_reference(d, fit_units, name, scores, features):
    # All labels, scaler fitting and target offsets come ONLY from named source WSIs.
    references = {"spatial": [], "mixture": []}
    X = []
    Y = []
    trace = []
    for uid in fit_units:
        ix = d.index[(d.split == cc.DEV) & (d.image_id == uid)].to_numpy()
        clean = scores["clean"][name][ix]
        zz = logits(clean)
        ff = features["clean"][ix]
        a, nn = anchor(zz, ff)
        m, diag = mixture(zz, ff)
        references["spatial"].append(a)
        references["mixture"].append(m)
        tc = unit_threshold(d.loc[ix, "ground_truth_binary"].to_numpy(), clean)
        for scenario in MODERATE:
            s = scores[scenario][name][ix]
            ts = unit_threshold(d.loc[ix, "ground_truth_binary"].to_numpy(), s)
            X.append(summary(logits(s), features[scenario][ix]))
            Y.append(
                float(
                    logit(np.clip(tc, 1e-7, 1 - 1e-7))
                    - logit(np.clip(ts, 1e-7, 1 - 1e-7))
                )
            )
            trace.append(
                {
                    "fit_unit": uid,
                    "scenario": scenario,
                    "n": len(ix),
                    "clean_unit_threshold": tc,
                    "scenario_unit_threshold": ts,
                }
            )
    scaler = StandardScaler().fit(X)
    reg = Ridge(alpha=10).fit(scaler.transform(X), Y)
    return {
        "spatial": float(np.median(references["spatial"]))
        if all(x is not None for x in references["spatial"])
        else None,
        "mixture": float(np.median(references["mixture"]))
        if all(x is not None for x in references["mixture"])
        else None,
        "scaler": scaler,
        "ridge": reg,
        "trace": trace,
        "units": list(fit_units),
        "regression_targets": Y,
    }


def correction(method, s, features, ref):
    z = logits(s)
    if method == "spatial":
        own, n = anchor(z, features)
        target = ref["spatial"]
        b = 0.0 if own is None or target is None else n / (n + 200) * (target - own)
        detail = {
            "n_candidates": n,
            "own_anchor": own,
            "reference": target,
            "fallback_identity": own is None or target is None,
        }
    elif method == "mixture":
        own, detail = mixture(z, features)
        target = ref["mixture"]
        b = 0.0 if own is None or target is None else target - own
        detail = {
            **detail,
            "reference": target,
            "own_mixture_boundary": own,
            "fallback_identity": own is None or target is None,
        }
    else:
        xx = summary(z, features)
        b = float(ref["ridge"].predict(ref["scaler"].transform(xx[None]))[0])
        detail = {"proxy_features": xx.tolist(), "fallback_identity": False}
    return float(np.clip(b, -2, 2)), {
        **detail,
        "unclipped_b": float(b),
        "clamped": bool(abs(b) > 2),
    }


def dev():
    started = time.time()
    d, _ = cc.load_data()
    scenarios, scores, features = load_stage0(d)
    units = sorted(d.loc[d.split == cc.DEV, "image_id"].unique())
    idx = d.index[d.split == cc.DEV].to_numpy()
    y = d.ground_truth_binary.to_numpy(int)
    weak = weak_mask(d)
    rows = []
    params = []
    fits = []
    gates = {}
    for name in NAMES:
        references = {}
        threshold = np.full(len(d), np.nan)
        for uid in units:
            fit_units = [x for x in units if x != uid]
            assert uid not in fit_units
            ref = fit_reference(d, fit_units, name, scores, features)
            references[uid] = ref
            fit_idx = d.index[
                (d.split == cc.DEV) & (d.image_id.isin(fit_units))
            ].to_numpy()
            tt = cc.threshold_fpr(d, scores["clean"][name], fit_idx)
            threshold[d.index[(d.split == cc.DEV) & (d.image_id == uid)]] = tt
            fits.append(
                {
                    "model": name,
                    "heldout_WSI": uid,
                    "fit_units": fit_units,
                    "no_derived_variant_of_heldout_in_fit": True,
                    "clean_threshold_from_fit_sources_only": tt,
                    "regression_fit": ref["trace"],
                    "scaler_mean": ref["scaler"].mean_.tolist(),
                    "scaler_scale": ref["scaler"].scale_.tolist(),
                    "ridge_coef": ref["ridge"].coef_.tolist(),
                    "ridge_intercept": float(ref["ridge"].intercept_),
                }
            )
        for method in METHODS:
            fail = []
            for scenario in scenarios:
                native = scores[scenario][name]
                adjusted = native.astype(float).copy()
                for uid in units:
                    ix = d.index[(d.split == cc.DEV) & (d.image_id == uid)].to_numpy()
                    b, detail = correction(
                        method, native[ix], features[scenario][ix], references[uid]
                    )
                    adjusted[ix] = shifted(native[ix], b)
                    params.append(
                        {
                            "model": name,
                            "method": method,
                            "scenario": scenario,
                            "unit": uid,
                            "b": b,
                            "detail": json.dumps(detail, sort_keys=True),
                        }
                    )
                for mode in ["fixed_0.5", "source_LOO_FPR01"]:
                    t = np.full(len(d), 0.5) if mode == "fixed_0.5" else threshold
                    for unit, ii in [("pooled", idx)] + [
                        (u, d.index[(d.split == cc.DEV) & (d.image_id == u)].to_numpy())
                        for u in units
                    ]:
                        base = cc.metrics(y[ii], native[ii].astype(float), t[ii])
                        m = cc.metrics(y[ii], adjusted[ii], t[ii])
                        ww = ii[weak[ii]]
                        rec = {
                            "model": name,
                            "method": method,
                            "scenario": scenario,
                            "unit": unit,
                            "threshold_mode": mode,
                            "threshold": float(t[ii][0])
                            if unit != "pooled"
                            else np.nan,
                            "weak_n": len(ww),
                            "weak_recall": float((adjusted[ww] > t[ww]).mean())
                            if len(ww)
                            else np.nan,
                            "delta_F1": m["F1"] - base["F1"],
                            "delta_AP": m["AP"] - base["AP"],
                            "baseline_F1": base["F1"],
                            "baseline_AP": base["AP"],
                            **m,
                        }
                        rows.append(rec)
                        if unit == "pooled" and mode == "source_LOO_FPR01":
                            if scenario == "clean" and rec["delta_F1"] < -1e-12:
                                fail.append("clean_F1_degradation")
                            if rec["delta_AP"] < -0.003 - 1e-12:
                                fail.append(scenario + ":AP_loss_gt_.003")
            gates[(method, name)] = {"passed": not fail, "reasons": fail}
            print("DEV_GATE", method, name, gates[(method, name)], flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(R / "results/ZERO_LABEL_DEV_METRICS.csv", index=False)
    pd.DataFrame(params).to_csv(
        R / "results/ZERO_LABEL_DEV_CORRECTIONS.csv", index=False
    )
    atomic_json(R / "provenance/ZERO_LABEL_SOURCE_FOLDS.json", fits)
    decisions = {
        method: {
            "passed_all3seeds_and_ENS": all(
                gates[(method, n)]["passed"] for n in NAMES
            ),
            "models": {n: gates[(method, n)] for n in NAMES},
        }
        for method in METHODS
    }
    atomic_json(
        R / "models/ZERO_LABEL_SELECTION_FROZEN.json",
        {
            "decisions": decisions,
            "passed_methods": [
                m for m in METHODS if decisions[m]["passed_all3seeds_and_ENS"]
            ],
            "frozen_before_external_adaptation": True,
            "external_labels_used": False,
            "internal_is_development_not_untouched_validation": True,
            "time": time.time(),
            "protocol_sha256": file_sha(R / "PROTOCOL.json"),
            "code_sha256": file_sha(__file__),
            "calibration_code_sha256": file_sha(R / "scripts/calibration_common.py"),
            "seconds_cpu_wall": time.time() - started,
        },
    )
    print("ZERO_LABEL_DEV_COMPLETE", decisions, flush=True)


def external():
    started = time.time()
    choice = json.loads((R / "models/ZERO_LABEL_SELECTION_FROZEN.json").read_text())
    assert choice["code_sha256"] == file_sha(__file__)
    methods = choice["passed_methods"]
    if not methods:
        atomic_json(
            R / "results/ZERO_LABEL_EXTERNAL_SKIPPED.json",
            {
                "reason": "all3predeclaredcandidates_failed_development_gate",
                "external_candidates_scored": 0,
                "selection": choice,
            },
        )
        return
    d, _ = cc.load_data()
    scenarios, scores, features = load_stage0(d)
    idx = d.index[d.split == cc.TEST].to_numpy()
    units = sorted(d.loc[d.split == cc.DEV, "image_id"].unique())
    path = R / "data/CLEAN_EXTERNAL_CELL_PROXIES.npy"
    if path.exists():
        ff = np.load(path)
    else:
        ds = Views(d)
        ff = np.full((len(d), 4), np.nan, np.float32)
        for x, a, y, ii in cc.loader(ds, idx, batch=512):
            ff[ii.numpy()] = a.numpy()[:, :4]
        assert np.isfinite(ff[idx]).all()
        atomic_npy(path, ff)
    rows = []
    params = []
    for name in NAMES:
        native = scores["clean"][name]
        ref = fit_reference(d, units, name, scores, features)
        thresholds = baseline_thresholds()[name]
        for method in methods:
            adjusted = native.astype(float).copy()
            for uid, g in d.loc[idx].groupby("image_id", sort=True):
                ii = g.index.to_numpy()
                b, detail = correction(method, native[ii], ff[ii], ref)
                adjusted[ii] = shifted(native[ii], b)
                params.append(
                    {
                        "model": name,
                        "method": method,
                        "image_id": uid,
                        "b": b,
                        "detail": json.dumps(detail, sort_keys=True),
                    }
                )
            for mode in ["fixed_0.5", "dev_FPR01_each_WSI"]:
                t = float(np.asarray(thresholds[mode], dtype=native.dtype))
                rows.extend(
                    aggregate(
                        d,
                        idx,
                        adjusted,
                        t,
                        {
                            "model": name,
                            "method": method,
                            "threshold_mode": mode,
                            "external_known_exploratory": True,
                        },
                    )
                )
            path = R / "predictions" / ("AUTOCAL_" + method + "_" + name + ".npy")
            atomic_npy(path, adjusted)
            atomic_json(
                path.with_suffix(".json"),
                {
                    "sha256": file_sha(path),
                    "keys_sha256": cc.canonical_keys_hash(d),
                    "n_external": len(idx),
                    "selection_sha256": file_sha(
                        R / "models/ZERO_LABEL_SELECTION_FROZEN.json"
                    ),
                },
            )
    pd.DataFrame(rows).to_csv(
        R / "results/ZERO_LABEL_EXTERNAL_METRICS.csv", index=False
    )
    pd.DataFrame(params).to_csv(
        R / "results/ZERO_LABEL_EXTERNAL_CORRECTIONS.csv", index=False
    )
    atomic_json(
        R / "results/ZERO_LABEL_EXTERNAL_COMPLETE.json",
        {
            "methods": methods,
            "models": NAMES,
            "external_selected_models": False,
            "seconds_cpu_wall": time.time() - started,
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "external"], required=True)
    a = ap.parse_args()
    dev() if a.phase == "dev" else external()
