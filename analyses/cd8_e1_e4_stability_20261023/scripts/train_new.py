"""Train the four frozen HGB experiments with leave-one-training-WSI-out scores."""

import os

os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import time, json, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *


def threshold_select(y, s, groups):
    records = []
    for j in range(1, 100):
        vals = []
        for g in sorted(set(groups)):
            v = groups == g
            p = s[v] > j / 100
            z = y[v]
            tp = ((z == 1) & p).sum()
            fp = ((z == 0) & p).sum()
            fn = ((z == 1) & ~p).sum()
            vals.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        records.append(
            dict(grid_integer=j, threshold=j / 100, mean_WSI_F1=float(np.mean(vals)))
        )
    best = max(r["mean_WSI_F1"] for r in records)
    ties = [r for r in records if abs(r["mean_WSI_F1"] - best) <= 1e-15]
    selected = sorted(
        ties, key=lambda r: (abs(r["grid_integer"] - 50), -r["grid_integer"])
    )[0]
    return selected["threshold"], records


def main():
    for d in ["models", "predictions", "results", "curves"]:
        (A / d).mkdir(exist_ok=True)
    fm = json.loads((A / "features/FEATURES_COMPLETE.json").read_text())
    assert fm["rows"] == 1817783
    featurefile = A / "PhenoBIC_E1_E4_features.parquet"
    assert sha(featurefile) == fm["parquet_sha256"]
    cfg = json.loads((A / "RUN_CONFIG.json").read_text())
    for name, h in cfg["extraction_code"].items():
        assert sha(A / "scripts" / name) == h
    sig = dict(
        features_sha256=fm["parquet_sha256"],
        config_sha256=sha(A / "RUN_CONFIG.json"),
        protocol_sha256=sha(A / "E1_E4_PROTOCOL.json"),
        code_sha256=sha(__file__),
        keys_sha256=sha(A / "keys/key_manifest.csv"),
        cells_sha256=sha(A / "audit/cells.parquet"),
        parameters=PARAMS,
        features=SETS,
        cell_weight="uniform, identical to old model.fit without sample_weight",
    )
    write_json(A / "TRAINING_RUN.json", sig)
    signature = digest(sig)
    f = pd.read_parquet(featurefile)
    ann = pd.read_parquet(A / "audit/cells.parquet")
    featurecols = (
        I8 + S9 + [c for c in f if c.startswith("qc_")] + ["intensity_input_valid"]
    )
    d = ann.merge(
        f[KEYS + featurecols],
        on=KEYS,
        validate="one_to_one",
        how="outer",
        indicator=True,
        suffixes=("_old", ""),
    )
    assert len(d) == 1817783 and (d._merge == "both").all()
    d = d.drop(columns="_merge")
    assert not np.isinf(d[I8 + S9].to_numpy(float)).any()
    train = (d.split == "training_WSIs") & d.ground_truth_binary.isin([0, 1])
    tr = d.loc[train].copy()
    assert len(tr) == 101057
    y = tr.ground_truth_binary.to_numpy(int)
    groups = tr.image_id.to_numpy()
    assert len(set(groups)) == 3
    thresholds = {}
    selection = []
    fits = []
    oof = tr[
        KEYS + ["split", "context", "ground_truth_original", "ground_truth_binary"]
    ].copy()
    for name, cols in SETS.items():
        X = tr[cols].to_numpy(float)
        scores = np.full(len(tr), np.nan)
        for i, held in enumerate(sorted(set(groups))):
            path = A / "models" / ("%s_fold%d.joblib" % (name, i))
            fit = groups != held
            val = ~fit
            if path.exists():
                b = joblib.load(path)
                assert b["signature"] == signature and b["heldout"] == held
            else:
                start = time.time()
                est = HistGradientBoostingClassifier(**PARAMS).fit(X[fit], y[fit])
                b = dict(
                    estimator=est,
                    features=cols,
                    signature=signature,
                    heldout=held,
                    seconds=time.time() - start,
                    n_train=int(fit.sum()),
                    training_image_ids=sorted(set(groups[fit])),
                )
                joblib.dump(b, path, compress=3)
            scores[val] = b["estimator"].predict_proba(X[val])[:, 1]
            fits.append(
                dict(
                    model=name,
                    fit_kind="OOF",
                    fold=i,
                    heldout=held,
                    n_train=b["n_train"],
                    seconds=b["seconds"],
                    file=str(path.relative_to(A)),
                )
            )
            print("FIT", name, "OOF", held, flush=True)
        assert np.isfinite(scores).all()
        oof[name + "_score"] = scores
        t, rows = threshold_select(y, scores, groups)
        thresholds[name] = t
        for r in rows:
            r["model"] = name
            selection.append(r)
        path = A / "models" / (name + ".joblib")
        if path.exists():
            b = joblib.load(path)
            assert b["signature"] == signature
        else:
            start = time.time()
            est = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
            b = dict(
                estimator=est,
                features=cols,
                signature=signature,
                seconds=time.time() - start,
                n_train=len(y),
                training_image_ids=sorted(set(groups)),
                threshold_fixed=0.5,
                threshold_oof=t,
                positive_rule="score strictly greater than threshold",
                parameters=PARAMS,
                calibration=cfg["geometry"],
                calibration_config=str(A / "RUN_CONFIG.json"),
            )
            joblib.dump(b, path, compress=3)
        fits.append(
            dict(
                model=name,
                fit_kind="final",
                fold=-1,
                heldout="",
                n_train=len(y),
                seconds=b["seconds"],
                file=str(path.relative_to(A)),
            )
        )
        print("FIT", name, "FINAL; OOF threshold", t, flush=True)
    # No new internal/external metrics or threshold optimization precedes this freeze.
    freeze = dict(
        thresholds=thresholds,
        signature=signature,
        training_only=True,
        feature_sets=SETS,
        protocol_sha256=sig["protocol_sha256"],
        sample_sha256=sha(A / "keys/stability_sample.parquet"),
        rule="strict >",
        old_baselines="original .5 in both modes",
        OOF_selected_threshold_warning="OOF threshold development fitting value; not unbiased threshold-selection performance",
    )
    write_json(A / "THRESHOLDS_FROZEN.json", freeze)
    pd.DataFrame(selection).to_csv(A / "results/OOF_threshold_grid.csv", index=False)
    pd.DataFrame(fits).to_csv(A / "results/new_model_fit_log.csv", index=False)
    assert len(fits) == 16
    for mode in ["fixed_0.5", "OOF_threshold"]:
        out = oof.copy()
        out["threshold_mode"] = mode
        for name in SETS:
            t = 0.5 if mode == "fixed_0.5" else thresholds[name]
            out[name + "_threshold"] = t
            out[name + "_prediction"] = (out[name + "_score"] > t).astype("int8")
        out.to_parquet(
            A / "predictions" / ("training_OOF_" + mode + ".parquet"),
            index=False,
            compression="zstd",
        )
    for name, cols in SETS.items():
        est = joblib.load(A / "models" / (name + ".joblib"))["estimator"]
        score = np.empty(len(d))
        for start in range(0, len(d), 100000):
            score[start : start + 100000] = est.predict_proba(
                d.iloc[start : start + 100000][cols].to_numpy(float)
            )[:, 1]
        d[name + "_score"] = score
        print("PREDICTED", name, len(d), flush=True)
    keep = (
        KEYS
        + [
            "split",
            "context",
            "source_image_id",
            "ground_truth_original",
            "ground_truth_binary",
            "intensity_input_valid",
        ]
        + [c for c in d if c.startswith("qc_")]
        + [
            n + "_score"
            for n in ["GMM", "intensity_8", "morphospatial_90"] + list(SETS)
        ]
    )
    for mode in ["fixed_0.5", "OOF_threshold"]:
        out = d[keep].copy()
        out["threshold_mode"] = mode
        out["calibration_valid"] = ~out.qc_calibration_invalid
        for name in ["GMM", "intensity_8", "morphospatial_90"] + list(SETS):
            t = thresholds[name] if mode == "OOF_threshold" and name in SETS else 0.5
            out[name + "_threshold"] = t
            out[name + "_prediction"] = np.where(
                np.isfinite(out[name + "_score"]),
                (out[name + "_score"] > t).astype("int8"),
                -1,
            )
        out.to_parquet(
            A / "predictions" / ("final_all_cells_" + mode + ".parquet"),
            index=False,
            compression="zstd",
        )
    write_json(
        A / "results/TRAINING_COMPLETE.json",
        dict(
            status="complete",
            new_fits=len(fits),
            all_cells=len(d),
            training_cells=len(tr),
            thresholds=thresholds,
            signature=signature,
            model_hashes={n: sha(A / "models" / (n + ".joblib")) for n in SETS},
        ),
    )
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
