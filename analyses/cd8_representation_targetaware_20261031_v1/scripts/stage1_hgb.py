"""Seven fixed HGB fits. No external scores or metrics are inspected here."""
import os

os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import json, time, sys, resource, warnings
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
from experiment_common import *

PARAMS = {
    "learning_rate": 0.06,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 80,
    "l2_regularization": 2.0,
    "early_stopping": False,
    "random_state": 0,
}


def prepare(X, train, mode):
    meta = {"mode": mode}
    missing = np.isnan(X)
    if mode == "native":
        return X, meta
    if mode == "indicator_only":
        return missing.astype(float), meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(X[train], axis=0)
    meta["all_missing_train"] = np.flatnonzero(~np.isfinite(median)).tolist()
    median = np.where(np.isfinite(median), median, 0)
    A = np.where(missing, median[None, :], X)
    meta["median"] = median
    if mode == "impute_indicator_quantile":
        qt = QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            subsample=1000000000,
            random_state=0,
        )
        qt.fit(A[train])
        A = qt.transform(A)
        meta["quantile"] = qt
    if mode in ["impute_indicator", "impute_indicator_quantile"]:
        A = np.concatenate([A, missing.astype(float)], axis=1)
    return A, meta


def transform(X, meta):
    mode = meta["mode"]
    missing = np.isnan(X)
    if mode == "native":
        return X
    if mode == "indicator_only":
        return missing.astype(float)
    A = np.where(missing, meta["median"][None, :], X)
    if "quantile" in meta:
        A = meta["quantile"].transform(A)
    if mode in ["impute_indicator", "impute_indicator_quantile"]:
        A = np.concatenate([A, missing.astype(float)], axis=1)
    return A


def main():
    d, man = load()
    train = (d.split == TRAIN).to_numpy()
    dev = (d.split == DEV).to_numpy()
    development = train | dev
    y = d.ground_truth_binary.to_numpy(int)
    summary = []
    models = []
    specs = [
        ("H0", "H0", "native"),
        ("H1", "H1", "native"),
        ("H2", "H2", "native"),
        ("H3", "H2", "impute_indicator"),
        ("H4", "H2", "impute_indicator_quantile"),
        ("H5", "H2", "impute"),
        ("H6", "H2", "indicator_only"),
    ]
    for name, fset, mode in specs:
        path = RUN / "models" / (name + ".joblib")
        done = RUN / "models" / (name + ".json")
        if done.exists():
            models.append(json.loads(done.read_text()))
            continue
        cols = man["feature_sets"][fset]
        X = d[cols].to_numpy(float)
        A, pre = prepare(X, train, mode)
        start = time.time()
        est = HistGradientBoostingClassifier(**PARAMS).fit(A[train], y[train])
        fit_seconds = time.time() - start
        b = {
            "estimator": est,
            "features": cols,
            "preprocessing": pre,
            "parameters": PARAMS,
            "train_keys_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    d.loc[train, ["dataset_id", "image_id", "cell_id", "marker"]],
                    index=False,
                )
                .to_numpy()
                .tobytes()
            ).hexdigest(),
        }
        joblib.dump(b, path, compress=3)
        s = np.full(len(d), np.nan)
        start = time.time()
        s[development] = est.predict_proba(A[development])[:, 1]
        infer = time.time() - start
        np.save(RUN / "predictions" / (name + "_development.npy"), s)
        if name == "H0":
            old = joblib.load(
                ROOT / "analyses/cd8_e1_e4_stability_20261023/models/E2.joblib"
            )
            previous = old["estimator"].predict_proba(X[development])[:, 1]
            maxerr = float(np.max(abs(previous - s[development])))
            savej(
                RUN / "results/H0_LEGACY_REPRODUCTION.json",
                {
                    "max_abs_development_score_difference": maxerr,
                    "n_development_cells": int(development.sum()),
                    "same_environment": True,
                    "passed": maxerr <= 1e-12,
                },
            )
            if maxerr > 1e-12:
                raise ValueError("H0 failed legacy reproduction")
        ts, grid = thresholds(d.loc[dev], s)
        pd.DataFrame(grid).to_csv(
            RUN / "results" / (name + "_development_threshold_grid.csv"), index=False
        )
        ap = macro_ap(d.loc[dev], s)
        stats = metric(y[dev], s[dev], 0.5)
        n_nodes = sum(len(a[0].nodes) for a in est._predictors)
        record = {
            "model": name,
            "family": "HGB",
            "feature_set": fset,
            "missing_mode": mode,
            "n_input_features": A.shape[1],
            "fit_seconds": fit_seconds,
            "development_inference_seconds": infer,
            "development_macro_AP": ap,
            "thresholds": ts,
            "active_parameters": n_nodes,
            "parameter_count_interpretation": "HGB total learned tree nodes; not equivalent to neural scalar weights",
            "process_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "model_sha256": sha(path),
            "feature_file_sha256": man["feature_sha256"],
            "deployment_candidate": name != "H6",
        }
        record.update({"dev_fixed_" + k: v for k, v in stats.items()})
        savej(done, record)
        models.append(record)
        summary.append(record)
        print("HGB_DONE", json.dumps(record), flush=True)
    pd.DataFrame(
        [
            {k: v for k, v in x.items() if not isinstance(v, (dict, list))}
            for x in models
        ]
    ).to_csv(RUN / "results/HGB_DEVELOPMENT.csv", index=False)
    savej(
        RUN / "models/HGB_COMPLETE.json",
        {
            "models": [x["model"] for x in models],
            "n_fits": 7,
            "external_metrics_inspected": False,
            "feature_sha256": man["feature_sha256"],
            "code_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    main()
