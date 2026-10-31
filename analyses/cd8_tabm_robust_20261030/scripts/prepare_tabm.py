"""Audit existing features and fit all preprocessing using original training labels only."""
import argparse
import importlib.util
import time
import warnings
from common_tabm import *
from calibrated_inputs import (
    METHOD,
    RAW_RISK_FEATURES,
    source_paths,
    verify_source_contract,
    assemble_cells,
    attach_bounds,
    frozen_raw_risk,
)
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer


def main(root):
    root = Path(root)
    project = root.parents[1]
    old = project / "analyses/cd8_morphospatial_20261023"
    paths = source_paths(root)
    source, labels, dictionary = paths["features"], paths["labels"], paths["dictionary"]
    spec = importlib.util.spec_from_file_location(
        "source_mif_features", old / "scripts/mif_features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    requested = {
        "A": mod.WHOLE,
        "B": mod.WHOLE + mod.MORPH,
        "C": mod.WHOLE
        + mod.MORPH
        + [x for x in mod.FEATURES if x not in mod.WHOLE + mod.MORPH],
    }
    assert [len(requested[x]) for x in "ABC"] == [8, 27, 90]
    shared_manifest, bounds, normalization_signature = verify_source_contract(
        paths, requested
    )
    manifest = [
        dict(role=name, path=str(p), bytes=p.stat().st_size, sha256=sha256(p))
        for name, p in sorted(paths.items())
    ]
    saved = root / "data/DATA_READY.json"
    if saved.exists():
        prior = load_json(saved)
        assert prior["inputs"] == manifest and prior["preparation_sha256"] == sha256(
            __file__
        )
        assert prior["adapter_sha256"] == sha256(root / "scripts/calibrated_inputs.py")
        assert prior["normalization_signature"] == normalization_signature
        print("Compatible prepared data already exists", flush=True)
        return
    features = pd.read_parquet(source)
    raw_columns = list(
        dict.fromkeys(KEYS[:3] + mod.MORPH + RAW_RISK_FEATURES + ["split", "context"])
    )
    raw = pd.read_parquet(paths["raw_features"], columns=raw_columns)
    ann = pd.read_parquet(
        labels,
        columns=KEYS[:3]
        + [
            "ground_truth_original",
            "ground_truth_binary",
            "GMM_score",
            "GMM_prediction",
            "pixel_count",
            "mean_CD8",
        ],
    )
    data = attach_bounds(assemble_cells(features, raw, ann, requested), bounds)
    del features, raw, ann
    assert len(data) == 1817783
    assert key_hash(data) == shared_manifest["all_keys_sha256"]
    labeled = data.ground_truth_binary.isin([0, 1])
    train = labeled & data.split.eq(TRAIN)
    valid = labeled & data.split.eq(VALID)
    expected = {TRAIN: 101057, VALID: 539132, TEST: 437480}
    counts = data.loc[labeled].groupby("split").size().to_dict()
    assert counts == expected, counts
    assert (
        data.loc[train, "image_id"].nunique()
        == data.loc[valid, "image_id"].nunique()
        == 3
    )
    hashes = []
    for split, part in data.groupby("split", sort=True):
        for selection, frame in [
            ("all", part),
            ("labelled", part.loc[part.ground_truth_binary.isin([0, 1])]),
        ]:
            dest = root / "data" / f"keys_{split}_{selection}.csv.gz"
            frame[KEYS].to_csv(dest, index=False, compression="gzip")
            hashes.append(
                dict(
                    normalization_method=METHOD,
                    split=split,
                    selection=selection,
                    n=len(frame),
                    key_sha256=key_hash(frame),
                    file_sha256=sha256(dest),
                )
            )
    pd.DataFrame(hashes).to_csv(root / "results/key_hashes.csv", index=False)
    # Pixel-recomputed calibrated inputs remain unchanged; the imputation and transform are explicit fitted model components.
    raw = data[requested["C"]].to_numpy(dtype=np.float64)
    assert not np.isinf(raw).any(), "Unexpected infinity in saved features"
    tr = raw[train.to_numpy()]
    reasons = {}
    for j, name in enumerate(requested["C"]):
        finite = tr[np.isfinite(tr[:, j]), j]
        if len(finite) == 0:
            reasons[name] = "all_missing_in_training"
        elif np.min(finite) == np.max(finite):
            reasons[name] = "constant_observed_training_values"
    active = [x for x in requested["C"] if x not in reasons]
    active_idx = [requested["C"].index(x) for x in active]
    imputer = SimpleImputer(strategy="median")
    xtrain = imputer.fit_transform(tr[:, active_idx])
    # Official example: noisy training-only observations fit the quantile map; transform clean values.
    noise_rng = np.random.default_rng(0)
    noisy_fit = xtrain + noise_rng.normal(0, 1e-5, xtrain.shape).astype(xtrain.dtype)
    transformer = QuantileTransformer(
        n_quantiles=max(min(len(xtrain) // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
        random_state=0,
    )
    transformer.fit(noisy_fit)
    joblib.dump(
        dict(
            imputer=imputer,
            transformer=transformer,
            features=active,
            fit_split=TRAIN,
            normalization_method=METHOD,
            normalization_signature=normalization_signature,
        ),
        root / "models/preprocessor.joblib",
    )
    X = np.lib.format.open_memmap(
        root / "data/X.npy", mode="w+", dtype=np.float32, shape=(len(data), len(active))
    )
    for start in range(0, len(data), 65536):
        X[start : start + 65536] = transformer.transform(
            imputer.transform(raw[start : start + 65536, active_idx])
        ).astype(np.float32)
    assert np.isfinite(X).all()
    X.flush()
    effective = {g: [f for f in requested[g] if f in active] for g in requested}
    groups = {
        g: dict(
            requested=requested[g],
            effective=effective[g],
            matrix_indices=[active.index(f) for f in effective[g]],
            excluded={f: reasons[f] for f in requested[g] if f in reasons},
        )
        for g in requested
    }
    save_json(root / "FEATURE_SETS.json", groups)
    pd.DataFrame(
        [
            dict(
                normalization_method=METHOD,
                feature_set=g,
                position=i,
                feature=f,
                included=f in active,
                exclusion_reason=reasons.get(f, ""),
            )
            for g in requested
            for i, f in enumerate(requested[g])
        ]
    ).to_csv(root / "results/feature_manifest.csv", index=False)
    if dictionary.exists():
        (root / "FEATURE_DICTIONARY.csv").write_bytes(dictionary.read_bytes())
    rows = []
    for split, part in data.groupby("split", sort=True):
        for name in requested["C"]:
            a = part[name].to_numpy(float)
            rows.append(
                dict(
                    normalization_method=METHOD,
                    split=split,
                    feature=name,
                    n=len(a),
                    n_missing=int(np.isnan(a).sum()),
                    missing_rate=float(np.isnan(a).mean()),
                    n_zero=int((a == 0).sum()),
                    excluded_reason=reasons.get(name, ""),
                )
            )
    pd.DataFrame(rows).to_csv(root / "results/feature_availability.csv", index=False)
    risk = frozen_raw_risk(data, load_json(paths["risk"]), paths["risk"])
    save_json(root / "data/risk_strata.json", risk)
    data.to_parquet(root / "data/cells.parquet", index=False, compression="zstd")
    np.save(root / "data/train_indices.npy", np.flatnonzero(train.to_numpy()))
    np.save(root / "data/validation_indices.npy", np.flatnonzero(valid.to_numpy()))
    info = dict(
        normalization_method=METHOD,
        normalization_signature=normalization_signature,
        adapter_sha256=sha256(root / "scripts/calibrated_inputs.py"),
        calibration_invalid_cells=int((~data.calibration_valid).sum()),
        calibration_invalid_units=int(bounds.calibration_invalid.sum()),
        inputs=manifest,
        preparation_sha256=sha256(__file__),
        cells=len(data),
        labeled_counts=counts,
        mask_type="expanded_nucleus",
        patient_ids_available=sorted(str(x) for x in data.patient_id.dropna().unique()),
        grouping="Original WSI; no random cell split; cores are not independent patients",
        preprocessing_fit_count=int(train.sum()),
        requested_features=len(requested["C"]),
        effective_features=len(active),
        excluded=reasons,
        key_hash=key_hash(data),
        cells_sha256=sha256(root / "data/cells.parquet"),
        matrix_sha256=sha256(root / "data/X.npy"),
        preprocessor_sha256=sha256(root / "models/preprocessor.joblib"),
        train_indices_sha256=sha256(root / "data/train_indices.npy"),
        validation_indices_sha256=sha256(root / "data/validation_indices.npy"),
    )
    save_json(saved, info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    main(p.parse_args().run_dir)
