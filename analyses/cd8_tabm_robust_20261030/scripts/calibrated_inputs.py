"""Strict contracts for pixel-recomputed outer-percentile features.

The shared extractor owns image normalization. This adapter never transforms
old summary intensities to approximate new features, and never reads Pan-M.
"""
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from common_tabm import KEYS, TRAIN, sha256, load_json, validate_keys, key_hash

METHOD = "E1_outer_P1_P99"
RAW_RISK_FEATURES = [
    "whole_mean",
    "neighbor_edge_area_fraction",
    "neighbor_edge_mean_diff",
    "local_reference_median",
    "sector_signal_cv",
    "edge_inner_mean_diff",
]


def source_paths(root):
    project = Path(root).parents[1]
    old = project / "analyses/cd8_morphospatial_20261023"
    shared = project / "analyses/cd8_robust_calibration_20261030"
    return {
        "features": shared / "PhenoBIC_CD8_features.parquet",
        "manifest": shared / "extraction_manifest.json",
        "protocol": shared / "PROTOCOL.json",
        "bounds": shared / "calibration/unit_bounds.csv",
        "raw_features": old / "PhenoBIC_CD8_features.parquet",
        "labels": project / "analyses/cd8_e1_e4_stability_20261023/audit/cells.parquet",
        "risk": project / "analyses/cd8_tabm_20261029/data/risk_strata.json",
        "dictionary": shared / "FEATURE_DICTIONARY.csv",
        "experiment_protocol": project
        / "analyses/cd8_robust_comparison_20261030/RETRAIN_PROTOCOL.json",
        "feature_code": old / "scripts/mif_features.py",
    }


def verify_source_contract(paths, requested):
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Full calibrated inputs are required; missing:\n" + "\n".join(missing)
        )
    manifest = load_json(paths["manifest"])
    if manifest.get("status") != "complete" or manifest.get("method_id") != METHOD:
        raise ValueError(
            "Shared extraction is incomplete or uses a different normalization."
        )
    if manifest.get("rows") != 1817783:
        raise ValueError("Unexpected shared full-data cell count.")
    if manifest.get("feature_file") != paths["features"].name:
        raise ValueError("Manifest feature filename mismatch.")
    if manifest.get("calibration_file") != "calibration/unit_bounds.csv":
        raise ValueError("Manifest calibration filename mismatch.")
    checked = {
        "feature_sha256": "features",
        "protocol_sha256": "protocol",
        "calibration_sha256": "bounds",
        "source_feature_sha256": "raw_features",
        "dictionary_sha256": "dictionary",
    }
    for field, name in checked.items():
        if manifest.get(field) != sha256(paths[name]):
            raise ValueError("Shared input hash mismatch: " + field)
    names = manifest.get("feature_columns", [])
    if len(names) != 90 or len(set(names)) != 90 or set(names) != set(requested["C"]):
        raise ValueError(
            "Shared extractor must provide the complete original 90-feature schema."
        )
    bounds = pd.read_csv(paths["bounds"], float_precision="round_trip")
    required = {"image_id", "L", "U", "calibration_invalid", "n_eligible"}
    if not required.issubset(bounds.columns) or bounds.image_id.duplicated().any():
        raise ValueError("Missing or duplicated image calibration units.")
    if not bounds.calibration_invalid.isin([True, False]).all():
        raise ValueError("Invalid unit calibration flags.")
    if len(bounds) != 48:
        raise ValueError("Expected 3 training + 3 internal WSIs + 42 external cores.")
    signature = {
        "method_id": METHOD,
        "shared_manifest_sha256": sha256(paths["manifest"]),
        **{field: manifest[field] for field in checked},
        "all_keys_sha256": manifest["all_keys_sha256"],
        "experiment_protocol_sha256": sha256(paths["experiment_protocol"]),
    }
    return manifest, bounds, signature


def with_marker(frame):
    frame = frame.copy()
    if "marker" not in frame:
        frame["marker"] = "CD8"
    validate_keys(frame)
    return frame


def assemble_cells(features, raw, annotations, requested):
    """Join shuffled tables by explicit IDs, preserving every original instance."""
    data, raw, ann = map(with_marker, (features, raw, annotations))
    forbidden = {
        "ground_truth_original",
        "ground_truth_binary",
        "GMM_score",
        "GMM_prediction",
    }
    if forbidden.intersection(data.columns):
        raise ValueError("Shared image features must be label- and GMM-free.")
    required = set(requested["C"]) | {
        "mask_type",
        "split",
        "context",
        "source_image_id",
        "observed_pixel_count",
        "observed_mean_CD8",
        "raw_observed_mean_CD8",
        "calibration_valid",
    }
    if not required.issubset(data.columns):
        raise ValueError(
            "Shared feature columns missing: "
            + ", ".join(sorted(required - set(data.columns)))
        )
    if (
        len(requested["A"]) != 8
        or len(requested["B"]) != 27
        or len(requested["C"]) != 90
    ):
        raise ValueError("A/B/C must request 8/27/90 features.")
    morphology = [name for name in requested["B"] if name not in requested["A"]]
    raw_columns = list(
        dict.fromkeys(KEYS + morphology + RAW_RISK_FEATURES + ["split", "context"])
    )
    rename = {name: "audit_raw_" + name for name in RAW_RISK_FEATURES}
    rename.update({name: "_original_" + name for name in morphology})
    rename.update({"split": "_original_split", "context": "_original_context"})
    audit = raw[raw_columns].rename(columns=rename)
    data = data.merge(
        audit, on=KEYS, how="outer", validate="one_to_one", indicator=True
    )
    if not data["_merge"].eq("both").all():
        raise ValueError("Calibrated and original raw feature keys differ.")
    data = data.drop(columns="_merge")
    for name in ["split", "context"]:
        if not data[name].eq(data["_original_" + name]).all():
            raise ValueError("Original split/context changed: " + name)
    for name in morphology:
        np.testing.assert_array_equal(
            data[name].to_numpy(),
            data["_original_" + name].to_numpy(),
            err_msg="Geometry changed: " + name,
        )
    data = data.drop(columns=[c for c in data if c.startswith("_original_")])
    ann_columns = KEYS + [
        "ground_truth_original",
        "ground_truth_binary",
        "GMM_score",
        "GMM_prediction",
        "pixel_count",
        "mean_CD8",
    ]
    data = data.merge(
        ann[ann_columns], on=KEYS, how="outer", validate="one_to_one", indicator=True
    )
    if not data["_merge"].eq("both").all():
        raise ValueError("Calibrated feature and annotation keys differ.")
    data = data.drop(columns="_merge").sort_values(KEYS).reset_index(drop=True)
    validate_keys(data)
    if not data.mask_type.eq("expanded_nucleus").all():
        raise ValueError("Unexpected mask semantics.")
    if not np.array_equal(data.observed_pixel_count, data.pixel_count):
        raise ValueError("Instance pixel counts changed.")
    # Deliberately compare the RAW audit mean, not normalized observed_mean_CD8.
    if not np.allclose(
        data.raw_observed_mean_CD8, data.mean_CD8, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("Raw instance intensity mapping changed.")
    if (
        data.calibration_valid.isna().any()
        or not data.calibration_valid.isin([True, False]).all()
    ):
        raise ValueError("Calibration validity must be explicit for every cell.")
    valid = data.calibration_valid.to_numpy(bool)
    means = data.observed_mean_CD8.to_numpy(float)
    finite = np.isfinite(means) & valid
    if ((means[finite] < -1e-12) | (means[finite] > 1 + 1e-12)).any():
        raise ValueError("Normalized observations are outside [0,1].")
    if np.isinf(data[requested["C"]].to_numpy(float)).any():
        raise ValueError("Infinite normalized feature.")
    data["normalization_method"] = METHOD
    return data


def attach_bounds(data, bounds):
    lookup = bounds.set_index("image_id")
    if set(data.image_id.unique()) != set(lookup.index):
        raise ValueError("Calibration-unit keys do not match feature image keys.")
    invalid = data.image_id.map(lookup.calibration_invalid)
    if not np.array_equal(
        ~data.calibration_valid.to_numpy(bool), invalid.to_numpy(bool)
    ):
        raise ValueError("Cell calibration validity differs from the unit audit.")
    for name in ["L", "U", "n_eligible"]:
        data["calibration_" + name] = data.image_id.map(lookup[name])
    return data


def frozen_raw_risk(data, original_rules, source_path):
    """Reuse old cohorts, verifying cutoffs against the same labelled training set."""
    risk = copy.deepcopy(original_rules)
    if (
        risk.get("fit_split") != TRAIN
        or risk.get("weak_comparison") != "<="
        or risk.get("contact_comparison") != "strict >"
    ):
        raise ValueError("Existing risk definitions are incompatible.")
    training = data.loc[data.split.eq(TRAIN) & data.ground_truth_binary.isin([0, 1])]
    rules = [
        (
            "weak_positive_whole_mean_q25",
            "whole_mean",
            0.25,
            training.ground_truth_binary.eq(1),
        ),
        (
            "contact_bright_diff_q75",
            "neighbor_edge_mean_diff",
            0.75,
            training.audit_raw_neighbor_edge_area_fraction.gt(0),
        ),
        (
            "local_reference_q75",
            "local_reference_median",
            0.75,
            np.ones(len(training), dtype=bool),
        ),
        ("sector_cv_q75", "sector_signal_cv", 0.75, np.ones(len(training), dtype=bool)),
    ]
    for parameter, name, quantile, eligible in rules:
        observed = training.loc[eligible, "audit_raw_" + name].to_numpy(float)
        observed = observed[np.isfinite(observed)]
        expected = float(np.quantile(observed, quantile)) if len(observed) else None
        stored = risk.get(parameter)
        if expected is None:
            if stored is not None:
                raise ValueError("Unexpected stored risk cutoff: " + parameter)
        elif stored is None or not np.isclose(stored, expected, rtol=1e-12, atol=1e-14):
            raise ValueError(
                "Original training risk cutoff does not reproduce: " + parameter
            )
    risk.update(
        feature_columns={name: "audit_raw_" + name for name in RAW_RISK_FEATURES},
        feature_scale="original raw intensity; unchanged old/new risk cohorts",
        source_rules_path=str(source_path),
        source_rules_sha256=sha256(source_path),
        model_predictors_exclude_audit_raw_columns=True,
    )
    return risk


def audit_risk_frame(frame, config):
    """Provide raw audit columns to legacy risk logic without altering model X."""
    mapping = config.get("feature_columns", {})
    if not mapping:
        return frame
    missing = set(mapping.values()) - set(frame.columns)
    if missing:
        raise ValueError(
            "Missing original raw risk features: " + ", ".join(sorted(missing))
        )
    audit = frame.copy()
    for name, column in mapping.items():
        audit[name] = audit[column]
    return audit
