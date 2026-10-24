"""Reusable single-channel calibration, extraction and frozen-model inference."""
import numpy as np, pandas as pd, joblib
from common import A, SETS, KEYS, I8, S9
from compact_features import extract_compact, calibration_valid
from gmm_interface import score_fixed, fit_score


def calibrate_instances(observations):
    """observations: iterator of (explicit_cell_id, unique original finite pixel array).
    Caller pools a complete WSI/core, reconstructs FOV overlaps, and does not duplicate pixels.
    Labels and prior predictions are neither accepted nor used.
    """
    rows = []
    seen = set()
    for cid, pixels in observations:
        if cid in seen:
            raise ValueError("Duplicate original instance ID in calibration unit")
        seen.add(cid)
        v = np.asarray(pixels, np.float64)
        v = v[np.isfinite(v)]
        if len(v) >= 5:
            lo, hi = np.percentile(v, [10, 90])
            rows.append((int(cid), float(lo), float(hi), len(v)))
    if not rows:
        return dict(
            L=np.nan,
            U=np.nan,
            calibration_invalid=True,
            n_eligible=0,
            L_cell_ids=[],
            U_cell_ids=[],
        )
    L = min(r[1] for r in rows)
    U = max(r[2] for r in rows)
    return dict(
        L=L,
        U=U,
        calibration_invalid=not calibration_valid(L, U),
        n_eligible=len(rows),
        L_cell_ids=[r[0] for r in rows if r[1] == L],
        U_cell_ids=[r[0] for r in rows if r[2] == U],
        finite_pixels=sum(r[3] for r in rows),
    )


def extract_cell(
    image,
    instances,
    cell_id,
    calibration,
    valid_mask=None,
    force_truncated=False,
    tissue_mask=None,
):
    return extract_compact(
        image,
        instances,
        cell_id,
        calibration["L"],
        calibration["U"],
        valid=valid_mask,
        force_truncated=force_truncated,
        tissue=tissue_mask,
    )


def predict_features(
    frame, model="E4", threshold_mode="fixed_0.5", gmm_state=None, model_directory=None
):
    """Keep keys; NaN features are passed to HGB. Input validity is a separate audit field.
    E3/E4 need original keyed GMM_score or raw observed mean plus the proper frozen state.
    This interface never fits a model, tunes thresholds or reads outcome labels.
    """
    if model not in SETS:
        raise ValueError("Unknown frozen model")
    if threshold_mode not in ["fixed_0.5", "OOF_threshold"]:
        raise ValueError("Unknown threshold mode")
    if frame[KEYS].duplicated().any():
        raise ValueError("Duplicate canonical cell key")
    bundle = joblib.load((model_directory or A / "models") / (model + ".joblib"))
    f = frame.copy()
    if "GMM_score" in bundle["features"] and "GMM_score" not in f:
        if gmm_state is None:
            raise ValueError(
                "A verified original GMM score or frozen state is required"
            )
        f["GMM_score"] = score_fixed(f.observed_raw_mean.to_numpy(), gmm_state)
    X = f[bundle["features"]].to_numpy(float)
    if np.isinf(X).any():
        raise ValueError("Infinite feature value")
    scores = bundle["estimator"].predict_proba(X)[:, 1]
    t = (
        bundle["threshold_fixed"]
        if threshold_mode == "fixed_0.5"
        else bundle["threshold_oof"]
    )
    out = f[KEYS].copy()
    out["score"] = scores
    out["prediction"] = (scores > t).astype("int8")
    out["threshold"] = t
    out["threshold_mode"] = threshold_mode
    out["score_finite"] = np.isfinite(scores)
    out["input_valid"] = f.intensity_input_valid.astype(bool)
    if model in ["E3", "E4"]:
        out["input_valid"] &= np.isfinite(f.GMM_score)
    return out
