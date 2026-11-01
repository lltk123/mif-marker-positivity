import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
from pathlib import Path
import json, hashlib, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from runtime_paths import ROOT, RUN, SOURCE_ROOT

TRAIN = "training_WSIs"
DEV = "internal_val_WSIs"
TEST = "external_val_TMA_cores"


def savej(path, d):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, default=str))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load():
    manifest = json.loads((RUN / "data/FEATURES_COMPLETE.json").read_text())
    d = pd.read_parquet(RUN / "data/features.parquet")
    assert len(d) == 1077669 and np.array_equal(d.row_id.to_numpy(), np.arange(len(d)))
    assert not d.duplicated(["dataset_id", "image_id", "cell_id", "marker"]).any()
    return d, manifest


def metric(y, s, t=0.5):
    y = np.asarray(y, int)
    s = np.asarray(s)
    finite = np.isfinite(s)
    n = len(y)
    if not finite.all():
        return {
            "n_cells": n,
            "n_score_failed": int((~finite).sum()),
            "coverage": float(finite.mean()),
        }
    p = s > np.asarray(t, dtype=s.dtype)
    tp = int(np.sum((y == 1) & p))
    fp = int(np.sum((y == 0) & p))
    fn = int(np.sum((y == 1) & ~p))
    tn = int(np.sum((y == 0) & ~p))
    pos = tp + fn
    neg = tn + fp
    return {
        "n_cells": n,
        "n_score_failed": 0,
        "coverage": 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / pos if pos else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "specificity": tn / neg if neg else np.nan,
        "FPR": fp / neg if neg else np.nan,
        "AP": float(average_precision_score(y, s)) if pos else np.nan,
        "ROC_AUC": float(roc_auc_score(y, s)) if pos and neg else np.nan,
        "true_positive_fraction": pos / n if n else np.nan,
        "predicted_positive_fraction": float(p.mean()) if n else np.nan,
        "positive_fraction_absolute_error": abs(float(p.mean()) - float(y.mean()))
        if n
        else np.nan,
    }


def macro_ap(d, s):
    out = []
    for uid, g in d.groupby("image_id", sort=True):
        ix = g.index.to_numpy()
        y = g.ground_truth_binary.to_numpy(int)
        if len(np.unique(y)) == 2:
            out.append(average_precision_score(y, np.asarray(s)[ix]))
    return float(np.mean(out)) if out else np.nan


def thresholds(d, s):
    grid = np.arange(1, 1000, dtype=np.float64) / 1000.0
    records = []
    Y = d.ground_truth_binary.to_numpy(int)
    S = np.asarray(s)
    idx = d.index.to_numpy()
    ss = S[idx] if len(S) != len(d) or (len(d) and idx[-1] >= len(d)) else S
    # Callers pass full aligned arrays; preserve original score dtype for strict comparisons.
    vals = []
    for uid, g in d.groupby("image_id", sort=True):
        ix = g.index.to_numpy()
        scores = S[ix]
        y = g.ground_truth_binary.to_numpy(int)
        order = np.argsort(scores, kind="stable")
        ys = y[order]
        sv = scores[order]
        cum = np.r_[0, np.cumsum(ys, dtype=np.int64)]
        j = np.searchsorted(sv, grid.astype(sv.dtype), side="right")
        tp = ys.sum() - cum[j]
        fp = len(ys) - j - tp
        fn = ys.sum() - tp
        recall = tp / ys.sum()
        precision = np.divide(tp, tp + fp, out=np.zeros(len(grid)), where=(tp + fp) > 0)
        f1 = np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros(len(grid)),
            where=(2 * tp + fp + fn) > 0,
        )
        vals.append((precision, recall, f1))
    precision = np.mean([a[0] for a in vals], axis=0)
    recall = np.mean([a[1] for a in vals], axis=0)
    f1 = np.mean([a[2] for a in vals], axis=0)
    best = float(f1.max())
    ties = np.where(np.abs(f1 - best) <= 1e-15)[0]
    j = sorted(ties, key=lambda k: (abs(grid[k] - 0.5), -grid[k]))[0]
    selected = {"fixed_0.5": 0.5, "dev_F1": float(grid[j])}
    for target in [0.90, 0.95]:
        good = np.where(recall >= target)[0]
        selected["dev_recall_" + str(int(target * 100))] = (
            float(grid[good[-1]]) if len(good) else 0.0
        )
    for k in range(len(grid)):
        records.append(
            {
                "threshold": float(grid[k]),
                "precision_macro": float(precision[k]),
                "recall_macro": float(recall[k]),
                "f1_macro": float(f1[k]),
            }
        )
    return selected, records


def risks(d):
    p = json.loads(
        (ROOT / "analyses/cd8_tabm_robust_20261030/data/risk_strata.json").read_text()
    )
    y = d.ground_truth_binary.to_numpy(int)
    whole = d.audit_raw_whole_mean.to_numpy(float)
    area = d.audit_raw_neighbor_edge_area_fraction.to_numpy(float)
    diff = d.audit_raw_neighbor_edge_mean_diff.to_numpy(float)
    missing = d[[c for c in d if c.startswith("h1_")]].isna().any(axis=1).to_numpy()
    return {
        "all": np.ones(len(d), bool),
        "weak_positive": (y == 1) & (whole <= p["weak_positive_whole_mean_q25"]),
        "contact_bright_proxy_negative": (y == 0)
        & (area > 0)
        & (diff > p["contact_bright_diff_q75"]),
        "zero_target_signal": d.repair_whole_no_nonzero_signal.to_numpy(bool),
        "zero_crop": d.crop_all_zero.to_numpy(bool),
        "any_spatial_missing": missing,
        "boundary_abnormal": d.qc_truncated.to_numpy(bool)
        | d.qc_context_clipped.to_numpy(bool)
        | (d.crop_target_fraction.to_numpy(float) < 1),
        "no_boundary_abnormal": ~(
            d.qc_truncated.to_numpy(bool)
            | d.qc_context_clipped.to_numpy(bool)
            | (d.crop_target_fraction.to_numpy(float) < 1)
        ),
    }
