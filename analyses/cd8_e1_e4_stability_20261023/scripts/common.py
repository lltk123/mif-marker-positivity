"""Shared experiment paths, frozen settings, identity checks and provenance helpers."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

A = Path(__file__).resolve().parents[1]
PATHS = json.loads((A / "audit/paths.json").read_text())
OLD = Path(PATHS["old"])
G = Path(PATHS["gmm"])
ROOT = Path(PATHS["root"])
DATA = ROOT / "datasets/PhenoBIC-Data"
KEYS = ["dataset_id", "image_id", "cell_id"]
I8 = [
    "norm_whole_mean",
    "norm_whole_median",
    "norm_whole_std",
    "norm_whole_p10",
    "norm_whole_p90",
    "norm_whole_p95",
    "norm_whole_IQR",
    "norm_integrated_intensity",
]
S9 = [
    "edge_inner_normalized_diff",
    "edge_signal_enrichment",
    "high_signal_edge_coverage",
    "angular_R1",
    "angular_R2",
    "signal_centroid_offset_normalized",
    "contact_noncontact_normalized_diff",
    "neighbor_edge_area_fraction",
    "contact_neighbor_band_normalized_diff",
]
SETS = {
    "E1": I8,
    "E2": I8 + S9,
    "E3": I8 + ["GMM_score"],
    "E4": I8 + ["GMM_score"] + S9,
}
PARAMS = dict(
    learning_rate=0.06,
    max_iter=150,
    max_leaf_nodes=15,
    min_samples_leaf=80,
    l2_regularization=2.0,
    early_stopping=False,
    random_state=0,
)
PERTURB = [
    "gain_0.5",
    "gain_2",
    "bias_0.05",
    "bias_0.10",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "erode_1",
    "dilate_1",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, default=str)
    if path.exists():
        assert path.read_text() == content, "Refuse incompatible existing " + str(path)
    else:
        path.write_text(content)


def key_hash(df):
    return hashlib.sha256(
        df[KEYS].sort_values(KEYS).to_csv(index=False, line_terminator="\n").encode()
    ).hexdigest()


def metrics(y, s, t=0.5):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    good = np.isfinite(s)
    n = len(y)
    nvalid = int(good.sum())
    out = dict(
        n_total=n,
        n_scored=nvalid,
        n_failed=n - nvalid,
        prediction_coverage=nvalid / n if n else np.nan,
        complete_population_metrics=bool(good.all()),
    )
    if not good.all():
        for k in [
            "TP",
            "FP",
            "FN",
            "TN",
            "precision",
            "recall",
            "F1",
            "specificity",
            "ROC_AUC",
            "AP",
            "true_prevalence",
            "predicted_prevalence",
            "prevalence_absolute_error",
        ]:
            out[k] = np.nan
        return out
    pred = s > t
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    out.update(
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
        ROC_AUC=roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        AP=average_precision_score(y, s) if y.sum() else np.nan,
        true_prevalence=float(y.mean()) if n else np.nan,
        predicted_prevalence=float(pred.mean()) if n else np.nan,
        prevalence_absolute_error=abs(float(y.mean() - pred.mean())) if n else np.nan,
    )
    return out
