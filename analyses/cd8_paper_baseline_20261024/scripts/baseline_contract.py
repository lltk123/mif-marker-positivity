"""Model-independent CD8 identity and evaluation contracts. No fitting."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
FORBIDDEN_INPUTS = {
    "cell_id",
    "dataset_id",
    "image_id",
    "marker",
    "patient_id",
    "slide_id",
    "group_id",
    "batch_id",
    "split",
    "context",
    "ground_truth_binary",
    "ground_truth_original",
    "label",
    "prediction",
    "score",
    "GMM_score",
    "GMM_prediction",
    "rescale_score",
}


def validate_keys(table):
    missing = set(KEYS) - set(table.columns)
    if missing:
        raise ValueError("Missing explicit keys: " + str(sorted(missing)))
    if table[KEYS].isna().any().any():
        raise ValueError("Null cell keys")
    for c in ["dataset_id", "image_id", "marker"]:
        if table[c].astype(str).str.strip().eq("").any():
            raise ValueError("Blank key: " + c)
    if not pd.api.types.is_integer_dtype(table.cell_id):
        raise ValueError("cell_id requires integer dtype")
    if (table.cell_id <= 0).any():
        raise ValueError("cell_id must be a positive instance ID")
    if table[KEYS].duplicated().any():
        raise ValueError("Duplicate cell keys")


def exact_join(reference, values):
    validate_keys(reference)
    validate_keys(values)
    if set(reference.columns).intersection(values.columns) - set(KEYS):
        raise ValueError("Non-key columns overlap; keep metadata from reference")
    joined = reference.merge(
        values, on=KEYS, how="outer", validate="one_to_one", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            "Keys differ from prespecified population: "
            + str(joined["_merge"].value_counts().to_dict())
        )
    return joined.drop(columns="_merge")


def key_hash(table):
    validate_keys(table)
    canonical = table[KEYS].sort_values(KEYS).to_csv(index=False, line_terminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_feature_names(names, allowed):
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate features")
    invalid = (set(names) - set(allowed)) | set(names).intersection(FORBIDDEN_INPUTS)
    if invalid:
        raise ValueError("Unapproved model input: " + str(sorted(invalid)))


def predictions_from_scores(scores, threshold, operator):
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if operator not in ["gt", "ge"]:
        raise ValueError("Specify threshold comparison: gt or ge")
    a = np.asarray(scores, dtype=float)
    out = np.full(len(a), np.nan)
    finite = np.isfinite(a)
    out[finite] = a[finite] > threshold if operator == "gt" else a[finite] >= threshold
    return out


def binary_metrics(y, scores, threshold, operator="gt"):
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s):
        raise ValueError("Length mismatch")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Only known binary labels enter metrics")
    n = len(y)
    finite = np.isfinite(s)
    result = dict(
        n_total=n,
        n_scored=int(finite.sum()),
        n_failed=int((~finite).sum()),
        prediction_coverage=float(finite.mean()) if n else np.nan,
        complete_population_metrics=bool(n and finite.all()),
        n_true_positive=int(y.sum()),
        no_true_positives=bool(n and not y.sum()),
    )
    fields = [
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
    ]
    if not n or not finite.all():
        result.update({x: np.nan for x in fields})
        return result
    p = predictions_from_scores(s, threshold, operator).astype(bool)
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    tn = int(((y == 0) & ~p).sum())
    result.update(
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
        true_prevalence=float(y.mean()),
        predicted_prevalence=float(p.mean()),
        prevalence_absolute_error=abs(float(y.mean() - p.mean())),
        no_positive_predictions=not p.any(),
    )
    return result
