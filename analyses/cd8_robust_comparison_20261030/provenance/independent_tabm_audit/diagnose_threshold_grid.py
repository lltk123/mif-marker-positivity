from pathlib import Path
import json, numpy as np, pandas as pd, hashlib

R = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_tabm_robust_20261030"
)
model = "TabM_B_s1"
keys = ["dataset_id", "image_id", "cell_id", "marker"]
meta = json.loads((R / "models" / model / "metadata.json").read_text())
candidate = meta["candidate_model_id"]
cells = pd.read_parquet(
    R / "data/cells.parquet", columns=keys + ["ground_truth_binary", "split"]
)
idx = np.load(R / "data/validation_indices.npy")
v = cells.iloc[idx].copy()
members = np.load(R / "models" / candidate / "validation_members.npy")
v["validation_score"] = members.mean(1)
pred = pd.read_parquet(
    R / "predictions" / (model + ".parquet"),
    columns=keys + ["score", "prediction_validation_selected", "prediction_fixed_0_5"],
)
v = v.merge(pred, on=keys, validate="one_to_one", sort=False)
grid = pd.read_csv(
    R / "models" / candidate / "threshold_grid.csv", float_precision="round_trip"
)
th = grid.threshold.to_numpy()
results = {}
group = v.image_id.to_numpy()
y = v.ground_truth_binary.to_numpy()
for name, col, dtype in [
    ("validation32", "validation_score", np.float32),
    ("validation64", "validation_score", np.float64),
    ("inference32", "score", np.float32),
    ("inference64", "score", np.float64),
]:
    s = v[col].to_numpy(dtype=dtype)
    f1 = []
    for g in np.unique(group):
        mask = group == g
        ss = s[mask]
        yy = y[mask]
        pos = np.sort(ss[yy == 1])
        neg = np.sort(ss[yy == 0])
        thresholds = th.astype(dtype)
        tp = len(pos) - np.searchsorted(pos, thresholds, side="right")
        fp = len(neg) - np.searchsorted(neg, thresholds, side="right")
        fn = len(pos) - tp
        f1.append(2 * tp / (2 * tp + fp + fn))
    macro = np.mean(f1, axis=0)
    diff = macro - grid.validation_macro_F1.to_numpy()
    bad = np.abs(diff) > 1e-12
    results[name] = {
        "max_grid_error": float(abs(diff).max()),
        "mismatching_thresholds": th[bad].tolist(),
        "grid_differences": diff[bad].tolist(),
        "winner": float(th[np.argmax(macro)]),
    }
score_diff = v.validation_score.to_numpy(float) - v.score.to_numpy(float)
results["validation_vs_final"] = {
    "max_score_abs_delta": float(abs(score_diff).max()),
    "n_not_bitwise_equal": int((score_diff != 0).sum()),
    "n_validation_cells": len(v),
}
results["stored_prediction_vs_float64"] = {
    "threshold": meta["threshold_validation"],
    "validation_selected_mismatches_full": int(
        (
            pred.prediction_validation_selected.to_numpy()
            != (pred.score.to_numpy(float) > meta["threshold_validation"])
        ).sum()
    ),
    "fixed_0_5_mismatches_full": int(
        (
            pred.prediction_fixed_0_5.to_numpy() != (pred.score.to_numpy(float) > 0.5)
        ).sum()
    ),
}
print(json.dumps(results, indent=2))
