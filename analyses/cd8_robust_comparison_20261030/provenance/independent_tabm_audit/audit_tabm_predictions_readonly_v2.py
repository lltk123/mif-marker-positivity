"""Independent read-only checks of the 20261030 TabM full prediction outputs."""
from pathlib import Path
import json, hashlib, time, sys, os
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

P = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")
R = P / "analyses/cd8_tabm_robust_20261030"
PART = int(os.environ.get("AUDIT_PART", "0"))
OUT = Path("/tmp/cd8_tabm_independent_audit_20261031_v2_part" + str(PART))
OUT.mkdir(exist_ok=False)
KEYS = ["dataset_id", "image_id", "cell_id", "marker"]


def sha(f):
    h = hashlib.sha256()
    with Path(f).open("rb") as s:
        for b in iter(lambda: s.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(f):
    return json.loads(f.read_text())


start = time.time()
frozen = read_json(R / "FROZEN_TEST_PROTOCOL.json")
fs = sha(R / "FROZEN_TEST_PROTOCOL.json")
cols = KEYS + [
    "split",
    "context",
    "ground_truth_binary",
    "calibration_valid",
    "GMM_score",
]
source = pd.read_parquet(R / "data/cells.parquet", columns=cols)
va = np.load(R / "data/validation_indices.npy")
assert (
    sha(R / "data/validation_indices.npy") == frozen["signature"]["validation_indices"]
)
assert np.array_equal(
    va,
    np.flatnonzero(
        source.split.eq("internal_val_WSIs") & source.ground_truth_binary.isin([0, 1])
    ),
)
validation_source = source.iloc[va].copy()
validation_source["_position"] = np.arange(len(va))
validation_order = validation_source.sort_values(KEYS)._position.to_numpy()
cells = source.sort_values(KEYS).reset_index(drop=True)
del source, validation_source
assert (
    len(cells) == 1817783
    and not cells[KEYS].isna().any().any()
    and not cells.duplicated(KEYS).any()
)
keysha = hashlib.sha256(
    cells[KEYS].to_csv(index=False, lineterminator="\n").encode()
).hexdigest()
assert keysha == read_json(R / "DELIVERY_VALIDATION.json")["keys_sha256"]
metrics = pd.read_csv(R / "results/metrics_all.csv", float_precision="round_trip")
y_all = cells.ground_truth_binary.to_numpy()
labelled = np.isin(y_all, [0, 1])
records = []
metrics_checked = 0
for model in (sorted(frozen["models"]) + ["GMM"])[PART::2]:
    t0 = time.time()
    members_delta = None
    stored_prediction_mismatches = 0
    validation_final_delta = None
    if model == "GMM":
        score = cells.GMM_score.to_numpy(float)
        meta = {"threshold_validation": 0.5}
        n_members = 0
        prediction_sha = None
    else:
        path = R / "predictions" / (model + ".parquet")
        side = read_json(path.with_suffix(".json"))
        assert side["frozen_sha256"] == fs and side["n"] == 1817783
        prediction_sha = sha(path)
        assert prediction_sha == side["sha256"]
        meta = frozen["models"][model]
        assert meta == read_json(R / "models" / model / "metadata.json")
        if model.startswith("TabM"):
            members = ["submodel_" + str(i).zfill(2) for i in range(32)]
            import pyarrow.parquet as pq

            members = [
                n
                for n in pq.ParquetFile(path).schema.names
                if n.startswith("submodel_")
            ]
            assert len(members) == 32
        else:
            members = []
        pred = (
            pd.read_parquet(
                path,
                columns=KEYS
                + ["score", "prediction_fixed_0_5", "prediction_validation_selected"]
                + members,
            )
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        assert (
            len(pred) == 1817783
            and not pred[KEYS].isna().any().any()
            and not pred.duplicated(KEYS).any()
        )
        pd.testing.assert_frame_equal(
            cells[KEYS], pred[KEYS], check_dtype=False, check_exact=True
        )
        score = pred.score.to_numpy(float)
        n_members = len(members)
        stored_prediction_mismatches = int(
            np.sum(pred.prediction_fixed_0_5.to_numpy() != (score > 0.5))
            + np.sum(
                pred.prediction_validation_selected.to_numpy()
                != (score > meta["threshold_validation"])
            )
        )
        assert stored_prediction_mismatches == 0, (
            model,
            "stored classifications disagree with formal score interpretation",
            stored_prediction_mismatches,
        )
        if members:
            values = pred[members].to_numpy(float)
            assert (
                np.isfinite(values).all()
                and (values >= 0).all()
                and (values <= 1).all()
            )
            members_delta = float(np.max(np.abs(values.mean(axis=1) - score)))
            assert members_delta <= 2e-7
            del values
        del pred
    assert np.isfinite(score).all() and (score >= 0).all() and (score <= 1).all()
    rows = metrics[metrics.model_id == model]
    max_error = 0.0
    ranks = {}
    for _, row in rows.iterrows():
        selected = labelled & cells.split.eq(row["split"]).to_numpy()
        if row.group_type == "image":
            selected &= cells.image_id.eq(row.group_id).to_numpy()
        elif row.group_type == "context":
            selected &= cells.context.eq(row.group_id).to_numpy()
        else:
            assert row.group_type == "split"
        yy = y_all[selected].astype(int)
        ss = score[selected]
        pp = ss > row.threshold
        tp = int(np.sum(pp & (yy == 1)))
        fp = int(np.sum(pp & (yy == 0)))
        fn = int(np.sum(~pp & (yy == 1)))
        tn = int(np.sum(~pp & (yy == 0)))
        assert [tp, fp, fn, tn, len(yy)] == [
            row.tp,
            row.fp,
            row.fn,
            row.tn,
            row.n_cells,
        ]
        cache = (row["split"], row.group_type, row.group_id)
        if cache not in ranks:
            ranks[cache] = (
                average_precision_score(yy, ss) if yy.sum() else np.nan,
                roc_auc_score(yy, ss) if len(np.unique(yy)) == 2 else np.nan,
            )
        ap, auc = ranks[cache]
        actual = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "average_precision": ap,
            "roc_auc": auc,
            "predicted_positive_fraction": float(pp.mean()),
            "positive_fraction_absolute_error": abs(
                float(pp.mean()) - float(yy.mean())
            ),
        }
        for metric, value in actual.items():
            assert np.isclose(
                value, row[metric], atol=1e-12, rtol=1e-12, equal_nan=True
            ), (model, cache, metric, value, row[metric])
            if np.isfinite(value):
                max_error = max(max_error, abs(value - row[metric]))
        metrics_checked += 1
    threshold_grid_error = None
    winner = None
    if model != "GMM":
        candidate = meta.get("candidate_model_id", model)
        valid = labelled & cells.split.eq("internal_val_WSIs").to_numpy()
        validation_score = score[valid]
        if model.startswith("TabM"):
            stored_members = np.load(
                R / "models" / candidate / "validation_members.npy"
            )
            validation_score = stored_members.mean(axis=1)[validation_order]
            validation_final_delta = float(
                np.max(np.abs(validation_score.astype(float) - score[valid]))
            )
            del stored_members
        validation_y = y_all[valid]
        validation_groups = cells.image_id.to_numpy()[valid]
        thresholds = np.arange(1, 100, dtype=float) / 100
        f1 = []
        promoted = []
        for image in np.unique(validation_groups):
            use = validation_groups == image
            y = validation_y[use]
            s = validation_score[use]
            positives = np.sort(s[y == 1])
            negatives = np.sort(s[y == 0])
            current = []
            for threshold_values in [
                thresholds.astype(validation_score.dtype),
                thresholds.astype(np.float64),
            ]:
                tp = len(positives) - np.searchsorted(
                    positives, threshold_values, side="right"
                )
                fp = len(negatives) - np.searchsorted(
                    negatives, threshold_values, side="right"
                )
                fn = len(positives) - tp
                current.append(2 * tp / (2 * tp + fp + fn))
            f1.append(current[0])
            promoted.append(current[1])
        macro = np.mean(f1, axis=0)
        macro64 = np.mean(promoted, axis=0)
        i = sorted(
            range(99),
            key=lambda k: (
                -macro[k],
                round(abs(thresholds[k] - 0.5), 12),
                -thresholds[k],
            ),
        )[0]
        winner = float(thresholds[i])
        assert winner == meta["threshold_validation"], (
            model,
            winner,
            meta["threshold_validation"],
        )
        assert np.isclose(
            macro[i], meta["validation_threshold_macro_F1"], atol=1e-12, rtol=1e-12
        )
        grid = pd.read_csv(
            R / "models" / candidate / "threshold_grid.csv",
            float_precision="round_trip",
        ).sort_values("threshold")
        assert len(grid) == 99 and np.array_equal(grid.threshold.to_numpy(), thresholds)
        threshold_grid_error = float(
            np.max(np.abs(grid.validation_macro_F1.to_numpy() - macro))
        )
        assert threshold_grid_error < 1e-12
        promoted_error = float(np.max(np.abs(macro64 - macro)))
        promoted_thresholds = thresholds[np.abs(macro64 - macro) > 1e-12].tolist()
    else:
        promoted_error = None
        promoted_thresholds = []
    records.append(
        {
            "model_id": model,
            "n_all_cells": len(cells),
            "keys_sha256": keysha,
            "prediction_sha256": prediction_sha,
            "metrics_rows_checked": len(rows),
            "max_metric_error": max_error,
            "n_saved_members": n_members,
            "max_member_mean_error": members_delta,
            "threshold_recomputed": winner,
            "threshold_grid_max_error": threshold_grid_error,
            "validation_vs_final_max_score_delta": validation_final_delta,
            "float64_promotion_grid_max_error": promoted_error,
            "float64_promotion_changed_thresholds": promoted_thresholds,
            "stored_prediction_mismatches": stored_prediction_mismatches,
            "seconds": time.time() - t0,
            "passed": True,
        }
    )
    print(model, "PASS", round(time.time() - t0, 2), flush=True)
result = {
    "status": "passed",
    "scope": "read-only full predicted Parquet keys/scores/member probabilities; authoritative validation grid retains saved member mean dtype; no training/GPU/Pan-M",
    "n_cells": len(cells),
    "part": PART,
    "n_models": len(records),
    "n_saved_prediction_tables": sum(r["model_id"] != "GMM" for r in records),
    "n_metric_rows_recomputed": metrics_checked,
    "keys_sha256": keysha,
    "frozen_sha256": fs,
    "metrics_sha256": sha(R / "results/metrics_all.csv"),
    "seconds": time.time() - start,
    "script_sha256": sha(__file__),
    "python": sys.version,
    "pandas": pd.__version__,
    "numpy": np.__version__,
}
pd.DataFrame(records).to_csv(OUT / "FULL_PREDICTION_AUDIT.csv", index=False)
(OUT / "FULL_PREDICTION_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
