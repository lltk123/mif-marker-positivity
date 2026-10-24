"""Check numerical score parsing and strict decision-threshold conventions."""

from pathlib import Path
import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score, average_precision_score

A = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_morphospatial_20261023"
)
G = A.parent / "cd8_scimap_gmm_20261023"
pieces = []
for _, r in pd.read_csv(G / "results/phenobic/cell_table_index.csv").iterrows():
    a = pd.read_csv(
        G / r["file"],
        usecols=["ground_truth_binary", "rescale_score", "prediction_binary"],
        float_precision="round_trip",
    )
    a["split"] = r.unit_id.split("/")[1]
    pieces.append(a)
x = pd.concat(pieces)
x = x[x.ground_truth_binary.isin([0, 1])]
old = pd.read_csv(
    G / "results/phenobic/metrics_summary.csv", float_precision="round_trip"
)
out = []
for split, d in x.groupby("split"):
    prior = old[(old.level == "split") & (old.group == split)].iloc[0]
    auc = roc_auc_score(d.ground_truth_binary, d.rescale_score)
    ap = average_precision_score(d.ground_truth_binary, d.rescale_score)
    out.append(
        dict(
            split=split,
            auc=auc,
            auc_delta=auc - prior.ROC_AUC,
            ap=ap,
            ap_delta=ap - prior.AP,
        )
    )
print(json.dumps(out, indent=2))
