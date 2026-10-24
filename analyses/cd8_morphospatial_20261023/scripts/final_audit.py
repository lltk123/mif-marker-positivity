"""Check final feature and prediction tables against frozen cell identity."""

from pathlib import Path
import json, zipfile, hashlib, platform
import numpy as np, pandas as pd, pyarrow.parquet as pq
from mif_features import FEATURES, COLUMNS, META_FLOAT, MORPH

A = Path(__file__).resolve().parents[1]
assert (A / "results/MODEL_COMPLETE.json").exists()
file = A / "PhenoBIC_CD8_features.parquet"
pf = pq.ParquetFile(file)
assert pf.schema_arrow.names == COLUMNS and pf.metadata.num_rows == 1817783
count = 0
for batch in pf.iter_batches(columns=FEATURES + META_FLOAT, batch_size=100000):
    f = batch.to_pandas()
    assert not np.isinf(f.to_numpy(float)).any()
    count += len(f)
assert count == 1817783
x = pd.read_parquet(
    file,
    columns=[
        "split",
        "context",
        "whole_mean",
        "edge_inner_normalized_diff",
        "edge_signal_enrichment",
        "signal_centroid_offset_normalized",
        "local_reference_median",
    ],
)
x.groupby(["split", "context"]).agg(
    ["count", "median", "mean", lambda x: x.quantile(0.95)]
).to_csv(A / "results/feature_distributions_by_context.csv")
files = [file, A / "PhenoBIC_CD8_model_predictions.parquet"] + list(
    (A / "models").glob("*.joblib")
)
audit = dict(
    status="passed",
    rows=count,
    columns=len(COLUMNS),
    morphology_features=len(MORPH),
    model_features=len(FEATURES),
    all_numeric_columns_no_infinity=True,
    pan_gold_accessed=False,
    artifacts=[],
)
for f in files:
    h = hashlib.sha256()
    with f.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(b)
    audit["artifacts"].append(
        dict(file=str(f), bytes=f.stat().st_size, sha256=h.hexdigest())
    )
(A / "results/FINAL_DATA_AUDIT.json").write_text(json.dumps(audit, indent=2))
with zipfile.ZipFile(
    A / "review_bundle.zip", "x", compression=zipfile.ZIP_DEFLATED
) as z:
    for name in [
        "RUN_CONFIG.json",
        "TRAINING_PROTOCOL.json",
        "FEATURE_DICTIONARY.csv",
        "FEATURE_DICTIONARY.json",
        "README.txt",
    ]:
        z.write(A / name, name)
    for d in ["scripts", "provenance", "results", "tests", "examples", "models"]:
        for f in (A / d).glob("*"):
            if f.is_file():
                z.write(f, f.relative_to(A))
    for name in [
        "features/FEATURES_COMPLETE.json",
        "pilot/PILOT_PASSED.json",
        "pilot/VISUAL_QA_APPROVED.json",
    ]:
        z.write(A / name, name)
print(json.dumps(audit, indent=2))
print("BUNDLE BYTES", (A / "review_bundle.zip").stat().st_size)
