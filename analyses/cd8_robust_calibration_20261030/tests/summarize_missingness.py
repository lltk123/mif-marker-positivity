"""Paired missingness/QC audit, aligned by explicit instance keys; no labels."""
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
)
from pathlib import Path
import argparse, json, sys, hashlib
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
r = a.run_dir
sys.path.insert(0, str(r / "scripts"))
from feature_adapter import load_legacy

old = r.parent / "cd8_morphospatial_20261023"
m = load_legacy(old)
keys = ["dataset_id", "image_id", "cell_id"]
manifest = json.loads((r / "extraction_manifest.json").read_text())
assert manifest["status"] == "complete"
columns = keys + ["split", "context"] + m.FEATURES + m.QC
raw = pd.read_parquet(old / "PhenoBIC_CD8_features.parquet", columns=columns).set_index(
    keys
)
new = pd.read_parquet(r / "PhenoBIC_CD8_features.parquet", columns=columns).set_index(
    keys
)
assert raw.index.is_unique and new.index.is_unique and len(raw) == len(new) == 1817783
assert (
    len(raw.index.difference(new.index)) == 0
    and len(new.index.difference(raw.index)) == 0
)
raw = raw.reindex(new.index)
assert raw[["split", "context"]].equals(new[["split", "context"]])
records = []
quality = []
for (split, context), positions in new.groupby(["split", "context"]).indices.items():
    for name in m.FEATURES:
        before = raw[name].to_numpy()[positions]
        after = new[name].to_numpy()[positions]
        b = np.isnan(before)
        c = np.isnan(after)
        n = len(positions)
        records.append(
            dict(
                split=split,
                context=context,
                feature=name,
                family="geometry19"
                if name in m.MORPH
                else "intensity8"
                if name in m.WHOLE
                else "spatial63",
                rows=n,
                old_missing=int(b.sum()),
                new_missing=int(c.sum()),
                newly_missing=int((~b & c).sum()),
                recovered=int((b & ~c).sum()),
                net_missing_change=int(c.sum() - b.sum()),
                old_missing_fraction=float(b.mean()),
                new_missing_fraction=float(c.mean()),
                old_zero=int((before == 0).sum()),
                new_zero=int((after == 0).sum()),
            )
        )
    for name in m.QC:
        before = raw[name].to_numpy(bool)[positions]
        after = new[name].to_numpy(bool)[positions]
        quality.append(
            dict(
                split=split,
                context=context,
                qc=name,
                rows=len(positions),
                old_flagged=int(before.sum()),
                new_flagged=int(after.sum()),
                newly_flagged=int((~before & after).sum()),
                recovered=int((before & ~after).sum()),
            )
        )
    print("AUDITED", split, context, len(positions), flush=True)
output = r / "missingness_vs_raw.csv"
assert not output.exists()
pd.DataFrame(records).to_csv(output, index=False)
pd.DataFrame(quality).to_csv(r / "qc_vs_raw.csv", index=False)
meta = dict(
    status="complete",
    rows=len(new),
    source_feature_sha256=manifest["source_feature_sha256"],
    normalized_feature_sha256=manifest["feature_sha256"],
    protocol_sha256=manifest["protocol_sha256"],
    keys=keys,
    labels_used=False,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(r / "tests/missingness_audit_manifest.json").write_text(
    json.dumps(meta, indent=2) + "\n"
)
print("MISSINGNESS_AUDIT_COMPLETE", flush=True)
