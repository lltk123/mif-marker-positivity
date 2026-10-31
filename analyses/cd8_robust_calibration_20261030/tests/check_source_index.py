from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
sys.path.insert(0, str(r / "scripts"))
from calibration_core import key_hash

p = r.parent.parent
old = p / "analyses/cd8_morphospatial_20261023"
entries = json.loads(
    (
        p / "analyses/cd8_scimap_gmm_20261023/features/phenobic_extraction_audit.json"
    ).read_text()
)
rows = []
keys = []
for e in entries:
    with np.load(old / "bboxes" / (e["key"] + ".npz")) as z:
        ids = z["cell_id"].copy()
    uid = (
        e["split"]
        + "/"
        + e["context"]
        + ("/" + e["patch"] if e["split"] == "external_val_TMA_cores" else "")
    )
    rows.append(
        dict(
            source_key=e["key"],
            image_id=uid,
            split=e["split"],
            context=e["context"],
            source_image_id=e["patch"],
            n_instances=len(ids),
            empty=len(ids) == 0,
        )
    )
    if len(ids):
        keys.append(
            pd.DataFrame(
                dict(
                    dataset_id="PhenoBIC-Data", image_id=uid, cell_id=ids, marker="CD8"
                )
            )
        )
f = pd.concat(keys, ignore_index=True)
protocol = json.loads((r / "PROTOCOL.json").read_text())
assert len(f) == 1817783 and key_hash(f) == protocol["all_keys_sha256"]
a = r / "tests/source_index_integrity"
a.mkdir(exist_ok=False)
pd.DataFrame(rows).to_csv(a / "source_counts.csv", index=False)
result = dict(
    status="passed",
    rows=len(f),
    source_images=len(rows),
    empty_source_images=sum(x["empty"] for x in rows),
    all_keys_sha256=key_hash(f),
    protocol_sha256=__import__("hashlib")
    .sha256((r / "PROTOCOL.json").read_bytes())
    .hexdigest(),
)
(a / "PASSED.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
