"""Check the pinned SCIMAP implementation and its score semantics."""

from pathlib import Path
import json, sys, os

p = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_scimap_gmm_20261023"
)
for x in (p / "features").glob("*.json"):
    d = json.loads(x.read_text())
    if "key" in d:
        print(
            {
                k: d[k]
                for k in [
                    "split",
                    "context",
                    "n_cells",
                    "negative_pixels",
                    "nonfinite_pixels",
                    "segmented_pixel_min",
                    "segmented_pixel_max",
                ]
            }
        )
import numpy as np, pandas as pd, anndata as ad

sys.path.insert(0, str(p / "scripts"))
import scimap_rescale_upstream as sm

rng = np.random.RandomState(0)
a = np.r_[rng.normal(1, 0.05, 100), rng.normal(3, 0.05, 100), rng.normal(8, 0.05, 100)]
b = a * 10
adata = ad.AnnData(
    np.r_[a, b][:, None],
    obs=pd.DataFrame(
        {"imageid": ["A"] * 300 + ["B"] * 300},
        index=["cell_" + str(i) for i in range(600)],
    ),
    var=pd.DataFrame(index=["CD8"]),
    dtype=np.float64,
)
x = sm.rescale(adata.copy(), method="by_image", verbose=False)
individual = {}
for unit in ["A", "B"]:
    d = sm.rescale(adata[adata.obs.imageid == unit].copy(), method="all", verbose=False)
    individual[unit] = float(d.uns["gates"].loc["CD8", unit])
check = {
    "multi_image_by_image_gates": x.uns["gates"].loc["CD8"].to_dict(),
    "separate_call_gates": individual,
    "workaround": "One upstream rescale call per independent WSI/core/FOV; upstream source unchanged.",
}
(p / "provenance/upstream_multi_image_gate_audit.json").write_text(
    json.dumps(check, indent=2)
)
print("UPSTREAM CHECK", check)
