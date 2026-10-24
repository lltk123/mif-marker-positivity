"""Validate saved GMM results against their original labels and explicit cell keys."""

from pathlib import Path
import sys, json, csv, hashlib, zipfile
import numpy as np, pandas as pd

P = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")
A = P / "analyses/cd8_scimap_gmm_20261023"
sys.path.insert(0, str(A / "scripts"))
from extract_cd8_v2 import read

root = P / "datasets/PhenoBIC-Data"
out = A / "results/phenobic"
if not (out / "STATUS.json").exists():
    raise RuntimeError("Analysis not complete")
audit = json.loads((A / "features/phenobic_extraction_audit.json").read_text())
entries = {}
for e in audit:
    unit = (
        e["context"]
        if e["split"] in ("training_WSIs", "internal_val_WSIs")
        else e["patch"]
    )
    uid = e["dataset"] + "/" + e["split"] + "/" + e["context"] + "/" + unit
    entries.setdefault(uid, []).append(e)
index = pd.read_csv(out / "cell_table_index.csv")
checks = []
for _, u in index.iterrows():
    df = pd.read_csv(A / u["file"])
    target = []
    for v in [0, 1]:
        cand = df[(df.ground_truth_binary == v) & df.eligible_binary_evaluation]
        if len(cand):
            target.append(cand.iloc[len(cand) // 2])
    for row in target:
        cid = int(row.cell_id)
        pixels = {}
        for e in entries[u.unit_id]:
            z = np.load(A / "features" / (e["key"] + ".npz"))
            pos = np.flatnonzero(z["cell_id"] == cid)
            if not len(pos):
                continue
            i = pos[0]
            cx = z["x_sum"][i] / z["pixel_count"][i] - e["offset_x"]
            cy = z["y_sum"][i] / z["pixel_count"][i] - e["offset_y"]
            mask = read(root / e["source"]["mask"])
            im = read(root / e["source"]["image"])
            h, w = mask.shape
            radius = 64
            while True:
                top = max(0, int(cy) - radius)
                bottom = min(h, int(cy) + radius + 1)
                left = max(0, int(cx) - radius)
                right = min(w, int(cx) + radius + 1)
                b = mask[top:bottom, left:right]
                selected = b == cid
                atedge = (
                    (top > 0 and selected[0, :].any())
                    or (bottom < h and selected[-1, :].any())
                    or (left > 0 and selected[:, 0].any())
                    or (right < w and selected[:, -1].any())
                )
                if not atedge:
                    break
                radius *= 2
            yy, xx = np.where(selected)
            vals = im[top:bottom, left:right][selected]
            assert len(vals) == int(
                z["pixel_count"][i]
            ), "Sample ROI did not include full cell"
            for y, x, v in zip(yy, xx, vals):
                k = (int(y + top + e["offset_y"]), int(x + left + e["offset_x"]))
                if k in pixels:
                    assert pixels[k] == float(v)
                pixels[k] = float(v)
        direct_mean = float(np.mean(list(pixels.values()), dtype=np.float64))
        count_ok = len(pixels) == int(row.pixel_count)
        mean_ok = bool(np.isclose(direct_mean, row.mean_CD8, atol=1e-10, rtol=1e-10))
        assert count_ok and mean_ok, (
            u.unit_id,
            cid,
            direct_mean,
            row.mean_CD8,
            len(pixels),
            row.pixel_count,
        )
        checks.append(
            dict(
                unit_id=u.unit_id,
                cell_id=cid,
                ground_truth=int(row.ground_truth_binary),
                unique_pixels=len(pixels),
                direct_mean=direct_mean,
                table_mean=row.mean_CD8,
                pixel_count_exact=count_ok,
                mean_matches=mean_ok,
            )
        )
    print("VERIFIED", u.unit_id, flush=True)
pd.DataFrame(checks).to_csv(out / "independent_pixel_checks.csv", index=False)
coverage = pd.read_csv(out / "annotation_coverage.csv")
metrics = pd.read_csv(out / "metrics_summary.csv")
assert (metrics.TP + metrics.TN + metrics.FP + metrics.FN == metrics.n_evaluated).all()
assert (
    coverage.matched_annotations + coverage.annotations_without_mask
    == coverage.n_annotations
).all()
assert (
    coverage.n_binary_evaluated.sum()
    == json.loads((out / "STATUS.json").read_text())["n_evaluated"]
)
report = {
    "independent_direct_pixel_checks": len(checks),
    "all_sample_counts_exact": all(c["pixel_count_exact"] for c in checks),
    "all_sample_means_agree_rtol_atol_1e_10": all(c["mean_matches"] for c in checks),
    "metric_count_identities_passed": True,
    "annotation_accounting_passed": True,
    "analysis_scripts_sha256": {
        x.name: hashlib.sha256(x.read_bytes()).hexdigest()
        for x in (A / "scripts").glob("*.py")
    },
}
(out / "VERIFICATION.json").write_text(json.dumps(report, indent=2))
print("VERIFICATION COMPLETE", json.dumps(report), flush=True)
# A compact delivery bundle includes summary tables, provenance and curve data, but no raw images or holdout data.
with zipfile.ZipFile(
    A / "phenobic_summary_bundle.zip", "x", compression=zipfile.ZIP_DEFLATED
) as z:
    for folder in ["results/phenobic", "provenance", "scripts"]:
        for f in (A / folder).glob("*"):
            if f.is_file() and not f.name.endswith("_cells.csv.gz"):
                z.write(f, f.relative_to(A))
print("BUNDLE", A / "phenobic_summary_bundle.zip", flush=True)
