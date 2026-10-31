"""Small source-grounded summary CSVs; no data/model fit occurs here."""
from pathlib import Path
import argparse
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
r = a.run_dir
c = pd.read_csv(r / "calibration/clipping_audit.csv")
bounds = pd.read_csv(r / "calibration/unit_bounds.csv", float_precision="round_trip")
c = c.merge(bounds[["image_id", "L"]], on="image_id", validate="one_to_one")
groups = (
    [("overall", "all", "all", c)]
    + [("split", s, "all", g) for s, g in c.groupby("split")]
    + [("context", s, t, g) for (s, t), g in c.groupby(["split", "context"])]
)
rows = []
for level, split, context, g in groups:
    n = int(g.finite_instance_pixels.sum())
    row = dict(
        aggregation=level,
        split=split,
        context=context,
        n_units=len(g),
        rows=int(g.rows.sum()),
        finite_instance_pixels=n,
        calibration_invalid_units=int((~g.calibration_valid).sum()),
        all_L_equal_zero=bool((g.L == 0).all()),
    )
    for name in ["below_L", "above_U", "at_zero", "at_one"]:
        count = int(round((g["fraction_" + name] * g.finite_instance_pixels).sum()))
        row[name + "_pixel_count"] = count
        row[name + "_fraction"] = count / n
        row[name + "_percent"] = 100 * count / n
    rows.append(row)
out = r / "calibration/clipping_summary.csv"
assert not out.exists()
pd.DataFrame(rows).to_csv(out, index=False)
f = pd.read_csv(r / "missingness_vs_raw.csv")
sums = [
    "rows",
    "old_missing",
    "new_missing",
    "newly_missing",
    "recovered",
    "net_missing_change",
    "old_zero",
    "new_zero",
]
tables = [f.assign(aggregation="context")]
split = f.groupby(["split", "feature", "family"], as_index=False)[sums].sum()
split["context"] = "all"
split["aggregation"] = "split"
tables.append(split)
all_ = f.groupby(["feature", "family"], as_index=False)[sums].sum()
all_["split"] = "all"
all_["context"] = "all"
all_["aggregation"] = "overall"
tables.append(all_)
m = pd.concat(tables, ignore_index=True)
m["old_missing_fraction"] = m.old_missing / m.rows
m["new_missing_fraction"] = m.new_missing / m.rows
m["newly_missing_fraction"] = m.newly_missing / m.rows
m["recovered_fraction"] = m.recovered / m.rows
m["new_missing_percent"] = 100 * m.new_missing_fraction
m["old_missing_percent"] = 100 * m.old_missing_fraction
m.to_csv(r / "missingness_summary.csv", index=False)
important = [
    "whole_mean",
    "Area",
    "edge_inner_normalized_diff",
    "edge_signal_enrichment",
    "high_signal_edge_coverage",
    "signal_centroid_offset_normalized",
    "neighbor_edge_mean_diff",
    "sector_signal_cv",
]
m[m.feature.isin(important)].to_csv(
    r / "feature_missingness_highlights.csv", index=False
)
print(
    pd.DataFrame(rows)
    .query('aggregation=="split"')[
        [
            "split",
            "finite_instance_pixels",
            "above_U_percent",
            "at_zero_percent",
            "at_one_percent",
            "all_L_equal_zero",
        ]
    ]
    .to_string(index=False)
)
print(
    m.query(
        'aggregation=="split" and feature in ["high_signal_edge_coverage","edge_signal_enrichment"]'
    )[
        [
            "split",
            "feature",
            "old_missing",
            "new_missing",
            "newly_missing",
            "recovered",
            "old_missing_percent",
            "new_missing_percent",
        ]
    ].to_string(
        index=False
    )
)
print("GLOBAL_NEWLY_MISSING_TOP")
print(
    m.query('aggregation=="overall"')
    .sort_values("newly_missing", ascending=False)[
        ["feature", "old_missing", "new_missing", "newly_missing", "recovered"]
    ]
    .head(15)
    .to_string(index=False)
)
