"""Research export examples for the frozen PhenoBIC CD8 experiment."""

import json
import numpy as np, pandas as pd
from common import *
from io_adapter import BYKEY, boxes, roi
from compact_features import normalize
from legacy_features import extract_roi, FeatureConfig


def main():
    dest = A / "examples"
    dest.mkdir(exist_ok=True)
    f = pd.read_parquet(A / "PhenoBIC_E1_E4_features.parquet")
    s = pd.read_parquet(A / "keys/stability_sample.parquet")
    d = f.merge(s[KEYS], on=KEYS, validate="one_to_one")
    bounds = pd.read_csv(
        A / "calibration/calibration_parameters.csv", float_precision="round_trip"
    ).set_index("image_id")
    rows = []
    for j, uid in enumerate(
        json.loads((A / "audit/representative_units.json").read_text())
    ):
        g = d[(d.image_id == uid) & d.intensity_input_valid & np.isfinite(d.angular_R1)]
        r = g.sort_values("norm_whole_mean", ascending=False).iloc[0]
        e = BYKEY[r.source_key]
        ids, bb = boxes(e)
        box = bb[np.flatnonzero(ids == r.cell_id)[0]]
        im, lab, v, origin, trunc = roi(e, box, halo=8)
        L = bounds.loc[uid, "L"]
        U = bounds.loc[uid, "U"]
        n = normalize(im, L, U, v)
        _, regions = extract_roi(
            im,
            lab,
            int(r.cell_id),
            config=FeatureConfig(mask_type="expanded_nucleus"),
            valid_mask=v,
            force_truncated=trunc,
            return_regions=True,
        )
        fn = "case_%02d.npz" % j
        np.savez_compressed(
            dest / fn,
            raw=im,
            instances=lab,
            valid=v,
            normalized=n,
            whole=regions["whole"],
            edge=regions["edge"],
            inner=regions["inner"],
            reference=regions["reference"],
        )
        row = {c: r[c] for c in KEYS + I8 + S9}
        row.update(
            file=fn,
            L=L,
            U=U,
            selection="highest normalized whole mean among preselected sampled cells with valid R1; no outcome selection",
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(dest / "example_index.csv", index=False)
    print("EXAMPLES", len(rows))


if __name__ == "__main__":
    main()
