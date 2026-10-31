"""Explain two frozen feature NaNs by formula guards; no images or labels read."""
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
)
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
r = a.run_dir
features = ["high_signal_edge_coverage", "edge_signal_enrichment"]
flags = [
    "qc_truncated",
    "qc_disconnected",
    "qc_small_whole",
    "qc_invalid_pixels",
    "qc_no_signal",
]
columns = (
    ["split", "context"]
    + features
    + flags
    + [
        "edge_valid_pixel_count",
        "inner_valid_pixel_count",
        "local_reference_pixel_count",
        "local_reference_MAD",
        "epsilon_intensity",
    ]
)
cfg = json.loads((r / "PROTOCOL.json").read_text())["geometry"]
records = []
for version, path in [
    ("raw", r.parent / "cd8_morphospatial_20261023/PhenoBIC_CD8_features.parquet"),
    ("normalized", r / "PhenoBIC_CD8_features.parquet"),
]:
    frame = pd.read_parquet(
        path,
        columns=columns + (["calibration_valid"] if version == "normalized" else []),
    )
    assert len(frame) == 1817783
    groups = [("overall", "all", "all", np.arange(len(frame)))]
    groups += [
        ("split", split, "all", positions)
        for split, positions in frame.groupby("split").indices.items()
    ]
    groups += [
        ("context", split, context, positions)
        for (split, context), positions in frame.groupby(
            ["split", "context"]
        ).indices.items()
    ]
    whole_invalid = frame[flags[:4]].any(axis=1).to_numpy()
    calibration_invalid = (
        (~frame.calibration_valid.to_numpy(bool))
        if version == "normalized"
        else np.zeros(len(frame), bool)
    )
    for feature in features:
        remaining = frame[feature].isna().to_numpy()
        codes = np.full(len(frame), -1, np.int16)
        candidates = [
            ("calibration_invalid", calibration_invalid),
            ("mask_geometry_or_pixel_support_invalid", whole_invalid),
            (
                "edge_pixels_below_minimum",
                (frame.edge_valid_pixel_count < cfg["min_region_pixels"]).to_numpy(),
            ),
        ]
        if feature == "edge_signal_enrichment":
            candidates.append(
                (
                    "inner_pixels_below_minimum",
                    (
                        frame.inner_valid_pixel_count < cfg["min_region_pixels"]
                    ).to_numpy(),
                )
            )
        candidates.append(
            (
                "local_reference_pixels_below_minimum",
                (
                    frame.local_reference_pixel_count < cfg["min_reference_pixels"]
                ).to_numpy(),
            )
        )
        if feature == "high_signal_edge_coverage":
            candidates += [
                (
                    "local_reference_MAD_exactly_zero",
                    (frame.local_reference_MAD == 0).to_numpy(),
                ),
                (
                    "local_reference_MAD_positive_not_above_epsilon",
                    (
                        (frame.local_reference_MAD > 0)
                        & (frame.local_reference_MAD <= frame.epsilon_intensity)
                    ).to_numpy(),
                ),
            ]
        else:
            candidates.append(
                (
                    "corrected_total_signal_not_above_epsilon_times_area",
                    frame.qc_no_signal.to_numpy(bool),
                )
            )
        for code, (cause, condition) in enumerate(candidates):
            chosen = remaining & condition
            codes[chosen] = code
            remaining[chosen] = False
        if remaining.any():
            raise RuntimeError(
                "Unexplained NaN: %s %s %s" % (version, feature, int(remaining.sum()))
            )
        for aggregation, split, context, positions in groups:
            n = len(positions)
            n_missing = int((codes[positions] >= 0).sum())
            for code, (cause, condition) in enumerate(candidates):
                count = int((codes[positions] == code).sum())
                records.append(
                    dict(
                        version=version,
                        aggregation=aggregation,
                        split=split,
                        context=context,
                        feature=feature,
                        cause=cause,
                        priority=code,
                        rows=n,
                        feature_missing=n_missing,
                        cause_count=count,
                        fraction_of_all_rows=count / n,
                        fraction_of_missing=count / n_missing if n_missing else np.nan,
                    )
                )
        print("EXPLAINED", version, feature, int((codes >= 0).sum()), flush=True)
output = r / "missingness_causes.csv"
assert not output.exists()
pd.DataFrame(records).to_csv(output, index=False)
meta = dict(
    scope="Computational applicability guards, not biological causality; mutually exclusive by listed priority",
    high_signal_edge_coverage="Requires whole_valid, edge>=5, reference>=20, MAD>epsilon; no_signal alone does not make coverage missing",
    edge_signal_enrichment="Requires whole_valid, edge>=5, inner>=5, reference>=20, total corrected S>epsilon*area. Remaining NaNs after these preceding guards must carry no_signal.",
    no_images_read=True,
    no_labels_read=True,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(r / "tests/missingness_cause_definitions.json").write_text(
    json.dumps(meta, indent=2) + "\n"
)
print("CAUSE_AUDIT_COMPLETE", flush=True)
