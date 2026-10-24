"""Research delivery extra audit for the frozen PhenoBIC CD8 experiment."""

import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
import pandas as pd, pyarrow.parquet as pq
from common import *
from test_inference_delivery import main

main()
q = pd.read_parquet(A / "stability/perturbed_feature_QC.parquet")
bad = q[~q.intensity_input_valid]
bad.to_csv(A / "stability/input_invalid_cases.csv", index=False)
print(
    bad[
        [
            "image_id",
            "cell_id",
            "perturbation",
            "observed_pixels",
            "finite_pixels",
            "qc_reasons",
        ]
    ].to_string(index=False)
)
paths = [
    A / "PhenoBIC_E1_E4_features.parquet",
    A / "calibration/cell_percentiles.parquet",
    A / "calibration/bound_contributors.parquet",
    A / "calibration/original_gmm_states.json",
    A / "stability/paired_cell_results.parquet",
    A / "stability/perturbed_feature_QC.parquet",
]
for folder, pattern in [
    ("predictions", "*.parquet"),
    ("models", "*.joblib"),
    ("keys", "*.parquet"),
    ("curves", "*.npz"),
    ("stability/gmm_states", "*.json"),
]:
    paths += sorted((A / folder).glob(pattern))
rows = []
for p in paths:
    rows.append(
        dict(
            file=str(p.relative_to(A)),
            server_absolute_path=str(p),
            bytes=p.stat().st_size,
            rows=pq.read_metadata(p).num_rows if p.suffix == ".parquet" else None,
            sha256=sha(p),
        )
    )
pd.DataFrame(rows).to_csv(A / "FULL_ARTIFACT_PATHS.csv", index=False)
print("FULL ARTIFACTS", len(rows))
