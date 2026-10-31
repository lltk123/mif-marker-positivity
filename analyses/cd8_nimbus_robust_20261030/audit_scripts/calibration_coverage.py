"""Audit calibration coverage independently of predictions, labels never fit bounds."""
from pathlib import Path
import sys

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from common import load_cells, KEYS, PREP, sha, dump, key_hash
from normalization import calibration, SHARED
import pandas as pd


def main():
    cells = load_cells()
    bounds, signature, manifest = calibration()
    assert len(cells) == 1817783 and key_hash(cells) == manifest["all_keys_sha256"]
    labelled = cells.loc[
        cells.ground_truth_binary.isin([0, 1]), KEYS + ["split", "context"]
    ]
    joined = labelled.merge(
        bounds[["image_id", "calibration_invalid"]].reset_index(drop=True),
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    assert joined.calibration_invalid.notna().all() and len(joined) == len(labelled)
    rows = []
    for level, columns in [
        ("split", ["split"]),
        ("slide", ["split", "context"]),
        ("image", ["split", "context", "image_id"]),
    ]:
        for key, group in joined.groupby(columns, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            invalid = int(group.calibration_invalid.sum())
            rows.append(
                dict(
                    level=level,
                    **dict(zip(columns, key)),
                    n_cells=len(group),
                    n_calibration_invalid=invalid,
                    calibration_coverage=1 - invalid / len(group)
                )
            )
    destination = R / "results/calibration_coverage.csv"
    pd.DataFrame(rows).reindex(
        columns=[
            "split",
            "level",
            "context",
            "image_id",
            "n_cells",
            "n_calibration_invalid",
            "calibration_coverage",
        ]
    ).to_csv(destination, index=False)
    dump(
        R / "provenance/calibration_coverage_complete.json",
        dict(
            status="complete",
            table_sha256=sha(destination),
            n_all_cells=len(cells),
            source_sha256={
                "cell_source": sha(PREP / "artifacts/cells.parquet"),
                "shared_protocol": sha(SHARED / "PROTOCOL.json"),
                "shared_bounds": sha(SHARED / "calibration/unit_bounds.csv"),
                "shared_manifest": sha(
                    SHARED / "calibration/calibration_manifest.json"
                ),
                "audit_code": sha(__file__),
            },
            definition="calibration-invalid labelled cells / all original labelled cells in each metric group; independent of model scoring failure",
        ),
    )
    print("CALIBRATION_COVERAGE_COMPLETE", len(rows))


if __name__ == "__main__":
    main()
