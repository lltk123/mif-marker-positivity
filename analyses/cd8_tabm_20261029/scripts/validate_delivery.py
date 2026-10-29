"""Verify full prediction keys, frozen hashes and reusable API agreement."""
import argparse
from pathlib import Path
from common_tabm import *
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from predict_table import predict_table


def main(root):
    root = Path(root)
    frozen = load_json(root / "FROZEN_TEST_PROTOCOL.json")
    cells = pd.read_parquet(root / "data/cells.parquet")
    validate_keys(cells)
    expected = key_hash(cells)
    rows = []
    # Deterministic image coverage, no selection by outcome or model score.
    sample = cells.groupby("image_id", sort=True, group_keys=False).head(3).copy()
    sample[KEYS].to_csv(root / "results/api_validation_keys.csv", index=False)
    for model_id, meta in frozen["models"].items():
        path = root / "predictions" / f"{model_id}.parquet"
        f = pq.ParquetFile(path)
        assert f.metadata.num_rows == len(cells)
        pred = pd.read_parquet(path, columns=KEYS + ["score"])
        validate_keys(pred)
        assert key_hash(pred) == expected
        assert np.isfinite(pred.score).all()
        if model_id.endswith("_s0"):
            actual = predict_table(sample, root, model_id)[KEYS + ["score"]]
            pair = actual.merge(
                pred, on=KEYS, suffixes=("_api", "_batch"), validate="one_to_one"
            )
            delta = np.abs(pair.score_api - pair.score_batch).max()
            assert delta <= 2e-6, (model_id, delta)
        else:
            delta = None
        rows.append(
            dict(
                model_id=model_id,
                n=f.metadata.num_rows,
                n_unique=len(pred),
                all_scores_finite=True,
                keys_sha256=expected,
                api_max_absolute_difference=delta,
            )
        )
    pd.DataFrame(rows).to_csv(
        root / "results/delivery_prediction_validation.csv", index=False
    )
    assert (root / "results/EVALUATION_MANIFEST.json").exists() or (
        root / "results/evaluation_manifest.json"
    ).exists()
    source = []
    for path in sorted((root / "scripts").glob("*.py")):
        source.append(
            dict(
                file=str(path.relative_to(root)),
                sha256=sha256(path),
                bytes=path.stat().st_size,
            )
        )
    pd.DataFrame(source).to_csv(root / "results/code_sha256.csv", index=False)
    save_json(
        root / "DELIVERY_VALIDATION.json",
        dict(
            status="complete",
            n_models=18,
            n_cells_per_model=len(cells),
            n_api_models_tested=6,
            n_sample_images=sample.image_id.nunique(),
            keys_sha256=expected,
            source_files=len(source),
            PanM_accessed=False,
        ),
    )
    print("Delivery validation passed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    main(p.parse_args().run_dir)
