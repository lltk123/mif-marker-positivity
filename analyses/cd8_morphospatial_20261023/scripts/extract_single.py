"""Generic single-image entry point. Large PhenoBIC WSIs use batch_features.py."""
from pathlib import Path
import argparse, json, hashlib, sys, platform
from dataclasses import asdict
import numpy as np
import tifffile
import pyarrow as pa
import pyarrow.parquet as pq
from mif_features import *


def write_features(frame, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Existing output is preserved: " + str(path))
    schema = pa.schema(
        [
            (
                c,
                pa.string()
                if c in META_STR
                else pa.int64()
                if c in COUNTS + ["cell_id"]
                else pa.bool_()
                if c in QC
                else pa.float64(),
            )
            for c in COLUMNS
        ]
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["image", "mask", "dataset-id", "image-id", "output"]:
        p.add_argument("--" + k, required=True)
    p.add_argument("--config")
    p.add_argument("--tissue-mask")
    p.add_argument("--valid-mask")
    p.add_argument("--csv", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    targets = [out, Path(str(out) + ".run.json")] + (
        [out.with_suffix(".csv")] if a.csv else []
    )
    for f in targets:
        if f.exists():
            raise FileExistsError(f)
    cfg = (
        FeatureConfig(**json.loads(Path(a.config).read_text()))
        if a.config
        else FeatureConfig()
    )

    def load(path):
        try:
            return tifffile.memmap(path)
        except ValueError:
            return tifffile.imread(path)

    im = load(a.image)
    mask = load(a.mask)
    frame = extract_image(
        im,
        mask,
        a.dataset_id,
        a.image_id,
        cfg,
        tissue_mask=load(a.tissue_mask) > 0 if a.tissue_mask else None,
        valid_mask=load(a.valid_mask) > 0 if a.valid_mask else None,
    )
    write_features(frame, out)
    if a.csv:
        frame.to_csv(out.with_suffix(".csv"), index=False)
    provenance = dict(
        image=str(Path(a.image).resolve()),
        mask=str(Path(a.mask).resolve()),
        image_dtype=str(im.dtype),
        mask_dtype=str(mask.dtype),
        shape=list(im.shape),
        cells=len(frame),
        config=asdict(cfg),
        python=platform.python_version(),
        feature_code_sha256=hashlib.sha256(
            (Path(__file__).parent / "mif_features.py").read_bytes()
        ).hexdigest(),
    )
    Path(str(out) + ".run.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
