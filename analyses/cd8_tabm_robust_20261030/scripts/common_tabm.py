"""Shared contracts for the frozen PhenoBIC TabM experiment."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
sys.dont_write_bytecode = True
KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"
TRAIN = "training_WSIs"
VALID = "internal_val_WSIs"
TEST = "external_val_TMA_cores"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def key_hash(df):
    return hashlib.sha256(
        df[KEYS].sort_values(KEYS).to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def core_hash(root):
    names = [
        "common_tabm.py",
        "calibrated_inputs.py",
        "prepare_tabm.py",
        "train_tabm.py",
        "resources_tabm.py",
    ]
    return hashlib.sha256(
        "".join(sha256(Path(root) / "scripts" / n) for n in names).encode()
    ).hexdigest()


def validate_keys(df):
    import pandas as pd

    if not pd.api.types.is_integer_dtype(df.cell_id.dtype):
        raise ValueError(
            "cell_id must be an integer instance ID, never row position or fractional ID"
        )
    assert not df[KEYS].isna().any().any()
    assert not df.duplicated(KEYS).any()
    assert df.dataset_id.eq("PhenoBIC-Data").all()
    assert df.marker.eq("CD8").all() and df.cell_id.gt(0).all()
