"""Nimbus/PhenoBIC run constants, provenance and exact-key contracts."""
from pathlib import Path
import hashlib, json, os, sys, time
from functools import lru_cache
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
R = Path(__file__).resolve().parents[1]
P = R.parents[1]
PREVIOUS_NIMBUS = P / "analyses/cd8_nimbus_20261024"
# All official imports resolve inside this new run; Trainer assets never reach the old package.
sys.path.insert(0, str(R / "upstream/src"))
MODEL_PRETRAINED = "Nimbus_outerP1P99_pretrained"
MODEL_FINETUNED = "Nimbus_outerP1P99_finetuned"
D = P / "datasets/PhenoBIC-Data"
G = P / "analyses/cd8_scimap_gmm_20261023"
OLD = P / "analyses/cd8_morphospatial_20261023"
E = P / "analyses/cd8_e1_e4_stability_20261023"
PREP = P / "analyses/cd8_paper_baseline_20261024"
KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
COMMIT = "241281ef85ba34d9857cc03886cc45aedd0f0b34"
WEIGHT_SHA = "0651a3852fbafc03be415a1ec80ef947cc1913a081cf0bc649301ade887a82a9"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False, default=str)
    )
    tmp.replace(p)


def protocol():
    return json.loads((R / "NIMBUS_PROTOCOL.json").read_text())


def code_hash():
    h = hashlib.sha256()
    for p in sorted((R / "scripts").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def entries():
    return json.loads((G / "features/phenobic_extraction_audit.json").read_text())


def image_id(e):
    return (
        e["split"]
        + "/"
        + e["context"]
        + ("/" + e["patch"] if e["split"] == "external_val_TMA_cores" else "")
    )


def units():
    groups = {}
    for e in entries():
        groups.setdefault(image_id(e), []).append(e)
    return groups


def unit_code(i):
    return hashlib.sha256(i.encode()).hexdigest()[:16]


def unit_dir(i):
    return R / "cache/units" / unit_code(i)


def load_cells():
    return pd.read_parquet(PREP / "artifacts/cells.parquet")


def exact_keys(df):
    assert not df[KEYS].isna().any().any()
    assert not df.duplicated(KEYS).any()
    assert (df.cell_id > 0).all()
    assert (df.marker == "CD8").all()


def key_hash(df):
    return hashlib.sha256(
        df[KEYS].sort_values(KEYS).to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def verify_inputs():
    """No scans of Pan-M. Metadata check against previously SHA-256 verified PhenoBIC."""
    import csv

    src = E / "audit/input_manifest.csv"
    df = pd.read_csv(src)
    for row in df.itertuples(index=False):
        path = Path(row.path)
        if "Pan-M" in str(path):
            raise ValueError("sealed source cannot be accessed")
        stat = path.stat()
        assert (
            stat.st_size == row.bytes and stat.st_mtime_ns == row.mtime_ns
        ), "source changed: " + str(path)
    return {
        "prior_source_manifest_sha256": sha(src),
        "prior_cells_sha256": sha(PREP / "artifacts/cells.parquet"),
    }


def read_tiff(path):
    import tifffile

    try:
        a = tifffile.memmap(path, mode="r").squeeze()
    except ValueError:
        a = tifffile.imread(path).squeeze()
    assert a.ndim == 2
    return a


def allocate(path, shape, dtype):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def cached(i, name, mode="r"):
    return np.load(unit_dir(i) / (name + ".npy"), mmap_mode=mode)


def seed_all():
    import random, torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rss_gib():
    import psutil

    return psutil.Process().memory_info().rss / 1024**3
