"""Verify actual source-file hashes before resuming a frozen stage."""
import argparse, subprocess, sys, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from common import *


def main(stage):
    manifest = pd.read_csv(A / "audit/input_manifest.csv")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for r, actual in zip(manifest.itertuples(), pool.map(sha, manifest.path)):
            if actual != r.sha256:
                raise RuntimeError("Source data hash changed: " + r.path)
    cfg = json.loads((A / "RUN_CONFIG.json").read_text())
    for name, h in cfg["extraction_code"].items():
        if sha(A / "scripts" / name) != h:
            raise RuntimeError("Extraction code hash changed: " + name)
    command = {
        "extract": ["batch_extract.py", "full", "--workers", "8"],
        "train": ["train_new.py"],
        "evaluate": ["evaluate.py"],
        "stability": ["stability.py", "--workers", "6"],
    }[stage]
    subprocess.run(
        [sys.executable, str(A / "scripts" / command[0])] + command[1:], check=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["extract", "train", "evaluate", "stability"])
    main(p.parse_args().stage)
