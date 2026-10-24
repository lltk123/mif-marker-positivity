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
    markers = {
        "extract": "features/FEATURES_COMPLETE.json",
        "train": "results/TRAINING_COMPLETE.json",
        "evaluate": "results/EVALUATION_COMPLETE.json",
        "stability": "stability/STABILITY_COMPLETE.json",
    }
    if (A / markers[stage]).exists():
        if stage == "extract":
            marker = json.loads((A / markers[stage]).read_text())
            assert (
                sha(A / "PhenoBIC_E1_E4_features.parquet") == marker["parquet_sha256"]
            )
        elif stage == "train":
            training = json.loads((A / "TRAINING_RUN.json").read_text())
            assert training["code_sha256"] == sha(A / "scripts/train_new.py")
            marker = json.loads((A / markers[stage]).read_text())
            for n, h in marker["model_hashes"].items():
                assert sha(A / "models" / (n + ".joblib")) == h
        elif stage == "evaluate":
            marker = json.loads((A / markers[stage]).read_text())
            assert marker["code_sha256"] == sha(A / "scripts/evaluate.py") and marker[
                "threshold_freeze_sha256"
            ] == sha(A / "THRESHOLDS_FROZEN.json")
        else:
            signature = json.loads((A / "stability/STABILITY_RUN.json").read_text())
            for path, h in signature["inputs"].items():
                assert sha(path) == h
        print("Verified completed stage; no work repeated:", stage)
        return
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
