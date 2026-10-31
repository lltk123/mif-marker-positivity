"""Create isolated pinned Nimbus source/assets without changing the old environment."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
OLD = R.parent / "cd8_nimbus_20261024"
COMMIT = "241281ef85ba34d9857cc03886cc45aedd0f0b34"
WEIGHT_SHA = "0651a3852fbafc03be415a1ec80ef947cc1913a081cf0bc649301ade887a82a9"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def copy_checked(source, destination, expected):
    if sha(source) != expected:
        raise ValueError(f"source checksum mismatch: {source}")
    if destination.exists():
        if sha(destination) != expected:
            raise ValueError(
                f"existing output differs; refusing overwrite: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=OLD)
    parser.add_argument("--source-package", type=Path)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Copy only source; no weights or GPU operation",
    )
    args = parser.parse_args()
    source = args.source_run.resolve()
    candidates = (
        ([args.source_package] if args.source_package else [])
        + list(source.glob("upstream/Nimbus-Inference-*/src/nimbus_inference"))
        + list(source.glob(".venv/lib/python*/site-packages/nimbus_inference"))
    )
    package = next(
        (p for p in candidates if p is not None and (p / "nimbus.py").is_file()), None
    )
    if package is None:
        raise FileNotFoundError("pinned Nimbus source not found; pass --source-package")
    pins = json.loads((R / "UPSTREAM_SOURCE_PINS.json").read_text())
    if pins["commit"] != COMMIT:
        raise ValueError("unexpected official source revision")
    for name, digest in pins["files"].items():
        copy_checked(package / name, R / "upstream/src/nimbus_inference" / name, digest)
    for name in [
        "cache",
        "models",
        "weights",
        "predictions",
        "results",
        "logs",
        "tests",
        "provenance",
        "figures",
        "development_review",
    ]:
        (R / name).mkdir(exist_ok=True)
    if not args.source_only:
        copy_checked(source / "weights/V1.pt", R / "weights/V1.pt", WEIGHT_SHA)
        copy_checked(
            R / "weights/V1.pt",
            R / "upstream/src/nimbus_inference/assets/V1.pt",
            WEIGHT_SHA,
        )
    record = {
        "source_run": str(source),
        "source_package": str(package),
        "official_commit": COMMIT,
        "isolated_import_path": str(R / "upstream/src"),
        "source_only": args.source_only,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "old_source_or_assets_modified": False,
    }
    (R / "provenance/bootstrap.json").write_text(json.dumps(record, indent=2))
    if not args.source_only:
        # The existing isolated runtime need not include pip; inspect installed metadata.
        from importlib.metadata import distributions

        packages = sorted(
            {
                f"{d.metadata['Name']}=={d.version}"
                for d in distributions()
                if d.metadata.get("Name")
            }
        )
        (R / "provenance/requirements.lock.txt").write_text("\n".join(packages) + "\n")
        device = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        (R / "provenance/hardware.csv").write_text(device.stdout)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
