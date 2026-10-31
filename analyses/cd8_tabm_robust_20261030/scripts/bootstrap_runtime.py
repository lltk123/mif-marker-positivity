"""Copy only pinned official runtime files from the old run into this new run.

Existing adapter code/results are never replaced. No fitted artifacts or data
are copied. The caller chooses the existing compatible Python interpreter.
"""
import argparse
import hashlib
import json
from pathlib import Path
from common_tabm import COMMIT, sha256, save_json

FORBIDDEN = {
    ".parquet",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".tif",
    ".tiff",
    ".zip",
    ".rar",
}


def bootstrap(source, target):
    source, target = Path(source).resolve(strict=True), Path(target).resolve(
        strict=True
    )
    if (
        source.parent != target.parent
        or source.parent.name != "analyses"
        or source == target
    ):
        raise ValueError(
            "Source and new run must be distinct direct siblings under analyses."
        )
    official = source / "upstream/SOURCE_MANIFEST.json"
    manifest = json.loads(official.read_text(encoding="utf-8-sig"))
    if manifest.get("commit") != COMMIT:
        raise ValueError("Official TabM commit differs from the pinned run.")
    files = [official]
    for record in manifest["files"]:
        name = Path(record["file"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe upstream manifest path.")
        p = source / "upstream" / name
        if sha256(p) != record["sha256"]:
            raise ValueError("Official source hash mismatch: " + str(p))
        files.append(p)
    for directory in ["dependencies", "upstream"]:
        d = source / directory
        for p in sorted(d.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p in files:
                continue
            if directory == "upstream" and p.suffix != ".whl":
                continue
            files.append(p)
    records = []
    for p in files:
        if (
            p.is_symlink()
            or source not in p.resolve().parents
            or p.suffix.lower() in FORBIDDEN
        ):
            raise ValueError("Unsafe runtime source: " + str(p))
        relative = p.relative_to(source)
        checksum = sha256(p)
        if (
            p.suffix == ".whl"
            and checksum
            != "87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d"
        ):
            raise ValueError(
                "Embedding dependency wheel differs from the pinned artifact."
            )
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(destination) != checksum:
                raise ValueError(
                    "Will not overwrite incompatible runtime: " + str(destination)
                )
        else:
            with destination.open("xb") as stream:
                stream.write(p.read_bytes())
        records.append(
            dict(file=str(relative), sha256=checksum, bytes=p.stat().st_size)
        )
    for name in [
        "data",
        "models",
        "predictions",
        "results",
        "curves",
        "logs",
        "figures",
        "development_review",
    ]:
        (target / name).mkdir(exist_ok=True)
    record = dict(
        status="runtime_ready_not_trained",
        source_run=str(source),
        official_commit=COMMIT,
        files=records,
        copied_fitted_artifacts=False,
        PanM_accessed=False,
    )
    saved = target / "RUNTIME_BOOTSTRAP.json"
    if saved.exists():
        if json.loads(saved.read_text(encoding="utf-8")) != record:
            raise ValueError("Existing bootstrap record differs.")
    else:
        save_json(saved, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = bootstrap(args.source_run, args.run_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))
