"""Package small review artifacts; keep images, weights and cell-scale data on server."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import zipfile

R = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((R / "provenance/delivery_complete.json").read_text())
    assert complete["status"] in ["complete", "complete_with_unscorable_cells"]
    assert sha(R / "ANALYSIS_REPORT.md") == complete["report_sha256"]
    allowed_roots = {
        "scripts",
        "audit_scripts",
        "provenance",
        "results",
        "tests",
        "figures",
        "development_review",
        "models",
    }
    allowed_suffixes = {".py", ".md", ".json", ".csv", ".txt", ".png", ".svg"}
    records = []
    for path in sorted(R.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(R)
        if len(relative.parts) > 1 and relative.parts[0] not in allowed_roots:
            continue
        if path.suffix not in allowed_suffixes:
            continue
        if "curves" in relative.parts or path.name.startswith("pilot_cells_"):
            continue
        if path.stat().st_size > 25 * 1024**2:
            raise ValueError(
                f"unexpectedly large review artifact: {relative}; inspect before packaging"
            )
        records.append(
            dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha(path))
        )
    for entry in records:
        assert not any(
            token in Path(entry["path"]).parts
            for token in ["weights", ".venv", "upstream"]
        )
    if args.destination.exists():
        raise FileExistsError(
            f"refusing to replace an existing review package: {args.destination}"
        )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(records)
    with zipfile.ZipFile(
        args.destination, "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for entry in records:
            archive.write(R / entry["path"], entry["path"])
        archive.writestr("REVIEW_PACKAGE_CONTENTS.csv", buffer.getvalue())
    print(
        json.dumps(
            dict(
                package=str(args.destination),
                sha256=sha(args.destination),
                files=len(records),
                bytes=args.destination.stat().st_size,
                full_predictions_and_weights_remain_on_server=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
