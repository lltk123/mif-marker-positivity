"""Download exactly the audited official TabM source; never download datasets or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"
FILES = [
    "tabm.py",
    "README.md",
    "LICENSE",
    "example.ipynb",
    "pyproject.toml",
    "paper/README.md",
]


def main(root):
    root = Path(root)
    records = []
    for name in FILES:
        url = f"https://raw.githubusercontent.com/yandex-research/tabm/{COMMIT}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            contents = response.read()
        destination = root / "upstream" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != contents:
            raise FileExistsError(
                "Different existing official source: " + str(destination)
            )
        if not destination.exists():
            destination.write_bytes(contents)
        records.append(
            dict(
                file=name,
                url=url,
                bytes=len(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
            )
        )
    manifest = root / "upstream/SOURCE_MANIFEST.json"
    payload = dict(commit=COMMIT, files=records)
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8-sig"))
        prior_hashes = {item["file"]: item["sha256"] for item in prior["files"]}
        expected_hashes = {item["file"]: item["sha256"] for item in records}
        if prior.get("commit") != COMMIT or prior_hashes != expected_hashes:
            raise ValueError("Existing official source manifest differs")
    else:
        with manifest.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
    print("Pinned official source is ready:", root / "upstream")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    main(p.parse_args().run_dir)
