"""Check the exact publication file set without reading research datasets or credentials."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 8 * 1024**2
FORBIDDEN_SUFFIXES = {
    ".parquet",
    ".feather",
    ".arrow",
    ".h5",
    ".h5ad",
    ".hdf5",
    ".npy",
    ".npz",
    ".tif",
    ".tiff",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pem",
    ".key",
    ".ppk",
    ".p12",
    ".pfx",
    ".zip",
    ".rar",
    ".7z",
    ".gz",
    ".bundle",
}
FORBIDDEN_PARTS = {
    "datasets",
    "data",
    "raw",
    "ground_truth",
    ".ssh",
    ".aws",
    ".codex",
    ".venv",
    "venv",
    "env",
    "weights",
    "upstream",
    "cache",
    "predictions",
    "annotations",
    "keys",
    "logs",
    "__pycache__",
    "pan_gold",
}
SECRET_PATTERNS = {
    "private-key block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "GitHub token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"
    ),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential in URL": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
    "literal secret assignment": re.compile(
        r"""(?i)\b(?:password|passwd|api_key|access_token|secret_key)\s*=\s*["'][A-Za-z0-9_+/=.-]{16,}["']"""
    ),
}


def git_paths(staged: bool = False) -> list[str]:
    """Return only files Git would publish, not ignored neighboring research files."""
    args = ["git", "ls-files", "-z", "--cached"]
    if not staged:
        args += ["--others", "--exclude-standard"]
    out = subprocess.check_output(args, cwd=ROOT).decode("utf-8")
    return sorted(set(filter(None, out.split("\0"))))


def path_problem(path: str) -> str | None:
    """Reject data, model and credential paths independently of .gitignore."""
    p = Path(path)
    parts = {x.lower() for x in p.parts}
    if parts & FORBIDDEN_PARTS or any(x.startswith(("pan-m", "pan_m")) for x in parts):
        return "forbidden data/environment/credential directory"
    name = p.name.lower()
    if any(x in name for x in ("password", "credential", "secret", "token")):
        return "credential-like filename"
    if name.startswith(
        ("id_rsa", "id_ed25519", ".env", "pilot_cells_", "all_cells", "cell_keys")
    ):
        return "credential or cell-level filename"
    if p.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "raw data, model, archive or secret extension"
    return None


def secret_findings(text: str) -> list[tuple[str, int]]:
    """Report rule names and line numbers without echoing matched credential content."""
    return [
        (name, text.count("\n", 0, match.start()) + 1)
        for name, pattern in SECRET_PATTERNS.items()
        for match in pattern.finditer(text)
    ]


def local_links(text: str) -> list[str]:
    """Extract ordinary Markdown links; web URLs and anchors are not filesystem reads."""
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    return [
        unquote(link.split("#", 1)[0].strip("<>"))
        for link in links
        if "://" not in link and not link.startswith(("#", "mailto:"))
    ]


def inspect_files(paths: list[str]) -> dict:
    """Check the proposed tree, including table granularity and links to included files."""
    included = {(ROOT / p).resolve() for p in paths}
    errors = []
    total_bytes = 0
    counts = {"python": 0, "markdown": 0, "csv": 0, "figures": 0}
    for relative in paths:
        p = ROOT / relative
        problem = path_problem(relative)
        if problem:
            errors.append({"path": relative, "reason": problem})
            continue
        if not p.is_file() or p.is_symlink():
            errors.append({"path": relative, "reason": "not a regular file"})
            continue
        size = p.stat().st_size
        total_bytes += size
        if size > MAX_FILE_BYTES:
            errors.append({"path": relative, "reason": "file larger than 8 MiB"})
        if p.suffix.lower() in {".png", ".pdf"}:
            counts["figures"] += 1
            continue
        try:
            text = p.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            errors.append({"path": relative, "reason": "unexpected binary format"})
            continue
        for rule, line in secret_findings(text):
            errors.append({"path": relative, "reason": rule, "line": line})
        if p.suffix == ".py":
            ast.parse(text, filename=relative)
            counts["python"] += 1
        elif p.suffix == ".md":
            counts["markdown"] += 1
            for link in local_links(text):
                destination = (p.parent / link).resolve()
                if destination not in included:
                    errors.append(
                        {
                            "path": relative,
                            "reason": "link not in publication tree",
                            "target": link,
                        }
                    )
        elif p.suffix == ".csv":
            counts["csv"] += 1
            with p.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                n_rows = sum(1 for _ in reader)
            cell_key = bool(
                set(header)
                & {
                    "cell_id",
                    "instance_id",
                    "label",
                    "target_cell_id",
                    "neighbor_cell_id",
                }
            )
            if cell_key and n_rows > 1000:
                errors.append({"path": relative, "reason": "large cell-level CSV"})
        elif p.suffix == ".svg":
            counts["figures"] += 1
    return {
        "files": len(paths),
        "bytes": total_bytes,
        "counts": counts,
        "passed": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged", action="store_true", help="Check the staged/index file set."
    )
    args = parser.parse_args()
    result = inspect_files(git_paths(args.staged))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
