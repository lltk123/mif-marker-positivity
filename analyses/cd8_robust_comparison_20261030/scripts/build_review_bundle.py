"""Build a review-only ZIP after all four 20261030 runs pass completion gates.

Standard library only. No model execution, data download, Parquet or image reading.
Usage: python build_review_bundle.py --project-root PROJECT --output-zip NEW.zip
The independent NEW.bundle_manifest.json and the ZIP both refuse overwrite.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RUNS = {
    "shared": "analyses/cd8_robust_calibration_20261030",
    "nimbus": "analyses/cd8_nimbus_robust_20261030",
    "tabm": "analyses/cd8_tabm_robust_20261030",
    "comparison": "analyses/cd8_robust_comparison_20261030",
}
DOCS = "docs/normalization_percentiles_20261030"
SERVER_ROOT = "/mnt/disk3/mIHC/mif_marker_positivity_20261022"
MAX_BYTES = 10 * 1024 * 1024
MAX_CSV_ROWS = 50000
EXTENSIONS = {".py", ".md", ".json", ".csv", ".png", ".svg"}
COMMON_DIRS = {
    "scripts",
    "tests",
    "results",
    "figures",
    "provenance",
    "audit_scripts",
    "development_review",
    "report_tables",
    "figure_source",
    "models",
}
DENIED_DIRS = {
    ".git",
    ".ssh",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "datasets",
    "raw",
    "raw_data",
    "cache",
    "cached",
    "arrays",
    "weights",
    "checkpoints",
    "predictions",
    "curves",
    "shards",
    "keys",
    "revisions",
    "code_revisions",
    "figure_layout_original",
    "metric_sandbox",
    "implementation_fix_001",
    "implementation_fix_002",
}
CASE_CSV = {
    "case_index.csv",
    "case_availability.csv",
    "case_model_comparison.csv",
    "development_cases.csv",
    "visible_neighbors.csv",
    "neighbor_context.csv",
}
DATA_JSON = {
    "DATA_READY.json",
    "risk_strata.json",
    "SHARED_REAL_PILOT_ADAPTER_AUDIT.json",
}
SECRET_NAME = re.compile(
    r"(^|[_.-])(password|passwd|secret|credentials|id_rsa|id_ed25519|token)([_.-]|$)",
    re.I,
)
SECRET_VALUE = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|"
    rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{35,}\b|"
    rb"\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9]{40,}\b"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def complete(path):
    value = load(path)
    require(value.get("status") == "complete", "Incomplete marker: " + str(path))
    return value


def reference(project, base, name):
    """Rebase recorded server paths; reject traversal and paths outside new runs."""
    text = str(name).replace("\\", "/")
    require(".." not in PurePosixPath(text).parts, "Traversal in manifest: " + text)
    if text.startswith(SERVER_ROOT + "/"):
        path = project / text[len(SERVER_ROOT) + 1 :]
    elif Path(text).is_absolute():
        path = Path(text)
    else:
        path = base / text
    path = path.resolve()
    try:
        rel = path.relative_to(project.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Manifest reference outside project: " + text) from error
    require(
        any(rel.startswith(r + "/") for r in list(RUNS.values()) + [DOCS]),
        "Manifest reference outside current review scope: " + rel,
    )
    return path


def verify(project, base, name, expected, verified):
    path = reference(project, base, name)
    require(
        path.is_file() and not path.is_symlink(),
        "Missing/nonregular bound file: " + str(path),
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", str(expected)) is not None,
        "Invalid SHA: " + str(name),
    )
    require(digest(path) == expected, "SHA mismatch: " + str(path))
    verified[path.relative_to(project).as_posix()] = expected


def verify_map(project, base, mapping, verified):
    require(isinstance(mapping, dict) and mapping, "Empty hash map: " + str(base))
    for name, expected in mapping.items():
        verify(project, base, name, expected, verified)


def verify_gates(project):
    """Check existing completion contracts without reading full cell outputs."""
    verified = {}
    paths = {key: project / rel for key, rel in RUNS.items()}
    shared, nimbus, tabm, comparison = [paths[k] for k in RUNS]
    extraction = complete(shared / "extraction_manifest.json")
    require(
        (extraction.get("rows"), extraction.get("n_units"), extraction.get("method_id"))
        == (1817783, 48, "E1_outer_P1_P99"),
        "Shared extraction population/method mismatch",
    )
    for filename, field in [
        ("PROTOCOL.json", "protocol_sha256"),
        ("calibration/unit_bounds.csv", "calibration_sha256"),
        ("FEATURE_DICTIONARY.csv", "dictionary_sha256"),
    ]:
        verify(project, shared, filename, extraction[field], verified)
    verify_map(project, shared, extraction["code_sha256"], verified)

    final = complete(nimbus / "provenance/FINAL_DELIVERY_VALIDATION.json")
    delivery = complete(nimbus / "provenance/delivery_complete.json")
    for filename, field in [
        ("ANALYSIS_REPORT.md", "report_sha256"),
        ("NIMBUS_PROTOCOL.json", "protocol_sha256"),
        ("results/delivery_validation.csv", "validation_csv_sha256"),
    ]:
        verify(project, nimbus, filename, final[field], verified)
    require(
        delivery["report_sha256"] == final["report_sha256"],
        "Nimbus report marker mismatch",
    )
    bilingual = complete(nimbus / "provenance/bilingual_figures_complete.json")
    require(
        bilingual.get("visual_review_completed") is True,
        "Nimbus bilingual visual QA pending",
    )
    verify_map(project, nimbus, bilingual["source_sha256"], verified)
    verify_map(project, nimbus, bilingual["figure_sha256"], verified)
    verify(
        project,
        nimbus,
        "results/all_metrics.csv",
        bilingual["metric_table_sha256_unchanged"],
        verified,
    )
    verify(
        project,
        nimbus,
        "ANALYSIS_REPORT.md",
        bilingual["report_sha256_unchanged"],
        verified,
    )
    for model in ["Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned", "GMM"]:
        marker = load(nimbus / "provenance" / (model + "_evaluation_complete.json"))
        require(
            marker.get("all_cells") == 1817783, "Nimbus evaluation cell count mismatch"
        )
        require(
            marker["signature"]["protocol_sha256"] == final["protocol_sha256"],
            "Nimbus evaluation protocol mismatch",
        )
        target = model + "_metrics.csv"
        hashes = [
            (name, sha)
            for name, sha in marker["output_sha256"].items()
            if PurePosixPath(name.replace("\\", "/")).name == target
        ]
        require(len(hashes) == 1, "Missing authoritative Nimbus metrics hash: " + model)
        verify(project, nimbus, hashes[0][0], hashes[0][1], verified)
    integrity = load(nimbus / "provenance/prediction_integrity_passed.json")
    require(
        (
            integrity.get("total_cells_per_model"),
            integrity.get("models"),
            integrity.get("units"),
        )
        == (1817783, 2, 48),
        "Nimbus prediction completeness mismatch",
    )
    require(
        integrity.get("key_sha256") == extraction["all_keys_sha256"],
        "Shared/Nimbus key mismatch",
    )
    require(
        integrity.get("all_finite") is True and integrity.get("total_unscorable") == 0,
        "Nimbus prediction integrity did not fully pass",
    )
    coverage = complete(nimbus / "provenance/calibration_coverage_complete.json")
    require(
        coverage.get("n_all_cells") == 1817783, "Nimbus calibration population mismatch"
    )
    verify(
        project,
        nimbus,
        "results/calibration_coverage.csv",
        coverage["table_sha256"],
        verified,
    )

    tabm_delivery = complete(tabm / "DELIVERY_VALIDATION.json")
    expected = {
        "normalization_method": "E1_outer_P1_P99",
        "n_models": 18,
        "n_cells_per_model": 1817783,
        "n_api_models_tested": 6,
        "n_sample_images": 48,
        "PanM_accessed": False,
    }
    require(
        all(tabm_delivery.get(k) == v for k, v in expected.items()),
        "TabM delivery contract mismatch",
    )
    require(
        tabm_delivery.get("keys_sha256") == extraction["all_keys_sha256"],
        "Shared/TabM key mismatch",
    )
    report = complete(tabm / "REPORT_GENERATION_MANIFEST.json")
    require(
        report.get("PanM_accessed") is False
        and report.get("no_fitting_or_model_selection") is True,
        "TabM report scope mismatch",
    )
    require(
        "results/metrics_all.csv" in report["inputs"]
        and "ANALYSIS_REPORT.md" in report["outputs"],
        "TabM report lacks main metric/report SHA",
    )
    verify_map(project, tabm, report["inputs"], verified)
    verify_map(project, tabm, report["outputs"], verified)
    verify(
        project, tabm, "scripts/write_tabm_report.py", report["script_sha256"], verified
    )
    evaluation = complete(tabm / "results/EVALUATION_MANIFEST.json")
    require(
        (
            evaluation.get("n_models"),
            evaluation.get("n_all_cells"),
            evaluation.get("n_metric_rows"),
        )
        == (19, 1817783, 1998),
        "TabM full evaluation mismatch",
    )
    verify(
        project,
        tabm,
        "FROZEN_TEST_PROTOCOL.json",
        evaluation["protocol_sha256"],
        verified,
    )
    predictions = load(tabm / "PREDICTIONS_COMPLETE.json")
    require(
        (predictions.get("models"), predictions.get("n_per_model")) == (18, 1817783),
        "TabM predictions incomplete",
    )
    verify(
        project,
        tabm,
        "FROZEN_TEST_PROTOCOL.json",
        predictions["frozen_sha256"],
        verified,
    )

    root_report = complete(comparison / "REPORT_COMPLETE.json")
    require(
        set(root_report.get("plot_languages", [])) == {"zh", "en"}
        and root_report.get("svg_text_editable") is True,
        "Comparison bilingual/vector contract incomplete",
    )
    require(
        "ANALYSIS_REPORT.md" in root_report["output_sha256"],
        "Comparison report SHA missing",
    )
    verify_map(project, comparison, root_report["input_sha256"], verified)
    verify_map(project, comparison, root_report["output_sha256"], verified)
    root_metrics = load(comparison / "results/COMPARISON_COMPLETE.json")
    verify_map(project, comparison / "results", root_metrics["output_sha256"], verified)
    # Shared narrative reports were created after extraction. Snapshot them here;
    # do not pretend their hashes were bound by the earlier frozen protocol.
    snapshots = {}
    for name in ["ANALYSIS_REPORT.md", "VALIDATION_REPORT.md"]:
        file = shared / name
        require(file.is_file(), "Missing shared report: " + name)
        snapshots[file.relative_to(project).as_posix()] = digest(file)
    return {
        "status": "complete",
        "verified_existing_sha256": verified,
        "shared_report_snapshot_sha256": snapshots,
        "shared_report_snapshot_scope": "Packaging-time snapshots, not prior completion-marker bindings",
        "full_prediction_integrity_scope": "Existing validated completion records; no Parquet/weight re-read",
    }


def allowed_location(relative):
    """Explicit directory policy supplements the file-type and row-level guards."""
    parts = relative.parts
    if not parts or any(part.lower() in DENIED_DIRS for part in parts):
        return False
    lowered = relative.as_posix().lower()
    if any(
        term in lowered for term in ["pan-m", "pan_m", "panmultiplex", "pan-multiplex"]
    ):
        return False
    dependency_text = relative.name in {"requirements.txt", "requirements.lock.txt"}
    if SECRET_NAME.search(relative.name) or (
        relative.suffix.lower() not in EXTENSIONS and not dependency_text
    ):
        return False
    if relative.name.lower() in {
        "cells.json",
        "keys.json",
        "all_cells.json",
        "all_keys.json",
        "cell_keys.json",
    }:
        return False
    if any(
        term in relative.name.lower()
        for term in ["before_", ".before", "legacy_delivery"]
    ):
        return False
    if len(parts) == 1:
        return True
    first = parts[0]
    if first == "data":
        return len(parts) == 2 and relative.name in DATA_JSON
    if first == "logs":
        return len(parts) == 2 and (
            relative.name.endswith("_resources.json")
            or relative.name.endswith("_resource_samples.csv")
            or relative.name in {"PIPELINE_STATUS.json", "job_status.json"}
        )
    if first == "upstream":
        return len(parts) == 2 and relative.suffix.lower() in {".py", ".json", ".md"}
    if first == "pilot":
        return len(parts) == 2 and relative.name in {"PASSED.json", "validation.csv"}
    if first == "calibration":
        return len(parts) == 2 and relative.suffix.lower() in {".csv", ".json"}
    if first in COMMON_DIRS:
        if first == "models":
            return relative.suffix.lower() in {".json", ".csv", ".md"}
        if first in {"scripts", "audit_scripts"}:
            return relative.suffix.lower() in {".py", ".md", ".json"}
        return True
    return False


def inspect_file(path, relative):
    if path.is_symlink():
        return "symlink"
    if not allowed_location(relative):
        return "outside_explicit_allowlist"
    if path.stat().st_size > MAX_BYTES:
        return "larger_than_10_MiB"
    data = path.read_bytes()
    if path.suffix.lower() != ".png" and SECRET_VALUE.search(data):
        raise ValueError("Possible secret in eligible artifact: " + str(path))
    if path.suffix.lower() == ".csv":
        rows = csv.reader(io.StringIO(data.decode("utf-8-sig")))
        header = next(rows, [])
        case_table = relative.parts[0] == "development_review" and path.name in CASE_CSV
        # Full per-cell/key outputs stay server-side even if renamed or small.
        if not case_table and any(
            x.strip().lower() in {"cell_id", "cellid", "instance_id"} for x in header
        ):
            return "cell_or_instance_key_rows"
        cap = 3000 if case_table else MAX_CSV_ROWS
        for count, _ in enumerate(rows, 1):
            if count > cap:
                return "too_many_csv_rows"
    return None


def collect(project):
    files, excluded = [], []
    for root_rel in list(RUNS.values()) + [DOCS]:
        root = project / root_rel
        require(
            root.is_dir() and not root.is_symlink(),
            "Missing or linked review directory: " + str(root),
        )
        for directory, dirs, names in os.walk(root, followlinks=False):
            # Do not traverse sealed data, caches or environments at all.
            dirs[:] = [
                d
                for d in sorted(dirs)
                if d.lower() not in DENIED_DIRS
                and not any(t in d.lower() for t in ["pan-m", "pan_m", "panmultiplex"])
                and not (Path(directory) / d).is_symlink()
            ]
            for name in sorted(names):
                path = Path(directory) / name
                local_rel = path.relative_to(root)
                reason = inspect_file(path, local_rel)
                rel = path.relative_to(project).as_posix()
                if reason:
                    excluded.append({"path": rel, "reason": reason})
                else:
                    files.append(
                        {
                            "path": rel,
                            "size_bytes": path.stat().st_size,
                            "sha256": digest(path),
                        }
                    )
    require(files, "No review artifacts")
    return sorted(files, key=lambda item: item["path"]), excluded


def build(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    require(output.suffix.lower() == ".zip", "Output must end in .zip")
    sidecar = output.with_suffix(".bundle_manifest.json")
    checksum = output.with_suffix(".zip.sha256")
    require(
        not any(p.exists() for p in [output, sidecar, checksum]),
        "Refusing to overwrite bundle or sidecar",
    )
    gates = verify_gates(project)
    files, excluded = collect(project)
    included = {row["path"] for row in files}
    required = set(gates["verified_existing_sha256"]) | set(
        gates["shared_report_snapshot_sha256"]
    )
    require(
        required <= included,
        "A bound review artifact violates packaging limits: "
        + ", ".join(sorted(required - included)),
    )
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": digest(__file__),
        "project_relative_layout": True,
        "completion_checks": gates,
        "files": files,
        "n_files": len(files),
        "total_uncompressed_bytes": sum(row["size_bytes"] for row in files),
        "policy": {
            "max_file_bytes": MAX_BYTES,
            "max_summary_csv_rows": MAX_CSV_ROWS,
            "full_cells_keys_weights_envs_caches_raw_data_excluded": True,
            "PanM_read": False,
            "old_run_directories_included": False,
        },
        "excluded_scanned_files": excluded,
        "full_server_artifacts": {
            "features": SERVER_ROOT
            + "/"
            + RUNS["shared"]
            + "/PhenoBIC_CD8_features.parquet",
            "TabM_models_and_predictions": SERVER_ROOT + "/" + RUNS["tabm"],
            "Nimbus_models_and_predictions": SERVER_ROOT + "/" + RUNS["nimbus"],
        },
    }
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build under a temporary name, then publish exclusively. A failed verification
    # leaves no apparently complete final ZIP and never edits source artifacts.
    with tempfile.TemporaryDirectory(
        prefix="review_bundle_", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "review.zip"
        with zipfile.ZipFile(
            staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.writestr("bundle_manifest.json", payload)
            for row in files:
                path = project / row["path"]
                require(
                    not path.is_symlink(), "Artifact became a symlink: " + row["path"]
                )
                data = path.read_bytes()
                require(
                    len(data) == row["size_bytes"]
                    and hashlib.sha256(data).hexdigest() == row["sha256"],
                    "Artifact changed during packaging: " + row["path"],
                )
                archive.writestr(row["path"], data)
        require(
            verify_gates(project) == gates,
            "Completion contracts changed during packaging",
        )
        with zipfile.ZipFile(staged) as archive:
            require(archive.testzip() is None, "ZIP integrity check failed")
        zip_sha = digest(staged)
        # Exclusive create avoids race-related overwrites; interrupted copies are
        # not advertised as complete because sidecars are written only afterwards.
        with output.open("xb") as dest, staged.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(block)
        with sidecar.open("xb") as stream:
            stream.write(payload)
        with checksum.open("x", encoding="ascii") as stream:
            stream.write(zip_sha + "  " + output.name + "\n")
    return {
        "status": "complete",
        "zip": str(output),
        "zip_sha256": zip_sha,
        "manifest": str(sidecar),
        "n_files": len(files),
        "bytes": output.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.project_root, args.output_zip), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
