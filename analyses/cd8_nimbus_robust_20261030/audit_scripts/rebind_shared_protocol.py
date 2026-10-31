"""Audited metadata-only migration for shared pilot fix 002; never rewrite arrays."""
from pathlib import Path
import ast
import csv
import json
import shutil
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from common import sha, dump, units, unit_dir, code_hash

SHARED = R.parent / "cd8_robust_calibration_20261030"
BEFORE = "4e6f070207d5a09fb2a38963c4bca6e23c28e2a6bc41a323482cfac21847e6fc"
AFTER = "db88a81f03992fb06c5937fa442167a3bdd2493970d2d963392f3c8f964de359"


def outside_pilot(path):
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef) or node.name != "pilot"
    ]
    return ast.dump(tree, include_attributes=False)


def main():
    evidence = SHARED / "tests/implementation_fix_002"
    proof = json.loads((evidence / "COMPATIBILITY_PROOF.json").read_text())
    assert sha(evidence / "PROTOCOL.json") == BEFORE
    assert sha(SHARED / "PROTOCOL.json") == AFTER
    assert (
        proof["before_protocol_sha256"] == BEFORE
        and proof["after_protocol_sha256"] == AFTER
    )
    assert sha(evidence / "run_features.py") == proof["before_run_features_sha256"]
    assert sha(SHARED / "scripts/run_features.py") == proof["after_run_features_sha256"]
    assert outside_pilot(evidence / "run_features.py") == outside_pilot(
        SHARED / "scripts/run_features.py"
    )
    assert (
        sha(SHARED / "scripts/calibration_core.py") == proof["normalize_module_sha256"]
    )
    assert sha(SHARED / "scripts/feature_adapter.py") == proof["feature_adapter_sha256"]
    assert sha(SHARED / "calibration/unit_bounds.csv") == proof["unit_bounds_sha256"]
    assert not list(
        (R / "predictions").rglob("*.parquet")
    ), "migration must precede any inference"
    assert not (R / "models/finetuned_complete.json").exists()
    assert (R / "tests/global_preparation_pixel_checks.csv").is_file()
    assert (R / "tests/evaluation_contract_checks.csv").is_file()
    protocol_path, config_path = R / "NIMBUS_PROTOCOL.json", R / "RUN_CONFIG.json"
    protocol, config = json.loads(protocol_path.read_text()), json.loads(
        config_path.read_text()
    )
    assert protocol["normalization"]["protocol_sha256"] == BEFORE
    assert protocol["all_keys_sha256"] == proof["all_keys_sha256"]
    assert config["code_sha256"] == code_hash(), "Nimbus code must be unchanged"
    assert config["protocol_sha256"] == sha(protocol_path)
    manifest = json.loads(
        (SHARED / "calibration/calibration_manifest.json").read_text()
    )
    assert (
        manifest["protocol_sha256"] == AFTER
        and manifest["all_keys_sha256"] == proof["all_keys_sha256"]
    )
    new_signature = {
        "protocol_sha256": AFTER,
        "bounds_sha256": sha(SHARED / "calibration/unit_bounds.csv"),
        "manifest_sha256": sha(SHARED / "calibration/calibration_manifest.json"),
    }
    old_signature = config["shared_calibration"]
    assert old_signature["bounds_sha256"] == new_signature["bounds_sha256"]
    directories = [unit_dir(image) for image in units()]
    ready = [directory / "prepared.json" for directory in directories]
    assert len(ready) == 48 and all(
        path.is_file() for path in ready
    ), "wait for all complete units"
    for path in ready:
        state = json.loads(path.read_text())
        assert state["signature"]["shared_calibration"] == old_signature
        assert state["signature"]["protocol_sha256"] == config["protocol_sha256"]
        assert not state[
            "calibration_invalid"
        ], "failed units require separate handling"
    backup = R / "provenance/revisions/shared_pilot_fix_v2"
    backup.mkdir(parents=True, exist_ok=False)
    # Hash every model-input/ID/support array. Other immutable preparation arrays
    # are recorded by inode/size/mtime; no numeric cache file is opened for writing.
    hashed_names = {
        "input.npy",
        "ids_scaled.npy",
        "needed.npy",
        "counts.npy",
        "observed_full.npy",
    }
    arrays = []
    for directory in directories:
        for path in sorted(directory.glob("*.npy")):
            stat = path.stat()
            arrays.append(
                dict(
                    path=str(path),
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    inode=stat.st_ino,
                    sha256=sha(path) if path.name in hashed_names else "",
                )
            )
        print("ARRAY_AUDIT", directory.name, flush=True)
    with (backup / "array_identity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(arrays[0]))
        writer.writeheader()
        writer.writerows(arrays)
    metadata = [protocol_path, config_path, R / "models/normalization.json", *ready]
    for path in metadata:
        destination = backup / path.relative_to(R)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for name in ["COMPATIBILITY_PROOF.json", "pilot_only.diff"]:
        shutil.copy2(evidence / name, backup / name)
    old_protocol_sha = sha(protocol_path)
    protocol["normalization"].update(new_signature)
    dump(protocol_path, protocol)
    config["protocol_sha256"] = sha(protocol_path)
    config["shared_calibration"] = new_signature
    dump(config_path, config)
    normalization_path = R / "models/normalization.json"
    normalization = json.loads(normalization_path.read_text())
    assert normalization["shared_calibration"] == old_signature
    normalization["shared_calibration"] = new_signature
    dump(normalization_path, normalization)
    for path in ready:
        state = json.loads(path.read_text())
        state["signature"].update(
            protocol_sha256=sha(protocol_path),
            normalization_sha256=sha(normalization_path),
            shared_calibration=new_signature,
        )
        dump(path, state)
    for row in arrays:
        stat = Path(row["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (
            row["bytes"],
            row["mtime_ns"],
            row["inode"],
        ), "array changed during metadata-only migration"
    dump(
        R / "provenance/CACHE_COMPATIBILITY_AUDIT.json",
        dict(
            status="complete",
            reason="Shared pilot example selection only; same exact bounds, keys and calculation modules",
            before_nimbus_protocol_sha256=old_protocol_sha,
            after_nimbus_protocol_sha256=sha(protocol_path),
            before_shared=old_signature,
            after_shared=new_signature,
            source_input_provenance=protocol["input_provenance"],
            n_complete_units=len(ready),
            numeric_arrays_rewritten=False,
            numeric_model_input_arrays_hashed=sum(
                bool(row["sha256"]) for row in arrays
            ),
            all_cache_array_stat_identities_unchanged=True,
            array_identity_sha256=sha(backup / "array_identity.csv"),
            preserved_metadata_directory=str(backup),
            proof_sha256=sha(evidence / "COMPATIBILITY_PROOF.json"),
            audit_code_sha256=sha(__file__),
            finished_unix=time.time(),
        ),
    )
    print("COMPATIBLE_CACHE_REBIND_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
