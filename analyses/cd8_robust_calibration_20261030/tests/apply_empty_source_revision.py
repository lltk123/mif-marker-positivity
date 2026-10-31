from pathlib import Path
import hashlib, json

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
a = r / "tests/implementation_fix_002"
a.mkdir()
assert not (r / "shards").exists()
p = r / "scripts/run_features.py"
assert (
    hashlib.sha256(p.read_bytes()).hexdigest()
    == "8f804dbe718969695ebdbcc118dd7fb79bef9ca32353e3ca9a65fe2fcf27afd8"
)
for file in [p, r / "PROTOCOL.json", r / "calibration/calibration_manifest.json"]:
    file.rename(a / file.name)
(r / "run_features.py.revision002").rename(p)
protocol = json.loads((a / "PROTOCOL.json").read_text())
protocol["implementation_revision_history"] = [protocol.pop("implementation_revision")]
revision = dict(
    id="002",
    reason="Pilot selects the first nonempty source in each required group; zero-instance source images remain in input audit and have zero full-extraction jobs",
    prior_protocol_sha256=hashlib.sha256(
        (a / "PROTOCOL.json").read_bytes()
    ).hexdigest(),
    bounds_and_normalization_and_features_unchanged=True,
    pre_full_extraction=True,
)
protocol["implementation_revision"] = revision
protocol["code_sha256"]["scripts/run_features.py"] = hashlib.sha256(
    p.read_bytes()
).hexdigest()
(r / "PROTOCOL.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
)
manifest = json.loads((a / "calibration_manifest.json").read_text())
manifest["protocol_sha256"] = hashlib.sha256(
    (r / "PROTOCOL.json").read_bytes()
).hexdigest()
(r / "calibration/calibration_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
)
(a / "REVISION.json").write_text(json.dumps(revision, indent=2) + "\n")
print(
    json.dumps(
        dict(
            protocol_sha256=manifest["protocol_sha256"],
            run_features_sha256=protocol["code_sha256"]["scripts/run_features.py"],
        )
    )
)
