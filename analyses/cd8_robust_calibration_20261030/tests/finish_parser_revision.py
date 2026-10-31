from pathlib import Path
import json, hashlib

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
a = r / "tests/implementation_fix_001"
p = r / "scripts/run_features.py"
p.rename(a / "run_features.py.intermediate")
(r / "run_features.py.revision001b").rename(p)
protocol = json.loads((r / "PROTOCOL.json").read_text())
(r / "PROTOCOL.json").rename(a / "PROTOCOL.intermediate.json")
protocol["code_sha256"]["scripts/run_features.py"] = hashlib.sha256(
    p.read_bytes()
).hexdigest()
protocol["implementation_revision"][
    "resume_note"
] = "Completed revised protocol is hash-verified and reused by --stage all; never silently rewritten"
(r / "PROTOCOL.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
)
m = r / "calibration/calibration_manifest.json"
manifest = json.loads(m.read_text())
m.rename(a / "calibration_manifest.intermediate.json")
manifest["protocol_sha256"] = hashlib.sha256(
    (r / "PROTOCOL.json").read_bytes()
).hexdigest()
m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(manifest["protocol_sha256"])
