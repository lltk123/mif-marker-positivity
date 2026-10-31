from pathlib import Path
import os, signal, hashlib, json, datetime

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
).resolve()
cmd = Path("/proc/470937/cmdline")
if cmd.exists():
    text = cmd.read_bytes().decode().replace("\x00", " ")
    if str(r) not in text or "--stage pilot" not in text:
        raise RuntimeError("PID command mismatch")
    os.kill(470937, signal.SIGTERM)
a = r / "tests/implementation_fix_001"
a.mkdir(parents=True, exist_ok=False)
p = r / "scripts/run_features.py"
assert (
    hashlib.sha256(p.read_bytes()).hexdigest()
    == "7d7538c93b0234f6b7d857ff0f0d6f357890cbe7389db5491d81e440be28881f"
)
assert not (r / "shards").exists()
for file in [p, r / "PROTOCOL.json", r / "calibration/calibration_manifest.json"]:
    assert r in file.resolve().parents
    file.rename(a / file.name)
(r / "run_features.py.revision001").rename(p)
protocol = json.loads((a / "PROTOCOL.json").read_text())
prior = hashlib.sha256((a / "PROTOCOL.json").read_bytes()).hexdigest()
protocol["code_sha256"]["scripts/run_features.py"] = hashlib.sha256(
    p.read_bytes()
).hexdigest()
protocol["implementation_revision"] = dict(
    id="001",
    reason="Round-trip CSV parsing makes shared L/U identical to Nimbus; no bounds, keys or formulas changed",
    prior_protocol_sha256=prior,
    max_observed_parser_difference=3.552713678800501e-15,
    pre_full_extraction=True,
)
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
if (r / "pilot").exists():
    (r / "pilot").rename(a / "pilot")
(a / "REVISION.json").write_text(
    json.dumps(protocol["implementation_revision"], indent=2) + "\n"
)
print(json.dumps(manifest, indent=2))
