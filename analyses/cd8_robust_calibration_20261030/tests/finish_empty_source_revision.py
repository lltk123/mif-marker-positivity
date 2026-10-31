from pathlib import Path
import ast, difflib, hashlib, json

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
a = r / "tests/implementation_fix_002"
p = r / "scripts/run_features.py"
new = r / "run_features.py.revision002final"
before = (a / "run_features.py").read_text()
after = new.read_text()


def without_pilot(s):
    tree = ast.parse(s)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "pilot"
        )
    ]
    return ast.dump(tree, include_attributes=False)


assert without_pilot(before) == without_pilot(after)
(a / "pilot_only.diff").write_text(
    "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="before/run_features.py",
            tofile="after/run_features.py",
        )
    )
)
p.rename(a / "run_features.py.intermediate")
new.rename(p)
protocol = json.loads((r / "PROTOCOL.json").read_text())
(r / "PROTOCOL.json").rename(a / "PROTOCOL.intermediate.json")
protocol["code_sha256"]["scripts/run_features.py"] = hashlib.sha256(
    p.read_bytes()
).hexdigest()
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
previous = json.loads((a / "PROTOCOL.json").read_text())
proof = dict(
    revision="002",
    before_protocol_sha256=hashlib.sha256(
        (a / "PROTOCOL.json").read_bytes()
    ).hexdigest(),
    after_protocol_sha256=manifest["protocol_sha256"],
    before_run_features_sha256=hashlib.sha256(
        (a / "run_features.py").read_bytes()
    ).hexdigest(),
    after_run_features_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    all_top_level_code_except_pilot_AST_identical=True,
    unit_bounds_sha256=manifest["unit_bounds_sha256"],
    all_keys_sha256=manifest["all_keys_sha256"],
    normalize_module_sha256=hashlib.sha256(
        (r / "scripts/calibration_core.py").read_bytes()
    ).hexdigest(),
    feature_adapter_sha256=hashlib.sha256(
        (r / "scripts/feature_adapter.py").read_bytes()
    ).hexdigest(),
    bounds_unchanged=previous["unit_bounds_sha256"] == protocol["unit_bounds_sha256"],
    keys_unchanged=previous["all_keys_sha256"] == protocol["all_keys_sha256"],
    normalize_and_feature_adapter_unchanged=all(
        previous["code_sha256"][name] == protocol["code_sha256"][name]
        for name in ["scripts/calibration_core.py", "scripts/feature_adapter.py"]
    ),
)
(a / "COMPATIBILITY_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof, indent=2))
