"""Run the complete frozen PhenoBIC TabM workflow, including audits and reports.

Use bootstrap_runtime.py to copy pinned runtime files into the new analysis directory. --dry-run
checks entry points and prints the plan without training, data reads or writes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from calibrated_inputs import source_paths

STAGES = [
    "environment",
    "prepare",
    "pilot",
    "freeze-risk",
    "numeric-audit",
    "train",
    "predict",
    "embedding-audit",
    "evaluate",
    "figures",
    "review",
    "validate",
    "report",
]
RULES = dict(
    development_only_visual_review=True,
    normalization_method="E1_outer_P1_P99",
    weak_positive="original raw training positive whole_mean <= original training q25; audit_raw mapping; model inputs calibrated",
    neighbor_proxy="neighbor_edge_area_fraction>0 and neighbor_edge_mean_diff>training contact q75",
    causal_claim=False,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_risk(root):
    """Idempotently freeze the original audit definitions; never backdate them."""
    ready = root / "data/DATA_READY.json"
    risk_path = root / "data/risk_strata.json"
    if not ready.is_file() or not risk_path.is_file():
        raise FileNotFoundError(
            "Prepare data and training-only risk strata before freezing"
        )
    risk = load_json(risk_path)
    if (
        risk.get("fit_split") != "training_WSIs"
        or risk.get("contact_comparison") != "strict >"
        or risk.get("weak_comparison") != "<="
    ):
        raise ValueError(
            "Risk definitions differ from the prescribed training-only rules"
        )
    fields = [
        "weak_positive_whole_mean_q25",
        "contact_bright_diff_q75",
        "local_reference_q75",
        "sector_cv_q75",
    ]
    for field in fields:
        value = risk[field]
        if value is not None and (
            not isinstance(value, (int, float))
            or not __import__("math").isfinite(value)
        ):
            raise ValueError("Non-finite risk cutoff: " + field)
    expected = dict(RULES, risk_strata_sha256=sha256(risk_path))
    path = root / "AUDIT_PROTOCOL.json"
    final = root / "FROZEN_TEST_PROTOCOL.json"
    if path.exists():
        previous = load_json(path)
        if any(previous.get(k) != v for k, v in expected.items()):
            raise RuntimeError(
                "Existing audit protocol is incompatible; it will not be overwritten"
            )
        when = previous.get("frozen_unix_time")
        if not isinstance(when, (int, float)) or not __import__("math").isfinite(when):
            raise ValueError("Existing audit timestamp is invalid")
        if final.exists() and when >= load_json(final)["frozen_unix_time"]:
            raise RuntimeError(
                "Audit timestamp must precede the final model/test protocol"
            )
        print("Compatible pre-training audit protocol retained", flush=True)
        return
    if final.exists():
        raise RuntimeError(
            "Cannot create a retrospective pre-training audit after model/test freeze"
        )
    write_new(path, dict(frozen_unix_time=time.time(), **expected))
    print("Training-only audit rules frozen", flush=True)


def record_environment(root):
    """Observe this runtime, keeping prior snapshots explicitly historical."""
    sys.path.insert(0, str(root / "dependencies"))
    sys.path.insert(0, str(root / "upstream"))
    packages = [
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "joblib",
        "psutil",
        "matplotlib",
        "scikit-image",
        "tifffile",
        "imagecodecs",
        "opencv-python",
        "rtdl-num-embeddings",
    ]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    import torch
    import psutil
    import tabm

    now = datetime.now(timezone.utc)
    official = load_json(root / "upstream/SOURCE_MANIFEST.json")
    dependencies = dict(
        captured_utc=now.isoformat(),
        runtime_executable=sys.executable,
        python=sys.version,
        packages=versions,
        tabm_version=tabm.__version__,
        tabm_commit=official["commit"],
        source="observed current interpreter; no environment installation or modification",
        runtime_reused_read_only=True,
    )
    hardware = dict(
        captured_utc=now.isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        hostname=platform.node(),
        cpu_count=os.cpu_count(),
        physical_memory_bytes=psutil.virtual_memory().total,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_available=torch.cuda.is_available(),
        torch_cuda_version=torch.version.cuda,
        visible_gpus=[
            dict(
                index=i,
                name=torch.cuda.get_device_name(i),
                total_memory_bytes=torch.cuda.get_device_properties(i).total_memory,
            )
            for i in range(torch.cuda.device_count())
        ],
    )
    if tabm.__version__ != "0.0.3" or versions["rtdl-num-embeddings"] != "0.0.12":
        raise RuntimeError(
            "The official TabM/embedding implementation versions are not the pinned ones"
        )
    for name, value in [
        ("DEPENDENCIES.json", dependencies),
        ("HARDWARE.json", hardware),
    ]:
        path = root / name
        if not path.exists():
            write_new(path, value)
        else:
            # Existing records keep their original capture time, never masquerading
            # as a fresh observation. A new actual check is stored alongside them.
            old = load_json(path)
            if name == "DEPENDENCIES.json":
                changed = {
                    k: (v, versions.get(k))
                    for k, v in old.get("packages", {}).items()
                    if v != "not installed" and versions.get(k) != v
                }
                if changed:
                    raise RuntimeError(
                        "Runtime dependencies changed since the original run: "
                        + str(changed)
                    )
            stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
            write_new(
                root / "logs/environment_checks" / f"{stamp}_{name}",
                dict(
                    value,
                    previous_record_path=str(path),
                    previous_record_sha256=sha256(path),
                ),
            )
    print("Actual dependency/hardware state recorded", flush=True)


def compatible_numeric_audit(root):
    path = root / "results/numeric_resolution_audit_manifest.json"
    outputs = [
        root / "results/numeric_resolution_train.csv",
        root / "results/NUMERIC_RESOLUTION_AUDIT.md",
    ]
    if not path.exists():
        if any(p.exists() for p in outputs):
            raise RuntimeError(
                "Partial numeric audit exists; preserve it and resolve before rerunning"
            )
        return False
    old = load_json(path)
    if old["script_sha256"] != sha256(root / "scripts/audit_numeric_resolution.py"):
        raise RuntimeError(
            "Numeric audit code changed; existing audit will not be overwritten"
        )
    for item in old["inputs"].values():
        source = Path(item["path"]).resolve()
        if root not in source.parents or sha256(source) != item["sha256"]:
            raise RuntimeError("Numeric audit input changed: " + str(source))
    for name, checksum in old["outputs"].items():
        output = root / "results" / name
        if sha256(output) != checksum:
            raise RuntimeError("Numeric audit output changed: " + str(output))
    print("Compatible training-only numeric audit retained", flush=True)
    return True


def command_plan(root, font_path=None):
    commands = {
        "prepare": ["prepare_tabm.py", "--run-dir", str(root)],
        "pilot": ["train_tabm.py", "--run-dir", str(root), "pilot"],
        "numeric-audit": ["audit_numeric_resolution.py", "--run-dir", str(root)],
        "train": ["train_tabm.py", "--run-dir", str(root), "train"],
        "predict": ["train_tabm.py", "--run-dir", str(root), "predict"],
        "embedding-audit": ["audit_embedding_bins.py", "--run-dir", str(root)],
        "evaluate": ["evaluate_tabm.py", "--run-dir", str(root)],
        "figures": ["plot_tabm.py", "--run-dir", str(root)],
        "review": ["review_tabm_errors.py", "--run-dir", str(root)],
        "validate": ["validate_delivery.py", "--run-dir", str(root)],
        "report": ["write_tabm_report.py", "--run-dir", str(root)],
    }
    if font_path is not None:
        for name in ["figures", "review"]:
            commands[name] += ["--font-path", str(font_path)]
    return commands


def run(root, stage="all", font_path=None, dry_run=False):
    root = Path(root).resolve()
    if not root.is_dir() or root.parent.name != "analyses":
        raise ValueError(
            "Run directory must already exist directly under PROJECT/analyses; use bootstrap_runtime.py"
        )
    if "pan-m" in str(root).lower():
        raise ValueError("Pan-M remains sealed")
    if font_path is not None:
        font_path = Path(font_path).resolve(strict=True)
        if not font_path.is_file():
            raise ValueError("--font-path must be an existing font file")
    stages = STAGES if stage == "all" else [stage]
    commands = command_plan(root, font_path)
    for name in stages:
        if name in commands and not (root / "scripts" / commands[name][0]).is_file():
            raise FileNotFoundError(
                "Missing pipeline entry point: " + commands[name][0]
            )
    if not dry_run and stage in {"all", "prepare", "pilot", "train", "predict"}:
        missing = [str(p) for p in source_paths(root).values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Full E1_outer_P1_P99 inputs are required; missing:\n"
                + "\n".join(missing)
            )
    if not dry_run and not (root / "upstream/SOURCE_MANIFEST.json").is_file():
        raise FileNotFoundError("Missing pinned upstream provenance")
    if dry_run:
        print(
            json.dumps(
                [
                    dict(
                        stage=name,
                        command=[
                            sys.executable,
                            str(root / "scripts" / commands[name][0]),
                            *commands[name][1:],
                        ]
                        if name in commands
                        else "internal: " + name,
                    )
                    for name in stages
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if stage not in {"all", "environment"} and any(
        not (root / name).is_file() for name in ["DEPENDENCIES.json", "HARDWARE.json"]
    ):
        record_environment(root)
    for name in stages:
        print("STAGE", name, flush=True)
        if name == "environment":
            record_environment(root)
            continue
        if name == "freeze-risk":
            freeze_risk(root)
            continue
        if name == "train":
            # An explicit --stage train is also protected by the pre-training audit.
            freeze_risk(root)
        if name == "numeric-audit" and compatible_numeric_audit(root):
            continue
        if name in [
            "predict",
            "embedding-audit",
            "evaluate",
            "figures",
            "review",
            "validate",
            "report",
        ]:
            if not (root / "FROZEN_TEST_PROTOCOL.json").is_file():
                raise FileNotFoundError(
                    "Final model/test protocol must be frozen before " + name
                )
            freeze_risk(root)
        cmd = commands[name]
        command = [sys.executable, str(root / "scripts" / cmd[0]), *cmd[1:]]
        if name == "validate" and stage != "all":
            # The reusable TabM API check uses CUDA for a small fixed sample.
            # Standalone validation therefore reacquires the shared lock after
            # a driver releases it for CPU postprocessing. --stage all is
            # already protected by the documented outer flock command.
            import fcntl

            with (root.parent / ".cd8_robust_gpu.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.run_dir, args.stage, args.font_path, args.dry_run)
