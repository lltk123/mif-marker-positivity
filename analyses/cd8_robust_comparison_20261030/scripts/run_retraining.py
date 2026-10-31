"""Reproduce both new model runs sequentially, retaining all previous runs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--python", type=Path, help="Pinned model-training interpreter")
    parser.add_argument(
        "--feature-python", type=Path, help="Pinned original morphometry interpreter"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "shared", "nimbus", "tabm", "compare", "report"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--font-path", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    analyses = project / "analyses"
    py = args.python or analyses / "cd8_nimbus_20261024/.venv/bin/python"
    feature_py = (
        args.feature_python or analyses / "cd8_morphospatial_20261023/.venv/bin/python"
    )
    shared = analyses / "cd8_robust_calibration_20261030"
    tabm = analyses / "cd8_tabm_robust_20261030"
    nimbus = analyses / "cd8_nimbus_robust_20261030"
    comparison = analyses / "cd8_robust_comparison_20261030"
    steps = [
        (
            "shared",
            [
                str(feature_py),
                str(shared / "scripts/run_features.py"),
                "--run-dir",
                str(shared),
                "--stage",
                "all",
                "--workers",
                str(args.workers),
            ],
            shared,
        ),
        (
            "nimbus",
            [
                str(py),
                str(nimbus / "scripts/run_pipeline.py"),
                "--stage",
                "all",
                "--font-path",
                args.font_path,
            ],
            nimbus,
        ),
        (
            "tabm",
            [
                str(py),
                str(tabm / "scripts/bootstrap_runtime.py"),
                "--source-run",
                str(analyses / "cd8_tabm_20261029"),
                "--run-dir",
                str(tabm),
            ],
            tabm,
        ),
        (
            "tabm",
            [
                str(py),
                str(tabm / "scripts/run_pipeline.py"),
                "--run-dir",
                str(tabm),
                "--stage",
                "all",
                "--font-path",
                args.font_path,
            ],
            tabm,
        ),
        (
            "compare",
            [
                str(py),
                str(comparison / "scripts/compare_results.py"),
                "--project-root",
                str(project),
                "--output-dir",
                str(comparison / "results"),
                "--require-delivery",
            ],
            comparison,
        ),
        (
            "report",
            [
                str(py),
                str(comparison / "scripts/make_report.py"),
                "--project-root",
                str(project),
                "--run-dir",
                str(comparison),
                "--font-path",
                args.font_path,
            ],
            comparison,
        ),
    ]
    selected = [step for step in steps if args.stage in ("all", step[0])]
    if args.dry_run:
        print(
            json.dumps(
                [{"stage": s, "command": c, "cwd": str(d)} for s, c, d in selected],
                indent=2,
            )
        )
        return
    for stage, command, cwd in selected:
        if not Path(command[0]).is_file():
            raise SystemExit("Missing pinned stage interpreter: " + command[0])
        if not Path(command[1]).is_file():
            raise SystemExit("Missing stage entry point: " + command[1])
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log_dir = comparison / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        "run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".jsonl"
    )
    with log_path.open("x", encoding="utf-8") as log:
        for stage, command, cwd in selected:
            started = time.time()
            print("START", stage, flush=True)
            log.write(
                json.dumps(
                    {
                        "stage": stage,
                        "event": "start",
                        "unix_time": started,
                        "command": command,
                    }
                )
                + "\n"
            )
            log.flush()
            result = subprocess.run(command, cwd=cwd, env=env, check=False)
            log.write(
                json.dumps(
                    {
                        "stage": stage,
                        "event": "finish",
                        "seconds": time.time() - started,
                        "returncode": result.returncode,
                    }
                )
                + "\n"
            )
            log.flush()
            if result.returncode:
                raise SystemExit(result.returncode)
    print(
        "Selected stages finished. Formal result status requires each completion manifest.",
        flush=True,
    )


if __name__ == "__main__":
    main()
