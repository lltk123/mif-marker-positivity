from pathlib import Path
import subprocess, time, json, traceback, os
from execution_guards import atomic_json, wait_bounded, neural_completion_state
from runtime_paths import ROOT as p, RUN as r, FEATURE_PYTHON, NEURAL_PYTHON


def write(name, d):
    atomic_json(r / "logs" / name, d)


try:
    deadline = time.time() + 7200
    while not (r / "data/FEATURES_COMPLETE.json").exists():
        if (r / "logs/PREPARATION_FAILED.json").exists():
            raise RuntimeError("preparation failed; inspect its recorded traceback")
        if time.time() > deadline:
            raise TimeoutError("preparation wait exceeded 2h")
        write(
            "PIPELINE_STAGE.json",
            {"stage": "waiting for feature completion", "updated": time.time()},
        )
        time.sleep(20)
    steps = [
        ("HGB", FEATURE_PYTHON, ["scripts/stage1_hgb.py"]),
        ("neural_smoke", NEURAL_PYTHON, ["scripts/stage2_neural.py", "--smoke-only"]),
        ("neural_training", NEURAL_PYTHON, ["scripts/stage2_neural.py"]),
    ]
    for name, python, args in steps:
        write("PIPELINE_STAGE.json", {"stage": name, "updated": time.time()})
        with (r / "logs" / (name + ".log")).open("a") as log:
            cmd = [str(python), "-u"] + [
                str(r / x) if x.startswith("scripts/") else x for x in args
            ]
            z = subprocess.Popen(
                cmd,
                cwd=str(r),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(
                name + "_process.json",
                {"pid": z.pid, "command": cmd, "started": time.time()},
            )
            limit = 28800.0
            if name == "neural_training" and (r / "logs/GPU_BUDGET.json").exists():
                limit = max(
                    0.0,
                    28800.0
                    - json.loads((r / "logs/GPU_BUDGET.json").read_text())[
                        "consumed_seconds"
                    ],
                )
            rc = wait_bounded(z, limit)
        if name == "neural_training":
            complete = (
                json.loads((r / "models/NEURAL_COMPLETE.json").read_text())
                if (r / "models/NEURAL_COMPLETE.json").exists()
                else None
            )
            state = neural_completion_state(
                rc, complete, (r / "models/NEURAL_BUDGET_REACHED.json").exists()
            )
            if state != "completed":
                write("PIPELINE_STAGE.json", {"stage": state, "updated": time.time()})
                raise RuntimeError(state)
        if rc:
            raise RuntimeError(name + " exited " + str(rc))
    write(
        "TRAINING_PIPELINE_COMPLETE.json",
        {"completed": time.time(), "external_metrics_inspected": False},
    )
    write(
        "PIPELINE_STAGE.json",
        {
            "stage": "training complete; ready for frozen selection and evaluation",
            "updated": time.time(),
        },
    )
except Exception:
    write(
        "PIPELINE_FAILED.json",
        {"traceback": traceback.format_exc(), "time": time.time()},
    )
    raise
