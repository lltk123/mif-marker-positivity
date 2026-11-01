from pathlib import Path
import subprocess, time, json, traceback, os
from execution_guards import atomic_json, wait_bounded
from runtime_paths import ROOT as p, RUN as r, FEATURE_PYTHON, NEURAL_PYTHON

f = FEATURE_PYTHON
n = NEURAL_PYTHON


def write(name, d):
    atomic_json(r / "logs" / name, d)


try:
    deadline = time.time() + 28800 + 7200
    while not (
        (r / "models/NEURAL_COMPLETE.json").exists()
        and (r / "results/COMPOSITION_COMPLETE.json").exists()
    ):
        for err in [
            "logs/PIPELINE_FAILED.json",
            "logs/PREPARATION_FAILED.json",
            "models/NEURAL_BUDGET_REACHED.json",
        ]:
            if (r / err).exists():
                raise RuntimeError("Dependency failure: " + err)
        if time.time() > deadline:
            raise TimeoutError(
                "Training/composition dependency wait exceeded 10h including preparation allowance"
            )
        write(
            "FINAL_PIPELINE_STAGE.json",
            {
                "stage": "waiting for fixed training and composition completion",
                "updated": time.time(),
            },
        )
        time.sleep(20)
    steps = [
        ("external_HGB", f, "stage3_freeze_hgb_predict.py"),
        ("external_neural", n, "stage3_neural_predict.py"),
        ("evaluation", f, "stage3_evaluate.py"),
        ("crop_support_QC", n, "stage4_input_quality.py"),
        ("real_review", n, "stage4_review.py"),
        ("figures", n, "stage4_figures.py"),
        ("report", n, "stage4_report.py"),
        ("delivery_validation", n, "stage5_validate_package.py"),
    ]
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    for name, python, script in steps:
        write("FINAL_PIPELINE_STAGE.json", {"stage": name, "updated": time.time()})
        cmd = [str(python), "-B", "-u", str(r / "scripts" / script)]
        with (r / "logs" / (name + ".log")).open("a") as log:
            z = subprocess.Popen(
                cmd,
                cwd=str(r),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            write(
                name + "_process.json",
                {"pid": z.pid, "command": cmd, "started": time.time()},
            )
            remaining = (
                max(
                    0.0,
                    28800.0
                    - json.loads((r / "logs/GPU_BUDGET.json").read_text())[
                        "consumed_seconds"
                    ],
                )
                if name == "external_neural"
                else 7200.0
            )
            rc = wait_bounded(z, remaining)
        if rc:
            raise RuntimeError(name + " exited " + str(rc))
    write(
        "FINAL_PIPELINE_COMPLETE.json",
        {
            "completed": time.time(),
            "human_case_review": "pending user",
            "visual_QA": "pending local inspection",
            "all_computational_stages_complete": True,
        },
    )
    write(
        "FINAL_PIPELINE_STAGE.json",
        {
            "stage": "numeric validation and packages complete; awaiting local visual QA",
            "updated": time.time(),
        },
    )
except Exception:
    write(
        "FINAL_PIPELINE_FAILED.json",
        {"traceback": traceback.format_exc(), "time": time.time()},
    )
    raise
