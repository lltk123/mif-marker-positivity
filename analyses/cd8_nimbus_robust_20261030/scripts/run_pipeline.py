"""One resumable entry point; every stage checks source, calibration and code identity."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from common import (
    R,
    PREVIOUS_NIMBUS,
    MODEL_PRETRAINED,
    MODEL_FINETUNED,
    sha,
    code_hash,
    dump,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "bootstrap",
            "prepare",
            "pilot",
            "pretrained",
            "finetune",
            "predict",
            "evaluate",
            "report",
        ],
        default="all",
    )
    parser.add_argument("--source-run", type=Path, default=PREVIOUS_NIMBUS)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    (R / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        MPLCONFIGDIR=str(R / "cache/matplotlib"),
    )
    if args.font_path:
        env["NIMBUS_FONT_PATH"] = str(args.font_path.resolve())

    def run(name, *arguments):
        log = R / "logs" / (name + ".log")
        print("START", name, datetime.datetime.now().isoformat(), flush=True)
        with log.open("ab") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(R / "scripts" / arguments[0]),
                    *arguments[1:],
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            tail = log.read_text(errors="replace").splitlines()[-18:]
            raise RuntimeError(f"{name} failed; inspect {log}\n" + "\n".join(tail))
        print("FINISHED", name, flush=True)

    if args.stage in ["all", "bootstrap", "prepare"]:
        run("bootstrap", "bootstrap_run.py", "--source-run", str(args.source_run))
    if args.stage == "bootstrap":
        return
    if args.stage in ["all", "prepare"]:
        run("freeze", "freeze_protocol.py")
    if not (R / "RUN_CONFIG.json").is_file():
        raise FileNotFoundError(
            f"prepare stage has not completed protocol freeze: {R / 'RUN_CONFIG.json'}; run --stage prepare after shared calibration"
        )
    frozen = json.loads((R / "RUN_CONFIG.json").read_text())
    assert (
        frozen["code_sha256"] == code_hash()
    ), "code changed after freeze; use a new run"
    assert frozen["protocol_sha256"] == sha(
        R / "NIMBUS_PROTOCOL.json"
    ), "protocol changed"
    if args.stage in ["all", "prepare"]:
        run("source_audit", "audit_sources.py")
        run("preparation", "prepare_data.py")
        run("numeric_checks", "validate_offline.py")
        run("pixel_checks", "validate_prepared_pixels.py")
        run("evaluation_checks", "validate_evaluation.py")
    if args.stage in ["all", "pilot"]:
        run("controlled_checks", "validate_contract.py")
        run("pilot", "pilot.py")
        run("sparse_tile_checks", "validate_sparse_tiles.py")
        run("training_tiles", "finetune.py", "--prepare-only")
        run("training_tile_checks", "validate_training_tiles.py")
    if args.stage in ["all", "pretrained"]:
        if not (R / "tests/PILOT_PASSED.json").exists():
            raise FileNotFoundError(
                "real official-weight pilot must pass before full inference"
            )
        run("pretrained_inference", "inference.py", "--model", MODEL_PRETRAINED)
        dump(
            R / "provenance/pretrained_inference_complete.json",
            dict(
                finished=datetime.datetime.now().isoformat(),
                protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
            ),
        )
    if args.stage in ["all", "finetune"]:
        path = R / "models/finetuned_complete.json"
        if path.exists():
            state = json.loads(path.read_text())
            assert state["code_sha256"] == sha(R / "scripts/finetune.py")
            assert state["protocol_sha256"] == sha(R / "NIMBUS_PROTOCOL.json")
            assert state["weight_sha256"] == sha(
                R / "models" / (MODEL_FINETUNED + ".pt")
            )
            print("REUSE_VERIFIED_COMPLETED_FIT", flush=True)
        else:
            run("finetuning", "finetune.py")
    if args.stage in ["all", "predict"]:
        run("finetuned_inference", "inference.py", "--model", MODEL_FINETUNED)
    if args.stage in ["all", "evaluate"]:
        run("evaluation", "evaluate.py")
        run("prediction_audit", "audit_predictions.py")
        dump(
            R / "provenance/numeric_pipeline_complete.json",
            dict(
                finished=datetime.datetime.now().isoformat(),
                protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
            ),
        )
    if args.stage in ["all", "report"]:
        run("figures", "plot_results.py")
        run("development_review", "review_errors.py")
        run("report", "write_report.py")


if __name__ == "__main__":
    main()
