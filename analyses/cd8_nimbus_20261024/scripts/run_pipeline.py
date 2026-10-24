"""Execute sequential frozen stages; never resume incompatible caches silently."""
from pathlib import Path
import os, sys, subprocess, json, datetime, hashlib

R = Path(__file__).resolve().parents[1]
py = sys.executable
env = os.environ.copy()
env.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
    MPLCONFIGDIR=str(R / "cache/matplotlib"),
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
)


def run(stage, args):
    log = R / "logs" / (stage + ".log")
    print("START", stage, datetime.datetime.now().isoformat(), flush=True)
    with log.open("ab") as f:
        r = subprocess.run(
            [py, "-u", str(R / "scripts" / args[0]), *args[1:]],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    if r.returncode:
        raise RuntimeError(f"{stage} failed ({r.returncode}): {log}")
    print("FINISHED", stage, datetime.datetime.now().isoformat(), flush=True)


if not (R / "NIMBUS_PROTOCOL.json").exists():
    run("freeze", ["freeze_protocol.py"])
if not (R / "provenance/source_audit.json").exists():
    run("source_audit", ["audit_sources.py"])
if not (R / "models/normalization.json").exists():
    run("normalization", ["prepare_data.py", "--image-id", "training_WSIs/WSI1"])
if not (R / "tests/controlled_contract_checks.csv").exists():
    run("controlled_checks", ["validate_contract.py"])
if not (R / "tests/PILOT_PASSED.json").exists():
    run("pilot", ["pilot.py"])
run("preparation", ["prepare_data.py"])
run("pretrained_inference", ["inference.py", "--model", "Nimbus_pretrained"])
(R / "provenance/pretrained_inference_complete.json").write_text(
    json.dumps({"finished": datetime.datetime.now().isoformat()})
)
fit = R / "models/finetuned_complete.json"
if fit.exists():
    state = json.loads(fit.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert state["protocol_sha256"] == digest(R / "NIMBUS_PROTOCOL.json")
    assert state["code_sha256"] == digest(R / "scripts/finetune.py")
    assert state["weight_sha256"] == digest(R / "models/Nimbus_finetuned.pt")
    print("REUSE completed compatible fine-tuning", flush=True)
else:
    run("finetuning", ["finetune.py"])
run("finetuned_inference", ["inference.py", "--model", "Nimbus_finetuned"])
run("evaluation", ["evaluate.py"])
(R / "provenance/numeric_pipeline_complete.json").write_text(
    json.dumps({"finished": datetime.datetime.now().isoformat()})
)
