"""Freeze a small Git source snapshot and artifact manifests, without data or weights."""
from common import *
import subprocess, shutil, datetime


def main():
    assert (R / "provenance/numeric_pipeline_complete.json").exists()
    snapshot = R / "provenance/adapter_source"
    snapshot.mkdir(exist_ok=True)
    chosen = list((R / "scripts").glob("*.py"))
    chosen += [
        R / n
        for n in [
            "NIMBUS_PROTOCOL.json",
            "RUN_CONFIG.json",
            "README.md",
            "NIMBUS_INPUT_DICTIONARY.csv",
            "PAPER_DATA_SCOPE.md",
            "FIGURE_CONTRACT.md",
        ]
    ]
    chosen += [R / "provenance/requirements.lock.txt"]
    records = []
    for src in chosen:
        if not src.exists():
            continue
        rel = src.relative_to(R)
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        records.append(dict(path=str(rel), bytes=src.stat().st_size, sha256=sha(src)))
    pd.DataFrame(records).sort_values("path").to_csv(
        R / "provenance/adapter_code_hashes.csv", index=False
    )

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Codex", "-c", "user.email=codex@localhost", *args],
            cwd=snapshot,
            text=True,
        ).strip()

    if not (snapshot / ".git").exists():
        git("init", "-q")
    git("add", "--all")
    if git("status", "--porcelain"):
        git(
            "commit",
            "-q",
            "-m",
            "Freeze Nimbus PhenoBIC CD8 task adaptation and reproducible analysis",
        )
    commit = git("rev-parse", "HEAD")
    bundle = R / "provenance/adapter-source.bundle"
    if bundle.exists():
        old = (
            R
            / "provenance/code_revisions"
            / ("adapter-source_" + sha(bundle)[:12] + ".bundle")
        )
        old.parent.mkdir(exist_ok=True)
        shutil.copy2(bundle, old)
    git("bundle", "create", str(bundle), "--all")
    dump(
        R / "provenance/adapter_code_commit.json",
        dict(
            commit=commit,
            source_snapshot=str(snapshot),
            bundle=str(bundle),
            scope="new adapter code, static documentation, fixed protocol and dependency lock; excludes raw data, cached arrays, weights and predictions",
            manifest_sha256=sha(R / "provenance/adapter_code_hashes.csv"),
        ),
    )
    rows = []
    for model, weight in [
        ("Nimbus_pretrained", R / "weights/V1.pt"),
        ("Nimbus_finetuned", R / "models/Nimbus_finetuned.pt"),
    ]:
        threshold = json.loads(
            (R / "models" / (model + "_thresholds.json")).read_text()
        )
        manifest = dict(
            model=model,
            weight_path=str(weight),
            weight_sha256=sha(weight),
            official_code_commit=COMMIT,
            initial_weight_download_revision=protocol()["weight_revision"],
            original_training_code_commit=protocol()["official_training_commit"],
            adapter_code_commit=commit,
            protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
            normalization_sha256=sha(R / "models/normalization.json"),
            input_channels=[
                "CD8 normalized image",
                "official instance interior binary mask",
            ],
            input_scale="official default assumed20x to10x (0.5), physical spacing unknown",
            mask_type="expanded_nucleus",
            seed=0,
            source_dataset="PhenoBIC-Data",
            training_source=(
                "author V1 pretraining; full image-level manifest unavailable"
                if model == "Nimbus_pretrained"
                else "author V1 plus original3 PhenoBIC training WSIs; explicit pos/neg labels only"
            ),
            validation_source="3 original internal validation WSIs",
            test_source="42 original external CD8 TMA cores",
            thresholds=threshold,
            training_complete=(
                json.loads((R / "models/finetuned_complete.json").read_text())
                if model == "Nimbus_finetuned"
                else None
            ),
        )
        dump(R / "models" / (model + "_manifest.json"), manifest)
        rows.append(
            {
                k: v
                for k, v in manifest.items()
                if k not in ["thresholds", "training_complete", "input_channels"]
            }
        )
    pd.DataFrame(rows).to_csv(R / "results/model_manifest.csv", index=False)
    print("ADAPTER_COMMIT", commit, flush=True)


if __name__ == "__main__":
    main()
