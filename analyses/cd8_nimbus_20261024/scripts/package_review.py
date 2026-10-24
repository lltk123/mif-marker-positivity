"""Compact review bundle; full cell tables and curves remain at explicitly listed server paths."""
from common import *
import zipfile, datetime


def main():
    assert (R / "provenance/prediction_integrity_passed.json").exists()
    assert json.loads((R / "provenance/final_numeric_checks_passed.json").read_text())[
        "all_checks_passed"
    ]
    assert (R / "development_review/VISUAL_REVIEW.md").exists()
    assert json.loads((R / "figures/FIGURE_QA.json").read_text()).get(
        "visual_inspection_complete"
    )
    all_paths = []
    for root in ["predictions", "results/curves", "models", "weights"]:
        for f in sorted((R / root).rglob("*")):
            if f.is_file():
                all_paths.append(
                    dict(path=str(f), bytes=f.stat().st_size, artifact_group=root)
                )
    for f in sorted((R / "provenance").glob("*")):
        if f.is_file() and ("keys" in f.name) and f.suffix in [".parquet", ".gz"]:
            all_paths.append(
                dict(
                    path=str(f),
                    bytes=f.stat().st_size,
                    artifact_group="provenance/full_keys",
                )
            )
    for f in sorted((R / "development_review").glob("*_all_development_errors.csv.gz")):
        all_paths.append(
            dict(
                path=str(f),
                bytes=f.stat().st_size,
                artifact_group="development_review/full_errors",
            )
        )
    pd.DataFrame(all_paths).to_csv(R / "FULL_ARTIFACT_PATHS.csv", index=False)
    selected = []
    for f in R.iterdir():
        if f.is_file() and f.suffix in [".md", ".json", ".csv"]:
            selected.append(f)
    for root in [
        "scripts",
        "results",
        "figures",
        "tests",
        "provenance",
        "development_review",
        "models",
    ]:
        for f in (R / root).rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(R)
            parts = rel.parts
            if any(
                x in parts
                for x in [
                    "curves",
                    "adapter_source",
                    "metric_sandbox",
                    "__pycache__",
                    "code_revisions",
                ]
            ):
                continue
            if (
                root == "provenance"
                and ("keys" in f.name or f.suffix == ".parquet")
                and f.name != "key_counts_hashes.csv"
            ):
                continue
            if root == "development_review" and f.name.endswith(
                "_all_development_errors.csv.gz"
            ):
                continue
            if root == "models" and f.suffix != ".json":
                continue
            if f.suffix not in [
                ".py",
                ".md",
                ".csv",
                ".json",
                ".txt",
                ".lock",
                ".bundle",
                ".png",
                ".svg",
                ".pdf",
                ".npz",
            ]:
                continue
            if root == "tests" and f.suffix == ".npz":
                continue
            selected.append(f)
    manifest = [
        dict(path=str(f.relative_to(R)), bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(set(selected))
    ]
    pd.DataFrame(manifest).to_csv(R / "REVIEW_PACKAGE_MANIFEST.csv", index=False)
    selected.append(R / "REVIEW_PACKAGE_MANIFEST.csv")
    target = R / (
        "Nimbus_review_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip"
    )
    with zipfile.ZipFile(
        target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for f in sorted(set(selected)):
            z.write(f, f.relative_to(R))
    result = dict(
        path=str(target),
        bytes=target.stat().st_size,
        sha256=sha(target),
        n_files=len(set(selected)),
        full_cell_and_curve_tables="server, see FULL_ARTIFACT_PATHS.csv",
    )
    dump(R / "provenance/review_package.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
