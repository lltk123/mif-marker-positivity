"""Model/data provenance audit without touching sealed Pan-M data."""
from common import *


def main():
    verify_inputs()
    table = pd.read_csv(E / "audit/input_manifest.csv")
    rows = []
    for e in entries():
        for kind in ["image", "mask"]:
            path = str(D / e["source"][kind])
            q = table.loc[table.path == path]
            assert len(q) == 1
            rows.append(
                dict(
                    split=e["split"],
                    context=e["context"],
                    image_id=image_id(e),
                    source_key=e["key"],
                    kind=kind,
                    path=path,
                    sha256=q.iloc[0].sha256,
                    bytes=int(q.iloc[0].bytes),
                )
            )
    d = pd.DataFrame(rows)
    d.to_csv(R / "provenance/PhenoBIC_source_checksums.csv", index=False)
    dup = d.groupby(["kind", "sha256"]).filter(lambda x: x.split.nunique() > 1)
    dup.to_csv(R / "results/exact_file_overlap_between_splits.csv", index=False)
    pd.DataFrame(
        [
            dict(
                source="Nimbus public paper and pinned repository README",
                known_training_platforms="CODEX, Vectra, MIBI",
                known_PanM_subsets="CODEX colon; Vectra colon; Vectra pancreas; MIBI breast; MIBI decidua",
                Gold_relation="expert-labelled subset of Pan-M, not asserted independent",
                local_PanM_read=False,
                PhenoBIC_pretraining_overlap_status="no author-provided complete V1 training image hash manifest; exact cross-pretraining overlap cannot be excluded",
                limitation="PhenoBIC release later than V1 is insufficient proof of independent image acquisition",
            )
        ]
    ).to_csv(R / "results/pretraining_exposure_audit.csv", index=False)
    import tifffile

    meta = []
    for e in entries():
        if e["split"] == "internal_val_WSIs" and any(
            r["split"] == e["split"] and r["context"] == e["context"] for r in meta
        ):
            continue
        with tifffile.TiffFile(D / e["source"]["image"]) as f:
            t = f.pages[0].tags
            tags = {
                k: str(t[k].value)[:1500]
                for k in [
                    "XResolution",
                    "YResolution",
                    "ResolutionUnit",
                    "ImageDescription",
                ]
                if k in t
            }
        meta.append(
            dict(
                image_id=image_id(e),
                split=e["split"],
                context=e["context"],
                tiff_tags=json.dumps(tags),
                physical_scale_interpretation="no verified um/pixel; official dataset_magnification=20 assumption, model10",
            )
        )
    pd.DataFrame(meta).to_csv(R / "results/image_scale_metadata.csv", index=False)
    dump(
        R / "provenance/source_audit.json",
        dict(
            source_files=len(d),
            exact_cross_split_duplicate_files=len(dup),
            local_PanM_read=False,
            weight_sha256=sha(R / "weights/V1.pt"),
            inference_commit=COMMIT,
        ),
    )
    print("SOURCE_AUDIT", len(d), len(dup), flush=True)


if __name__ == "__main__":
    main()
