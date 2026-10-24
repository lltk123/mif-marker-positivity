"""Audit authorized PhenoBIC inputs and prepare a model-independent cell manifest.
Does not train, generate predictions, choose a model or read Pan-M.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys

sys.dont_write_bytecode = True
import argparse, hashlib, importlib.metadata as metadata, json, platform, resource
import shutil, subprocess, time, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import tifffile
from baseline_contract import KEYS, exact_join, key_hash, validate_keys


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def js(p, x):
    Path(p).write_text(
        json.dumps(x, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def command(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return dict(
            returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip()
        )
    except Exception as e:
        return dict(error=str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    P = args.project.resolve()
    A = args.output.resolve()
    if A.exists():
        raise FileExistsError(
            "Use a new output directory; previous audits are preserved"
        )
    A.mkdir(parents=True)
    (A / "samples").mkdir()
    E = P / "analyses/cd8_e1_e4_stability_20261023"
    G = P / "analyses/cd8_scimap_gmm_20261023"
    O = P / "analyses/cd8_morphospatial_20261023"
    D = P / "datasets/PhenoBIC-Data"
    sourcepaths = [
        D / "READINESS.json",
        D / "raw/README.md",
        D / "raw/LICENSE",
        E / "audit/cells.parquet",
        E / "audit/input_manifest.csv",
        O / "PhenoBIC_CD8_features.parquet",
        O / "RUN_CONFIG.json",
        G / "features/phenobic_extraction_audit.json",
    ]
    if any(not p.is_file() for p in sourcepaths):
        raise FileNotFoundError([str(p) for p in sourcepaths if not p.is_file()])
    print("CHECK EXISTING FILES", flush=True)
    existing = pd.read_csv(E / "audit/input_manifest.csv")
    inventory = []
    for r in existing.itertuples(index=False):
        p = Path(r.path)
        if not (str(p).startswith(str(D) + "/") or str(p).startswith(str(O) + "/")):
            raise ValueError("Unexpected original manifest path: " + str(p))
        st = p.stat() if p.exists() else None
        inventory.append(
            dict(
                path=str(p),
                exists=st is not None,
                expected_bytes=int(r.bytes),
                current_bytes=st.st_size if st else None,
                size_matches=bool(st and st.st_size == int(r.bytes)),
                mtime_matches=bool(st and st.st_mtime_ns == int(r.mtime_ns)),
                historical_sha256=r.sha256,
                sha256_recomputed_this_run=False,
            )
        )
    checks = pd.DataFrame(inventory)
    checks.to_csv(A / "input_file_checks.csv", index=False)
    if not checks[["exists", "size_matches", "mtime_matches"]].all().all():
        raise RuntimeError(
            "Existing input changed or missing; inspect input_file_checks.csv"
        )
    ziprows = []
    sizes = {
        "training_WSIs.zip": 5053559948,
        "internal_val_WSIs.zip": 5064769108,
        "external_val_TMA_cores.zip": 294851987,
    }
    for name, expected in sizes.items():
        p = D / "raw" / name
        with p.open("rb") as f:
            signature = f.read(4)
        with zipfile.ZipFile(p) as z:
            members = len(z.infolist())
        ziprows.append(
            dict(
                file=str(p),
                bytes=p.stat().st_size,
                expected_bytes=expected,
                size_matches=p.stat().st_size == expected,
                zip_signature_ok=signature == b"PK\x03\x04",
                central_directory_readable=True,
                members=members,
                full_crc_retested=False,
            )
        )
    pd.DataFrame(ziprows).to_csv(A / "archives.csv", index=False)
    if not all(r["size_matches"] and r["zip_signature_ok"] for r in ziprows):
        raise ValueError("Archive size/signature mismatch")
    sys.path.insert(0, str(E / "scripts"))
    from io_adapter import AUDIT, boxes, image_id, roi, CACHE

    if not all(
        e["dataset"] == "PhenoBIC-Data" and e["source"]["image_channel"] == "CD8"
        for e in AUDIT
    ):
        raise ValueError("Only PhenoBIC CD8 is authorized")
    units = defaultdict(list)
    sources = []
    registry = []
    overlaps = []
    bypath = checks.set_index("path")
    for e in AUDIT:
        uid = image_id(e)
        units[uid].append(e)
        ip = D / e["source"]["image"]
        mp = D / e["source"]["mask"]
        with tifffile.TiffFile(ip) as f:
            ishape = f.series[0].shape
            itype = str(f.series[0].dtype)
        with tifffile.TiffFile(mp) as f:
            mshape = f.series[0].shape
            mtype = str(f.series[0].dtype)
        if tuple(ishape) != tuple(e["shape"]) or tuple(ishape) != tuple(mshape):
            raise ValueError("Image/mask shape mismatch: " + uid)
        if not np.issubdtype(np.dtype(mtype), np.integer):
            raise ValueError("Noninteger mask")
        ids, bbox = boxes(e)
        registry.append(
            pd.DataFrame(
                dict(
                    dataset_id="PhenoBIC-Data", image_id=uid, cell_id=ids, marker="CD8"
                )
            )
        )
        sources.append(
            dict(
                dataset_id="PhenoBIC-Data",
                image_id=uid,
                split=e["split"],
                context=e["context"],
                source_image_id=e["patch"],
                image=str(ip),
                mask=str(mp),
                labels=str(D / e["source"]["labels"]),
                height=ishape[0],
                width=ishape[1],
                image_dtype=itype,
                mask_dtype=mtype,
                source_instances=len(ids),
                image_historical_sha256=bypath.loc[str(ip), "historical_sha256"],
            )
        )
        for check in e.get("overlap_checks", []):
            overlaps.append(
                dict(
                    split=e["split"],
                    context=e["context"],
                    source_image_id=e["patch"],
                    **check,
                )
            )
    pd.DataFrame(sources).to_csv(A / "source_images.csv", index=False)
    pd.DataFrame(overlaps).to_csv(
        A / "within_WSI_overlap_checks_historical.csv", index=False
    )
    duplicates = []
    for digest, g in pd.DataFrame(sources).groupby("image_historical_sha256"):
        if g.split.nunique() > 1:
            duplicates.append(
                dict(
                    sha256=digest,
                    paths="|".join(g.image),
                    splits="|".join(sorted(g.split.unique())),
                )
            )
    pd.DataFrame(duplicates, columns=["sha256", "paths", "splits"]).to_csv(
        A / "cross_split_exact_image_duplicates.csv", index=False
    )
    if duplicates:
        raise ValueError("Exact image duplicate across official splits")
    print("BUILD EXPLICIT CELL KEYS", flush=True)
    columns = [
        "dataset_id",
        "image_id",
        "cell_id",
        "source_image_id",
        "split",
        "context",
        "ground_truth_original",
        "ground_truth_binary",
        "pixel_count",
        "mean_CD8",
    ]
    cells = pd.read_parquet(E / "audit/cells.parquet", columns=columns)
    cells["marker"] = "CD8"
    cells["cell_id"] = cells.cell_id.astype("int64")
    validate_keys(cells)
    if len(cells) != 1817783:
        raise ValueError("Unexpected population size")
    reg = pd.concat(registry, ignore_index=True)
    repeated = len(reg) - len(reg.drop_duplicates(KEYS))
    reg = reg.drop_duplicates(KEYS)
    reg["cell_id"] = reg.cell_id.astype("int64")
    exact_join(cells[KEYS], reg)
    meta = pd.read_parquet(
        O / "PhenoBIC_CD8_features.parquet",
        columns=[
            "dataset_id",
            "image_id",
            "cell_id",
            "patient_id",
            "batch_id",
            "mask_type",
            "qc_flags",
        ],
    )
    meta["marker"] = "CD8"
    meta["cell_id"] = meta.cell_id.astype("int64")
    cells = exact_join(cells, meta)
    if set(cells.mask_type) != set(["expanded_nucleus"]):
        raise ValueError("Mask semantics changed")
    print("ALIGN ORIGINAL LABEL CSV BY CELL ID", flush=True)
    annotations = []
    label_files = {}
    for uid, entries in sorted(units.items()):
        e = entries[0]
        src = e["source"]
        p = D / src["labels"]
        if str(p) not in label_files:
            label_files[str(p)] = pd.read_csv(p)
        d = label_files[str(p)].copy()
        if src["label_patch_filter"]:
            d = d[d.Patch.astype(str) == src["label_patch_filter"]].copy()
        d = d[["label", "CD8"]].rename(
            columns={"label": "cell_id", "CD8": "annotation_text"}
        )
        d["dataset_id"] = "PhenoBIC-Data"
        d["image_id"] = uid
        d["marker"] = "CD8"
        d["cell_id"] = d.cell_id.astype("int64")
        d["annotation_binary"] = d.annotation_text.map({"neg": 0, "pos": 1})
        d["annotation_source"] = str(p)
        validate_keys(d)
        annotations.append(d)
    ann = pd.concat(annotations, ignore_index=True)
    validate_keys(ann)
    unmatched = ann[KEYS].merge(cells[KEYS], on=KEYS, how="left", indicator=True)
    if not unmatched["_merge"].eq("both").all():
        raise ValueError("Annotation IDs missing from instance registry")
    cells = cells.merge(ann, on=KEYS, how="left", validate="one_to_one")
    if not np.allclose(
        cells.ground_truth_binary,
        cells.annotation_binary,
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise ValueError("Direct labels do not match existing cell table")
    known = cells.annotation_binary.isin([0, 1])
    if (
        not cells.loc[known, "ground_truth_original"]
        .eq(cells.loc[known, "annotation_text"])
        .all()
    ):
        raise ValueError("Original label strings mismatch")
    cells["ground_truth_original"] = cells.annotation_text
    cells["label_type"] = np.where(
        cells.annotation_text.notna(),
        "manual_independently_quality_reviewed",
        "unlabelled",
    )
    cells["label_binary_status"] = np.where(
        known,
        "known_pos_neg",
        np.where(cells.annotation_text.notna(), "unresolved_label", "unlabelled"),
    )
    cells["slide_id"] = (
        cells["split"] + "/" + cells["context"].str.replace("_CD8", "", regex=False)
    )
    patient = cells.patient_id.fillna("").astype(str).str.strip()
    cells["group_id"] = np.where(
        patient.ne(""),
        "PhenoBIC-Data/patient/" + patient,
        "PhenoBIC-Data/slide/" + cells.slide_id,
    )
    cells["grouping_level"] = np.where(patient.ne(""), "patient", "slide")
    if (cells.groupby("group_id").split.nunique() > 1).any():
        raise ValueError("Patient group crosses official splits")
    cells["experiment_role"] = cells.split.map(
        dict(
            training_WSIs="development",
            internal_val_WSIs="locked_internal_evaluation",
            external_val_TMA_cores="locked_external_evaluation",
        )
    )
    cells["prior_evaluation_exposure"] = cells.split.ne("training_WSIs")
    folds = sorted(cells.loc[cells.split == "training_WSIs", "group_id"].unique())
    if len(folds) != 3:
        raise ValueError("Expected three development WSI groups")
    cells["development_fold"] = cells.group_id.map({g: i for i, g in enumerate(folds)})
    foldplan = []
    for i, held in enumerate(folds):
        for group in folds:
            foldplan.append(
                dict(
                    fold=i,
                    group_id=group,
                    role="validation" if group == held else "training",
                )
            )
    pd.DataFrame(foldplan).to_csv(A / "development_group_folds.csv", index=False)
    keyrows = []
    countrows = []
    expected = {
        "training_WSIs": 101057,
        "internal_val_WSIs": 539132,
        "external_val_TMA_cores": 437480,
    }
    for split, g in cells.groupby("split"):
        lab = g[g.ground_truth_binary.isin([0, 1])]
        if len(lab) != expected[split]:
            raise ValueError("Labelled count changed: " + split)
        countrows.append(
            dict(
                split=split,
                n_all=len(g),
                n_labelled=len(lab),
                n_positive=int(lab.ground_truth_binary.sum()),
                n_negative=int(lab.ground_truth_binary.eq(0).sum()),
                n_unlabelled=int(g.annotation_text.isna().sum()),
                n_unresolved=int(g.label_binary_status.eq("unresolved_label").sum()),
                n_image_units=g.image_id.nunique(),
                n_group_units=g.group_id.nunique(),
                n_patient_ids=int(g.patient_id.fillna("").ne("").sum()),
            )
        )
        for population, t in [("all", g), ("labelled", lab)]:
            name = split + "_" + population + "_keys.parquet"
            t[KEYS].to_parquet(A / name, index=False)
            keyrows.append(
                dict(
                    split=split,
                    population=population,
                    rows=len(t),
                    key_sha256=key_hash(t),
                    path=str(A / name),
                )
            )
    pd.DataFrame(countrows).to_csv(A / "cell_counts.csv", index=False)
    pd.DataFrame(keyrows).to_csv(A / "key_manifest.csv", index=False)
    groupcols = [
        "dataset_id",
        "marker",
        "image_id",
        "slide_id",
        "group_id",
        "grouping_level",
        "split",
        "context",
        "experiment_role",
        "prior_evaluation_exposure",
    ]
    cells[groupcols].drop_duplicates().to_csv(A / "group_manifest.csv", index=False)
    labelsums = [
        dict(path=p, bytes=Path(p).stat().st_size, sha256=sha(p)) for p in label_files
    ]
    pd.DataFrame(labelsums).to_csv(A / "label_file_hashes.csv", index=False)
    print("SMALL REAL-IMAGE ALIGNMENT CHECK", flush=True)
    selected = []
    for split, context in [
        ("training_WSIs", None),
        ("internal_val_WSIs", None),
        ("external_val_TMA_cores", "TMA1_CD8"),
        ("external_val_TMA_cores", "TMA2_CD8"),
        ("external_val_TMA_cores", "TMA3_CD8"),
    ]:
        selected.append(
            next(
                e
                for e in AUDIT
                if e["split"] == split and (context is None or e["context"] == context)
            )
        )
    rng = np.random.RandomState(0)
    sample_rows = []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for unit_index, e in enumerate(selected):
        uid = image_id(e)
        ids, bb = boxes(e)
        lookup = {int(cid): box for cid, box in zip(ids, bb)}
        pool = cells[
            (cells.image_id == uid)
            & (cells.source_image_id == e["patch"])
            & cells.ground_truth_binary.isin([0, 1])
        ].sort_values("cell_id")
        pool = pool[
            pool.cell_id.map(
                lambda c: c in lookup
                and lookup[c][0] > 8
                and lookup[c][2] > 8
                and lookup[c][1] < e["shape"][0] - 8
                and lookup[c][3] < e["shape"][1] - 8
            )
        ]
        chosen = pool.iloc[
            np.sort(rng.choice(len(pool), min(3, len(pool)), replace=False))
        ]
        for j, r in enumerate(chosen.itertuples(index=False)):
            im, mask, valid, origin, truncated = roi(e, lookup[r.cell_id], halo=8)
            target = (mask == r.cell_id) & valid & np.isfinite(im)
            values = im[target].astype("float64")
            mean = float(values.mean()) if len(values) else np.nan
            mean_ok = bool(np.isclose(mean, r.mean_CD8, rtol=1e-6, atol=1e-7))
            count_ok = len(values) == r.pixel_count
            item = dict(
                dataset_id=r.dataset_id,
                image_id=uid,
                cell_id=r.cell_id,
                marker="CD8",
                split=r.split,
                source_image_id=e["patch"],
                ground_truth_original=r.ground_truth_original,
                ground_truth_binary=r.ground_truth_binary,
                image_dtype=str(im.dtype),
                mask_dtype=str(mask.dtype),
                observed_pixels=len(values),
                expected_pixels=r.pixel_count,
                pixel_count_matches=count_ok,
                raw_min=float(values.min()) if len(values) else np.nan,
                raw_max=float(values.max()) if len(values) else np.nan,
                raw_mean=mean,
                existing_raw_mean=r.mean_CD8,
                raw_mean_matches=mean_ok,
                label_join_verified=True,
                mask_id_present=bool(target.any()),
                valid_context_fraction=float(valid.mean()),
            )
            sample_rows.append(item)
            np.savez_compressed(
                A / "samples" / ("unit_%02d_cell_%s.npz" % (unit_index, r.cell_id)),
                CD8=im,
                instance_mask=mask,
                valid=valid,
                cell_id=r.cell_id,
                origin=origin,
            )
            if j == 0:
                fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                display = np.where(valid, im, np.nan)
                low = float(np.nanmin(display))
                high = float(np.nanmax(display))
                for ax in axes:
                    ax.imshow(display, cmap="gray", vmin=low, vmax=high)
                    ax.set_axis_off()
                axes[0].set_title("Raw CD8 (linear display)")
                axes[1].contour(
                    target.astype(float), levels=[0.5], colors=["lime"], linewidths=1
                )
                axes[1].set_title(
                    "Instance ID %s; label %s" % (r.cell_id, r.ground_truth_original)
                )
                fig.suptitle(uid + " | expanded nucleus", fontsize=9)
                fig.tight_layout()
                fig.savefig(
                    A / "samples" / ("input_alignment_%02d.png" % unit_index), dpi=160
                )
                plt.close(fig)
        CACHE.clear()
        print("SAMPLE", uid, "cells", len(chosen), flush=True)
    sm = pd.DataFrame(sample_rows)
    sm.to_csv(A / "sample_alignment.csv", index=False)
    if (
        len(sm) != 15
        or not sm[
            [
                "pixel_count_matches",
                "raw_mean_matches",
                "mask_id_present",
                "label_join_verified",
            ]
        ]
        .all()
        .all()
    ):
        raise ValueError("Real-image sample verification failed")
    cells = cells.drop(columns=["mean_CD8", "annotation_text", "annotation_binary"])
    cells.to_parquet(A / "cells.parquet", index=False, compression="zstd")
    feature_cfg = json.loads((O / "RUN_CONFIG.json").read_text())
    pd.DataFrame(
        {
            "available_existing_feature": feature_cfg["feature_columns"],
            "automatically_selected_for_paper_model": False,
        }
    ).to_csv(A / "available_features.csv", index=False)
    pending = dict(
        status="awaiting_paper",
        model_name=None,
        model_version=None,
        paper_reference=None,
        official_repository=None,
        code_revision=None,
        weights_sha256=None,
        pretraining_data=None,
        input_channels=None,
        official_preprocessing=None,
        threshold=None,
        threshold_operator=None,
        threshold_source=None,
        frozen_at=None,
        architecture_changed=False,
    )
    js(A / "MODEL_SPEC_PENDING.json", pending)
    protocol = dict(
        status="preflight_only_paper_not_selected",
        dataset="PhenoBIC-Data",
        marker="CD8",
        key_columns=KEYS,
        seed=0,
        mask_type="expanded_nucleus",
        mask_semantics_evidence="Inherited frozen RUN_CONFIG.json; current upstream PMC access returned a verification page, not independently reverified in this run",
        boundary_interpretation="Approximate expanded nuclear boundary; not a validated cell membrane",
        patient_ids_available=bool(patient.ne("").any()),
        grouping="patient if documented, else original WSI or TMA slide; TMA cores are not independent patients",
        fit_and_selection_scope="Official training_WSIs only; grouped development folds available; exact paper training protocol pending",
        preprocessing="Not selected. Preserve raw dtype/range; use target model official preprocessing after paper selection",
        validation_policy="Official internal/external splits locked for future model-specific choices; both have prior E1-E4 evaluation exposure",
        pan_gold="sealed; no file enumeration, image, label, distribution or prediction reads",
        cross_dataset_overlap="NOT CHECKED: Pan-M remains sealed; no no-overlap assertion",
        within_phenobic_overlap="Historical exact-file hashes plus current size/mtime check; overlapping internal FOVs use original union/ID logic; not a perceptual or patient identity proof",
        no_positive_metrics="Precision, recall and F1 use zero_division=0; AP and ROC-AUC undefined without positives; report flags and denominators",
        no_model_inputs="IDs, batch, truth and GMM scores excluded; feature selection not yet specified",
        code_commit=None,
        model_fits=0,
        model_predictions_generated=0,
    )
    js(A / "DATA_PROTOCOL.json", protocol)
    versions = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "tifffile",
        "pyarrow",
        "torch",
        "torchvision",
        "matplotlib",
        "psutil",
    ]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    gpu = command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    try:
        import torch

        cuda = dict(
            torch_cuda_version=torch.version.cuda,
            available=torch.cuda.is_available(),
            device_count=torch.cuda.device_count(),
        )
    except Exception as exc:
        cuda = dict(error=str(exc))
    hardware = dict(
        hostname=platform.node(),
        platform=platform.platform(),
        python=sys.version,
        executable=sys.executable,
        logical_cpu_count=os.cpu_count(),
        cpu=command(["lscpu"]),
        memory=Path("/proc/meminfo").read_text(),
        gpu=gpu,
        cuda=cuda,
        disk=dict(zip(["total", "used", "free"], shutil.disk_usage(P))),
    )
    js(
        A / "ENVIRONMENT.json",
        dict(
            versions=versions,
            hardware=hardware,
            git=command(["git", "-C", str(P), "rev-parse", "HEAD"]),
        ),
    )
    codepaths = list(Path(__file__).parent.glob("*.py")) + [
        E / "scripts/io_adapter.py",
        E / "scripts/common.py",
        O / "scripts/batch_features.py",
        G / "scripts/extract_cd8_v2.py",
        G / "scripts/analyze_cd8_v2.py",
    ]
    pd.DataFrame(
        [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in codepaths]
    ).to_csv(A / "code_hashes.csv", index=False)
    runtime = dict(
        stage="data_preflight_only",
        elapsed_seconds=time.perf_counter() - start,
        peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        training_seconds=None,
        inference_seconds=None,
        peak_model_gpu_MiB=None,
        model_not_selected=True,
        benchmark_scope="No model training/inference benchmark has been performed",
    )
    js(A / "RUNTIME.json", runtime)
    js(
        A / "PREFLIGHT_COMPLETE.json",
        dict(
            status="preflight_complete_awaiting_paper",
            all_cells=len(cells),
            labelled_cells=int(known.sum()),
            image_files=len(AUDIT),
            image_units=len(units),
            old_files_current_size_mtime_verified=len(checks),
            source_registry_repeated_FOV_observations=repeated,
            original_label_files_hashed=len(labelsums),
            real_sample_cells=len(sm),
            pan_gold_accessed=False,
            full_new_input_sha256_scan=False,
            new_model_fits=0,
            new_model_predictions=0,
        ),
    )
    print("PREFLIGHT COMPLETE; MODEL AWAITS PAPER", len(cells), flush=True)


if __name__ == "__main__":
    main()
