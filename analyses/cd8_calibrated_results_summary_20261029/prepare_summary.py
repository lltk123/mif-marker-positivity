"""Summarize frozen CD8 metrics using the Python standard library.

Run with --output-dir NEW_DIRECTORY to write the eight summary CSVs.
The existing image data, models, metrics, and prediction files are never changed.
"""
import argparse, csv, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = "analyses/cd8_e1_e4_stability_20261023/"
T = "analyses/cd8_tabm_20261029/"
N = "analyses/cd8_nimbus_20261024/"
MODELS = {"E1", "E2", "E3", "E4", "GMM"}
SPLITS = {"external_val_TMA_cores": 437480, "internal_val_WSIs": 539132}
SOURCES, CHECKS = {}, []


def source(relative):
    path = ROOT / relative
    data = path.read_bytes()
    SOURCES[relative] = dict(
        source_path=relative, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
    )
    return path


def value(text):
    if text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
            return number if math.isfinite(number) else None
        except ValueError:
            return text


def rows(relative):
    with source(relative).open(encoding="utf-8-sig", newline="") as handle:
        return [
            {**{k: value(v) for k, v in row.items()}, "source_path": relative}
            for row in csv.DictReader(handle)
        ]


def check(name, passed, detail):
    CHECKS.append(dict(check=name, passed=bool(passed), detail=str(detail)))
    if not passed:
        raise ValueError(f"{name}: {detail}")


def selected(items):
    return [r for r in items if r["model"] in MODELS and r["split"] in SPLITS]


def ranked(items, groups, metrics):
    for row in items:
        peers = [r for r in items if all(r[k] == row[k] for k in groups)]
        for metric in metrics:
            row[f"rank_{metric}"] = 1 + sum(r[metric] > row[metric] for r in peers)
    return sorted(
        items,
        key=lambda r: tuple(str(r[k]) for k in groups) + (r[f"rank_{metrics[0]}"],),
    )


def main(payload_path=None, output_dir=None):
    protocol = json.loads(
        source(E + "E1_E4_PROTOCOL.json").read_text(encoding="utf-8-sig")
    )
    frozen = json.loads(
        source(E + "THRESHOLDS_FROZEN.json").read_text(encoding="utf-8-sig")
    )
    split = selected(rows(E + "results/metrics_split.csv"))
    check("fixed_rows", len(split) == 20, "2 splits x 2 threshold panels x 5 methods")
    for row in split:
        key = f'{row["split"]}/{row["model"]}/{row["threshold_mode"]}'
        tp, fp, fn, tn = (row[c] for c in ("TP", "FP", "FN", "TN"))
        check(
            key + "/population",
            row["n_total"] == SPLITS[row["split"]] == tp + fp + fn + tn,
            row["n_total"],
        )
        check(
            key + "/coverage",
            row["n_failed"] == 0 and row["prediction_coverage"] == 1,
            "full labelled population",
        )
        for metric, rebuilt in {
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "F1": 2 * tp / (2 * tp + fp + fn),
        }.items():
            check(
                key + "/" + metric,
                abs(row[metric] - rebuilt) < 1e-12,
                "recomputed from TP/FP/FN",
            )
        oof = row["threshold_mode"] == "OOF_threshold" and row["model"] != "GMM"
        expected = frozen["thresholds"][row["model"]] if oof else 0.5
        check(key + "/threshold", row["threshold"] == expected, expected)
        row.update(
            threshold_panel="fixed_0.5"
            if row["threshold_mode"] == "fixed_0.5"
            else "training_OOF_E1_E4_GMM_original",
            threshold_is_OOF_selected=oof,
            feature_count=len(protocol["feature_sets"][row["model"]])
            if row["model"] != "GMM"
            else 1,
            supervision="none"
            if row["model"] == "GMM"
            else "original_3_training_WSI_labels",
            target_unit_unlabelled_adaptation=True,
            seed=0,
        )
    split = ranked(
        split,
        ("split", "threshold_panel"),
        ("F1", "AP", "ROC_AUC", "precision", "recall"),
    )
    contexts = ranked(
        selected(rows(E + "results/metrics_context.csv")),
        ("split", "context", "threshold_mode"),
        ("F1", "AP"),
    )
    units = selected(rows(E + "results/metrics_unit.csv"))
    macro = ranked(
        selected(rows(E + "results/metrics_macro_units.csv")),
        ("split", "threshold_mode"),
        ("F1_mean", "AP_mean"),
    )
    for row in split:
        peers = [
            r
            for r in units
            if all(r[c] == row[c] for c in ("split", "model", "threshold_mode"))
        ]
        check(
            f'unit_sum/{row["split"]}/{row["model"]}/{row["threshold_mode"]}',
            all(
                sum(r[c] for r in peers) == row[c]
                for c in ("TP", "FP", "FN", "TN", "n_total")
            ),
            len(peers),
        )
    paired = []
    for row in split:
        if row["model"] == "GMM":
            continue
        base = next(
            r
            for r in split
            if r["model"] == "GMM"
            and r["split"] == row["split"]
            and r["threshold_panel"] == row["threshold_panel"]
        )
        paired.append(
            dict(
                split=row["split"],
                threshold_panel=row["threshold_panel"],
                model=row["model"],
                reference="GMM_original_0.5",
                **{
                    f"delta_{c}": row[c] - base[c]
                    for c in (
                        "F1",
                        "AP",
                        "ROC_AUC",
                        "precision",
                        "recall",
                        "TP",
                        "FP",
                        "FN",
                    )
                },
                source_path=E + "results/metrics_split.csv",
            )
        )
    inventory = []
    for row in rows(E + "results/metrics_split.csv"):
        if (
            row["split"] != "external_val_TMA_cores"
            or row["threshold_mode"] != "fixed_0.5"
        ):
            continue
        model = row["model"]
        inventory.append(
            dict(
                model=model,
                completed=True,
                main_calibrated_ranking=model in MODELS,
                calibration="target_unit_unsupervised_GMM"
                if model == "GMM"
                else "target_unit_common_LU"
                if model in MODELS
                else "raw_features_no_common_LU",
                independence_status="no_evaluation_labels_in_calibration_or_training; retrospective_external",
                n_seeds=1,
                n_labelled_external=row["n_total"],
                F1_fixed_0_5=row["F1"],
                AP=row["AP"],
                source_path=row["source_path"],
            )
        )
    for row in rows(T + "results/report_performance_fixed_0.5.csv"):
        if (
            row["split"] != "external_val_TMA_cores"
            or row["group_type"] != "split"
            or row["model_family"] not in ("TabM", "HGB")
        ):
            continue
        inventory.append(
            dict(
                model=f'{row["model_family"]}_{row["feature_set"]}_TabM_run',
                completed=True,
                main_calibrated_ranking=False,
                calibration="train_only_imputation_QT_bins; no_target_unit_common_LU",
                independence_status="internal_validation_used_for_development; retrospective_external",
                n_seeds=row["n_seeds"],
                n_labelled_external=row["n_cells_mean"],
                F1_fixed_0_5=row["f1_mean"],
                F1_seed_sd=row["f1_std"],
                AP=row["average_precision_mean"],
                AP_seed_sd=row["average_precision_std"],
                source_path=row["source_path"],
            )
        )
    for row in rows(N + "results/overall_metrics.csv"):
        if (
            row["split"] != "external_val_TMA_cores"
            or row["threshold_mode"] != "official_default"
            or row["model"] == "GMM"
        ):
            continue
        inventory.append(
            dict(
                model=row["model"],
                completed=True,
                main_calibrated_ranking=False,
                calibration="official_train_only_p99.9_factor; no_target_unit_common_LU",
                independence_status="pretraining_overlap_unresolved; internal_validation_used_for_development",
                n_seeds=1,
                n_labelled_external=437480,
                F1_fixed_0_5=row["F1"],
                AP=row["AP"],
                source_path=row["source_path"],
            )
        )
    for relative in (
        E + "audit/AUDIT_COMPLETE.json",
        E + "results/FINAL_VALIDATION_CHECKS.csv",
        E + "results/FINAL_VALIDATION.json",
        E + "results/OOF_threshold_grid.csv",
        E + "ANALYSIS_REPORT.md",
        T + "TABM_PROTOCOL.json",
        N + "ANALYSIS_REPORT.md",
        N + "results/pretraining_exposure_audit.csv",
        "analyses/cd8_scimap_gmm_20261023/PROTOCOL.json",
    ):
        source(relative)
    check(
        "unique_inventory",
        len({r["model"] for r in inventory}) == len(inventory),
        len(inventory),
    )
    tables = {
        "CALIBRATED_RANKINGS.csv": split,
        "CALIBRATED_CONTEXT_METRICS.csv": contexts,
        "CALIBRATED_UNIT_METRICS.csv": units,
        "CALIBRATED_MACRO_METRICS.csv": macro,
        "PAIRED_VS_GMM.csv": paired,
        "EXPERIMENT_INVENTORY.csv": inventory,
        "SOURCE_MANIFEST.csv": list(SOURCES.values()),
        "AUDIT_CHECKS.csv": CHECKS,
    }
    config = dict(
        date="2026-10-29",
        task="summary of completed frozen experiments",
        primary_rank="external pooled labelled-cell F1; strict score > 0.5",
        secondary_panel="E1-E4 training OOF thresholds; GMM retains original >0.5",
        models=sorted(MODELS),
        splits=SPLITS,
        no_model_fit_or_prediction_run=True,
        pan_m_accessed=False,
        source_root=str(ROOT),
        limitations="Unsupervised target-image adaptation; historical external exposure; unavailable patient/source-image independence audit",
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if payload_path is not None:
        with payload_path.open("x", encoding="utf-8") as handle:
            json.dump(
                dict(config=config, tables=tables),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
    if output_dir is not None:
        export_tables(output_dir, tables, config)
    print(
        json.dumps(
            dict(tables={k: len(v) for k, v in tables.items()}, checks=len(CHECKS)),
            ensure_ascii=False,
        )
    )


def export_tables(output_dir, tables, config):
    """Write static CSVs to a new directory and retain source precision."""
    output_dir.mkdir(parents=True, exist_ok=False)
    preferred = [
        "rank_F1",
        "rank_F1_mean",
        "model",
        "split",
        "context",
        "threshold_panel",
        "threshold_mode",
        "threshold",
        "F1",
        "precision",
        "recall",
        "AP",
        "ROC_AUC",
        "n_total",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    manifest = []
    for filename, records in tables.items():
        all_columns = list(dict.fromkeys(key for row in records for key in row))
        columns = [key for key in preferred if key in all_columns]
        columns += [key for key in all_columns if key not in preferred]
        path = output_dir / filename
        with path.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in records:
                writer.writerow(
                    {
                        key: str(item).lower() if isinstance(item, bool) else item
                        for key, item in row.items()
                    }
                )
        manifest.append(
            {
                "file": filename,
                "rows": len(records),
                "columns": len(columns),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for filename, data in (
        ("SUMMARY_CONFIG.json", config),
        ("OUTPUT_MANIFEST.json", manifest),
    ):
        with (output_dir / filename).open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        type=Path,
        help="Optional new JSON file for the original JS exporter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory for CSVs and manifests; existing directories are rejected.",
    )
    args = parser.parse_args()
    if args.payload is None and args.output_dir is None:
        parser.error("Provide --output-dir or --payload.")
    main(args.payload, args.output_dir)
