"""Verify the final report, numerical artifacts, model bytes and small review evidence."""
from pathlib import Path
import sys
import json
import datetime
import pandas as pd
from PIL import Image

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from common import sha, code_hash, dump, WEIGHT_SHA


def main():
    rows = []
    verified = {}

    def check(name, condition, detail):
        assert bool(condition), name + ": " + str(detail)
        rows.append(dict(check=name, passed=True, detail=str(detail)))

    def verify(path, expected):
        path = Path(path)
        if str(path) not in verified:
            check("sha256:" + str(path), sha(path) == expected, expected)
            verified[str(path)] = dict(sha256=expected, bytes=path.stat().st_size)
        else:
            assert verified[str(path)]["sha256"] == expected

    config = json.loads((R / "RUN_CONFIG.json").read_text())
    verify(R / "NIMBUS_PROTOCOL.json", config["protocol_sha256"])
    check(
        "frozen_computational_code",
        code_hash() == config["code_sha256"],
        config["code_sha256"],
    )
    delivery = json.loads((R / "provenance/delivery_complete.json").read_text())
    check("delivery_complete", delivery["status"] == "complete", delivery["status"])
    verify(R / "ANALYSIS_REPORT.md", delivery["report_sha256"])
    fine = json.loads((R / "models/finetuned_complete.json").read_text())
    verify(R / "models/Nimbus_outerP1P99_finetuned.pt", fine["weight_sha256"])
    verify(R / "weights/V1.pt", WEIGHT_SHA)
    verify(R / "upstream/src/nimbus_inference/assets/V1.pt", WEIGHT_SHA)
    models = ["GMM", "Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned"]
    for model in models:
        state = json.loads(
            (R / "provenance" / (model + "_evaluation_complete.json")).read_text()
        )
        check(
            "evaluation_rows:" + model,
            state["all_cells"] == 1817783,
            state["all_cells"],
        )
        check(
            "evaluation_protocol:" + model,
            state["signature"]["protocol_sha256"] == config["protocol_sha256"],
            state["signature"]["protocol_sha256"],
        )
        for path, expected in state["signature"]["prediction_sources"].items():
            verify(path, expected)
        for path, expected in state["output_sha256"].items():
            verify(path, expected)
        print("VERIFIED_MODEL_ARTIFACTS", model, flush=True)
    all_metrics = pd.read_csv(
        R / "results/all_metrics.csv", float_precision="round_trip"
    )
    expected_metrics = pd.concat(
        [
            pd.read_csv(
                R / "results" / (m + "_metrics.csv"), float_precision="round_trip"
            )
            for m in models
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        all_metrics, expected_metrics, check_exact=False, atol=1e-12, rtol=0
    )
    check("all_metrics_exact_concatenation", True, len(all_metrics))
    counts = {
        "training_WSIs": 101057,
        "internal_val_WSIs": 539132,
        "external_val_TMA_cores": 437480,
    }
    split_rows = all_metrics.loc[all_metrics.level == "split"]
    check(
        "labelled_split_denominators",
        all(split_rows.n == split_rows.split.map(counts)),
        counts,
    )
    check(
        "prediction_coverage",
        all_metrics.n_failed.eq(0).all() and all_metrics.coverage.eq(1).all(),
        "all formal groups 100%",
    )
    integrity = json.loads(
        (R / "provenance/prediction_integrity_passed.json").read_text()
    )
    check(
        "full_cell_and_unit_coverage",
        integrity["total_cells_per_model"] == 1817783
        and integrity["models"] == 2
        and integrity["units"] == 48
        and integrity["total_unscorable"] == 0,
        integrity,
    )
    cells = pd.read_csv(R / "results/prediction_integrity.csv")
    check(
        "no_duplicate_or_missing_pixels",
        cells.duplicate_keys.eq(0).all()
        and cells.unpredicted_original_pixels.eq(0).all(),
        len(cells),
    )
    seam = pd.read_csv(R / "results/seam_cell_audit.csv")
    check(
        "seam_ID_coverage",
        seam.n_missing_seam_IDs.eq(0).all()
        and seam.n_near_seam_cells.eq(seam.n_finite_scores).all(),
        int(seam.n_near_seam_cells.sum()),
    )
    mask = pd.read_csv(R / "results/mask_pixel_count_comparison.csv")
    check(
        "original_instance_pixel_counts",
        mask.n_pixel_count_differences.eq(0).all(),
        int(mask.n_pixel_count_differences.sum()),
    )
    calibration = json.loads(
        (R / "provenance/calibration_coverage_complete.json").read_text()
    )
    verify(R / "results/calibration_coverage.csv", calibration["table_sha256"])
    coverage = pd.read_csv(R / "results/calibration_coverage.csv")
    check(
        "calibration_coverage",
        coverage.n_calibration_invalid.eq(0).all()
        and coverage.calibration_coverage.eq(1).all(),
        len(coverage),
    )
    review = pd.read_csv(R / "development_review/development_cases.csv")
    check(
        "development_only_case_images",
        review.split.isin(["training_WSIs", "internal_val_WSIs"]).all(),
        len(review),
    )
    layout = json.loads((R / "provenance/figure_layout_finalization.json").read_text())
    for path, expected in {
        **layout["source_sha256"],
        **layout["output_sha256"],
    }.items():
        verify(R / path, expected)
    for path in sorted((R / "figures").glob("*/*")):
        if path.suffix == ".png":
            with Image.open(path) as img:
                img.verify()
            check("readable_png:" + str(path.relative_to(R)), True, path.stat().st_size)
        elif path.suffix == ".svg":
            check(
                "editable_svg_text:" + str(path.relative_to(R)),
                "<text" in path.read_text(encoding="utf-8"),
                path.stat().st_size,
            )
    table = R / "results/delivery_validation.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    dump(
        R / "provenance/FINAL_DELIVERY_VALIDATION.json",
        dict(
            status="complete",
            created=datetime.datetime.now().isoformat(),
            checks=len(rows),
            verified_files=len(verified),
            verified_bytes=sum(x["bytes"] for x in verified.values()),
            protocol_sha256=config["protocol_sha256"],
            computational_code_sha256=config["code_sha256"],
            report_sha256=delivery["report_sha256"],
            validation_csv_sha256=sha(table),
            pretrained_weight_sha256=WEIGHT_SHA,
            finetuned_weight_sha256=fine["weight_sha256"],
            script_sha256=sha(Path(__file__)),
        ),
    )
    print("FINAL_DELIVERY_VALIDATED", len(rows), flush=True)


if __name__ == "__main__":
    main()
