"""Verify the published CD8 evidence without images, weights or GPU access."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]


def rows(relative):
    with (RUN / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class PublicationEvidence(unittest.TestCase):
    def test_source_manifest_and_frozen_selection(self):
        for record in rows("provenance/SOURCE_MANIFEST.csv"):
            actual = hashlib.sha256((RUN / record["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, record["publication_sha256"], record["path"])
            if record["path"].startswith(("results/", "models/", "reports/figures/")):
                self.assertEqual(actual, record["source_sha256"], record["path"])
        selection = RUN / "models/MODEL_SELECTION_FROZEN.json"
        frozen = json.loads(selection.read_text())
        self.assertEqual(frozen["selected_model"], "H1")
        self.assertEqual(len(frozen["model_file_sha256"]), 22)
        self.assertEqual(
            hashlib.sha256(selection.read_bytes()).hexdigest(),
            "288f2444df40fc1c0d82fa0105d044cffda74cd15bc39d922a8a207d869327a3",
        )

    def test_all_figure_exports_match_saved_hashes(self):
        exports = rows("reports/FIGURE_EXPORTS.csv")
        self.assertEqual(len(exports), 24)
        for record in exports:
            self.assertEqual(
                hashlib.sha256((RUN / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
                record["path"],
            )

    def test_confusion_counts_and_ensemble_semantics(self):
        external = rows("results/EXTERNAL_MAIN.csv")
        for record in external:
            tp, fp, fn, tn = (int(record[k]) for k in ("tp", "fp", "fn", "tn"))
            self.assertEqual(tp + fp + fn + tn, 437480)
            self.assertAlmostEqual(
                float(record["F1"]), 2 * tp / (2 * tp + fp + fn), places=12
            )
        ens = next(
            r
            for r in external
            if r["model"] == "ENS_image" and r["threshold_mode"] == "fixed_0.5"
        )
        mean = next(
            r
            for r in rows("reports/CONTROLLED_IMAGE_SEED_SUMMARY.csv")
            if r["group"] == "image" and r["threshold_mode"] == "fixed_0.5"
        )
        self.assertNotAlmostEqual(float(ens["AP"]), float(mean["AP_mean"]), places=4)

    def test_calibration_population_and_real_dtype_impact(self):
        calibration = [
            r
            for r in rows("results/TARGET_LABEL_CALIBRATION_SIMULATION.csv")
            if r["level"] == "split" and r["stratum"] == ""
        ]
        self.assertTrue(calibration)
        self.assertEqual(
            {r["budget_per_core"] for r in calibration}, {"0", "20", "50", "100"}
        )
        self.assertTrue(all(int(r["n_cells"]) == 433280 for r in calibration))
        self.assertTrue(
            all(
                r["same_evaluation_population_across_budgets"] == "True"
                for r in calibration
            )
        )
        audit = rows("results/CALIBRATION_NATIVE_THRESHOLD_AUDIT.csv")
        self.assertEqual(
            sum(int(r["old_implementation_changed_decisions"]) for r in audit), 0
        )
        self.assertTrue(all(r["native_mapping_exact"] == "True" for r in audit))

    def test_composition_denominator(self):
        stress = {
            r["model"]: r
            for r in rows("results/COMPOSITION_SUMMARY.csv")
            if r["pool"] == "lowest_70pct"
        }
        for model, flips in {"H0": 153, "H1": 137, "H2": 140}.items():
            self.assertEqual(int(stress[model]["n_targets"]), 300)
            self.assertAlmostEqual(
                float(stress[model]["prediction_change_fraction"]), flips / 300
            )

    def test_legacy_algorithms_are_unchanged(self):
        for legacy, publication in [
            (
                "analyses/cd8_e1_e4_stability_20261023/scripts/compact_features.py",
                "legacy_compact_features.py",
            ),
            (
                "analyses/cd8_morphospatial_20261023/scripts/batch_features.py",
                "legacy_batch_features.py",
            ),
        ]:

            def definitions(path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                return {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                }

            self.assertEqual(
                definitions(REPO / legacy), definitions(RUN / "scripts" / publication)
            )

    def test_portable_roots_do_not_read_data(self):
        env = dict(
            os.environ,
            MIF_PROJECT_ROOT=str(REPO / "nonexistent_inputs"),
            MIF_RUN_ROOT=str(REPO / "nonexistent_outputs"),
            MIF_SOURCE_ROOT=str(REPO),
            PYTHONDONTWRITEBYTECODE="1",
        )
        code = "import runtime_paths as p; print(p.ROOT); print(p.RUN); print(p.SOURCE_ROOT)"
        output = subprocess.check_output(
            [sys.executable, "-X", "utf8", "-B", "-c", code],
            cwd=RUN / "scripts",
            env=env,
            text=True,
        )
        self.assertEqual(
            output.splitlines(),
            [
                str((REPO / "nonexistent_inputs").resolve()),
                str((REPO / "nonexistent_outputs").resolve()),
                str(REPO),
            ],
        )

    def test_no_private_cases_and_review_remains_pending(self):
        self.assertEqual({p.name for p in (RUN / "review").iterdir()}, {"README.md"})
        public = json.loads((RUN / "provenance/FINAL_DELIVERY_PUBLIC.json").read_text())
        self.assertFalse(public["real_images_published"])
        self.assertIn("pending", public["human_case_review"])
        self.assertFalse(public["scientific_joint_target_met"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
