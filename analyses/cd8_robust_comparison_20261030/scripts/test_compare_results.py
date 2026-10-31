"""Fault injection for completed-run comparison. No new model results are used."""
from pathlib import Path
import copy
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

import compare_results as comparison


class ComparisonContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[2]
        old_tabm = comparison.read_tabm(
            base / "cd8_tabm_20261029/results/metrics_all.csv", "old"
        )
        old_nimbus = comparison.read_nimbus(
            base / "cd8_nimbus_20261024/results/all_metrics.csv", "old"
        )
        # Explicit synthetic duplicate: this tests comparison bookkeeping only.
        # It is not an executed new normalization experiment.
        new_tabm = old_tabm.loc[old_tabm.variant.eq("old")].copy()
        new_nimbus = old_nimbus.copy()
        for frame in [new_tabm, new_nimbus]:
            frame["variant"] = "new"
            frame["normalization"] = comparison.METHOD
            frame["n_calibration_invalid"] = 0
            frame["calibration_coverage"] = 1.0
        cls.fixture = pd.concat(
            [old_tabm, new_tabm, old_nimbus, new_nimbus], ignore_index=True
        )

    def frame(self):
        return self.fixture.copy(deep=True)

    @staticmethod
    def one_row(frame):
        return (
            frame.variant.eq("new")
            & frame.model_family.eq("TabM")
            & frame.feature_set.eq("C")
            & frame.seed.eq(2)
            & frame.split.eq("external_val_TMA_cores")
            & frame.level.eq("split")
            & frame.threshold_mode.eq("fixed_0.5")
        )

    def test_complete_fixture_and_zero_paired_changes(self):
        frame = self.frame()
        comparison.assert_complete(frame)
        self.assertEqual(len(frame), 4422)
        summary = comparison.build_summary(frame)
        self.assertTrue(summary.seed_matrix_complete.all())
        new = summary.loc[summary.variant.eq("new")]
        old = summary.loc[summary.variant.eq("old")]
        keys = (
            ["model_family", "feature_set"] + comparison.UNIT_KEYS + ["threshold_mode"]
        )
        pairs = new.merge(
            old, on=keys, suffixes=("_new", "_old"), validate="one_to_one"
        )
        self.assertEqual(len(pairs), 888)
        for field in [
            "f1",
            "precision",
            "recall",
            "average_precision",
            "tp",
            "fp",
            "fn",
        ]:
            np.testing.assert_allclose(
                pairs[field + "_mean_new"], pairs[field + "_mean_old"], equal_nan=True
            )

    def test_missing_one_seed_row_is_rejected(self):
        frame = self.frame()
        selected = self.one_row(frame)
        self.assertEqual(selected.sum(), 1)
        with self.assertRaisesRegex(ValueError, "Incomplete unit grid"):
            comparison.assert_complete(frame.loc[~selected].copy())

    def test_missing_threshold_mode_is_rejected(self):
        frame = self.frame()
        selected = (
            frame.variant.eq("new")
            & frame.model_family.eq("TabM")
            & frame.feature_set.eq("A")
            & frame.seed.eq(1)
            & frame.threshold_mode.eq("validation_selected")
        )
        with self.assertRaisesRegex(ValueError, "combinations"):
            comparison.assert_complete(frame.loc[~selected].copy())

    def test_missing_gmm_is_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "including GMM"):
            comparison.assert_complete(frame.loc[frame.model_family.ne("GMM")].copy())

    def test_missing_same_unit_everywhere_is_rejected(self):
        frame = self.frame()
        unit = frame.loc[
            frame.level.eq("image") & frame.split.eq("external_val_TMA_cores"), "unit"
        ].iloc[0]
        with self.assertRaisesRegex(ValueError, "Incomplete unit grid"):
            comparison.assert_complete(
                frame.loc[~(frame.level.eq("image") & frame.unit.eq(unit))].copy()
            )

    def test_changed_new_unit_key_is_rejected(self):
        frame = self.frame()
        selected = (
            frame.variant.eq("new")
            & frame.model_family.eq("Nimbus")
            & frame.feature_set.eq("pretrained")
            & frame.level.eq("image")
        )
        old_unit = frame.loc[selected, "unit"].iloc[0]
        frame.loc[selected & frame.unit.eq(old_unit), "unit"] = old_unit + "_incorrect"
        with self.assertRaisesRegex(ValueError, "Image-unit keys"):
            comparison.assert_complete(frame)

    def test_scoring_failure_is_retained_and_not_averaged_away(self):
        frame = self.frame()
        selected = self.one_row(frame)
        frame.loc[selected, "n_score_failed"] = 1
        frame.loc[selected, "n_score_finite"] -= 1
        frame.loc[selected, "prediction_coverage"] = (
            frame.loc[selected, "n_score_finite"] / frame.loc[selected, "n_cells"]
        )
        frame.loc[selected, ["tp", "fp", "fn", "tn"]] = np.nan
        comparison.assert_complete(frame)
        self.assertEqual(len(frame), 4422)
        self.assertTrue(frame.loc[selected, "f1"].isna().all())
        summary = comparison.build_summary(frame)
        row = summary.loc[
            summary.variant.eq("new")
            & summary.model_family.eq("TabM")
            & summary.feature_set.eq("C")
            & summary.split.eq("external_val_TMA_cores")
            & summary.level.eq("split")
            & summary.threshold_mode.eq("fixed_0.5")
        ]
        self.assertEqual(row.n_seeds.iloc[0], 3)
        self.assertFalse(row.all_seeds_complete.iloc[0])
        self.assertTrue(np.isnan(row.f1_mean.iloc[0]))
        self.assertGreater(row.n_score_failed_mean.iloc[0], 0)

    def test_calibration_failure_distinct_from_finite_model_score(self):
        frame = self.frame()
        selected = self.one_row(frame)
        frame.loc[selected, "n_calibration_invalid"] = 10
        frame.loc[selected, "calibration_coverage"] = (
            1 - 10 / frame.loc[selected, "n_cells"]
        )
        comparison.assert_complete(frame)
        self.assertTrue(frame.loc[selected, "prediction_coverage"].eq(1).all())
        self.assertTrue(frame.loc[selected, "f1"].notna().all())
        summary = comparison.build_summary(frame)
        row = summary.loc[
            summary.variant.eq("new")
            & summary.model_family.eq("TabM")
            & summary.feature_set.eq("C")
            & summary.split.eq("external_val_TMA_cores")
            & summary.level.eq("split")
            & summary.threshold_mode.eq("fixed_0.5")
        ]
        self.assertTrue(row.all_seeds_complete.iloc[0])
        self.assertFalse(row.all_calibration_valid.iloc[0])

    def test_missing_new_calibration_audit_rejected(self):
        frame = self.frame()
        frame.loc[self.one_row(frame), "calibration_coverage"] = np.nan
        with self.assertRaisesRegex(ValueError, "calibration audit"):
            comparison.assert_complete(frame)

    def test_bad_scored_population_rejected(self):
        frame = self.frame()
        frame.loc[self.one_row(frame), "n_score_finite"] -= 1
        with self.assertRaisesRegex(ValueError, "cover the population"):
            comparison.assert_complete(frame)

    def test_changed_comparison_code_and_protocol_reject_cache(self):
        signature = {
            "source_sha256": {"data.csv": "same"},
            "completion_manifest_sha256": {"manifest": "same"},
            "comparison_code_sha256": "code1",
            "protocol_sha256": "protocol1",
            "require_delivery": False,
        }
        self.assertTrue(comparison.cache_compatible(dict(signature), signature))
        for key in [
            "comparison_code_sha256",
            "protocol_sha256",
            "completion_manifest_sha256",
        ]:
            changed = dict(signature, **{key: "changed"})
            self.assertFalse(comparison.cache_compatible(signature, changed))

    def test_completed_tabm_markers_required_and_bound_to_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "results").mkdir()
            frozen = {
                "normalization_method": comparison.METHOD,
                "models": {
                    f"{family}_{group}_s{seed}": {}
                    for family in ["TabM", "HGB"]
                    for group in "ABC"
                    for seed in range(3)
                },
            }
            path = run / "FROZEN_TEST_PROTOCOL.json"
            path.write_text(json.dumps(frozen))
            checksum = comparison.sha(path)
            (run / "PREDICTIONS_COMPLETE.json").write_text(
                json.dumps(dict(models=18, n_per_model=1817783, frozen_sha256=checksum))
            )
            with self.assertRaises(FileNotFoundError):
                comparison.verify_tabm_completion(run, "new", False)
            evaluation = dict(
                status="complete",
                n_models=19,
                n_all_cells=1817783,
                n_metric_rows=1998,
                protocol_sha256=checksum,
                normalization_method=comparison.METHOD,
                curves="full ROC/PR arrays",
            )
            (run / "results/EVALUATION_MANIFEST.json").write_text(
                json.dumps(evaluation)
            )
            self.assertEqual(
                len(comparison.verify_tabm_completion(run, "new", False)), 3
            )
            path.write_text(json.dumps(dict(frozen, changed=True)))
            with self.assertRaisesRegex(ValueError, "not complete or frozen"):
                comparison.verify_tabm_completion(run, "new", False)

    def test_nimbus_concatenation_cannot_change_authoritative_rows(self):
        base = Path(__file__).resolve().parents[2] / "cd8_nimbus_20261024/results"
        combined = comparison.read_csv(base / "all_metrics.csv")
        individual = comparison.read_csv(base / "Nimbus_pretrained_metrics.csv")
        comparison.assert_same_metric_rows(combined, individual, "Nimbus_pretrained")
        changed = combined.copy()
        index = changed.index[changed.model.eq("Nimbus_pretrained")][0]
        changed.loc[index, "F1"] = 0.123456789
        with self.assertRaises(AssertionError):
            comparison.assert_same_metric_rows(changed, individual, "Nimbus_pretrained")

    def test_real_old_completion_markers_verify_read_only(self):
        base = Path(__file__).resolve().parents[2]
        comparison.verify_tabm_completion(base / "cd8_tabm_20261029", "old", True)
        comparison.verify_nimbus_completion(base / "cd8_nimbus_20261024", "old", True)


class LegacyNimbusDeliveryContracts(unittest.TestCase):
    @staticmethod
    def legacy_run(directory):
        run = Path(directory)
        evidence = [
            "results/final_delivery_checks.csv",
            "results/GMM_reuse_audit.csv",
            "results/prediction_integrity.csv",
            "figures/FIGURE_QA.json",
        ]
        for name in evidence:
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic evidence")
        delivery = dict(
            completed="2026-10-24T14:11:55.773427",
            numeric_checks_passed=True,
            visual_review_complete=True,
            PanM_accessed=False,
            model_count_new=2,
            n_cells_per_model=1817783,
            n_units=48,
            evidence=evidence,
        )
        marker = run / "provenance/delivery_complete.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(delivery))
        return run, marker, delivery

    def test_complete_legacy_delivery_is_accepted_and_evidence_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            run, marker, delivery = self.legacy_run(temp)
            files = comparison.verify_nimbus_delivery(run, "old")
            self.assertEqual(len(files), 5)
            self.assertIn(run / "figures/FIGURE_QA.json", files)

    def test_missing_legacy_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run, marker, delivery = self.legacy_run(temp)
            (run / delivery["evidence"][0]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "evidence missing"):
                comparison.verify_nimbus_delivery(run, "old")
        with tempfile.TemporaryDirectory() as temp:
            run, marker, delivery = self.legacy_run(temp)
            delivery["evidence"].pop()
            marker.write_text(json.dumps(delivery))
            with self.assertRaisesRegex(ValueError, "evidence list"):
                comparison.verify_nimbus_delivery(run, "old")

    def test_incomplete_legacy_flags_and_counts_are_rejected(self):
        for key, value in [
            ("numeric_checks_passed", False),
            ("visual_review_complete", False),
            ("PanM_accessed", True),
            ("model_count_new", 1),
            ("n_cells_per_model", 1),
            ("n_units", 47),
            ("completed", "not-a-timestamp"),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                run, marker, delivery = self.legacy_run(temp)
                delivery[key] = value
                marker.write_text(json.dumps(delivery))
                with self.assertRaises(ValueError):
                    comparison.verify_nimbus_delivery(run, "old")

    def test_new_delivery_does_not_accept_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            run, marker, delivery = self.legacy_run(temp)
            with self.assertRaisesRegex(ValueError, "New Nimbus"):
                comparison.verify_nimbus_delivery(run, "new")
            marker.write_text(json.dumps({"status": "complete"}))
            self.assertEqual(comparison.verify_nimbus_delivery(run, "new"), [marker])
            with self.assertRaises(ValueError):
                comparison.verify_nimbus_delivery(run, "old")


if __name__ == "__main__":
    unittest.main(verbosity=2)
