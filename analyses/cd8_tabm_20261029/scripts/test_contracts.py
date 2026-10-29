#!/usr/bin/env python3
"""Small, data-free contract tests for CD8 evaluation and error review.

Run from any working directory::

    python /path/to/scripts/test_contracts.py

The image-review selector can also be checked independently::

    python /path/to/scripts/review_tabm_errors.py --self-test

These tests use synthetic cell IDs and in-memory tables. They do not read project
images, train models, create report files, or establish scientific performance.
"""
from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from evaluate_tabm import (
    KEYS,
    METRIC_COLUMNS,
    TEST,
    TRAIN,
    VALID,
    aligned_scores,
    apply_risk_strata,
    classification_metrics,
    fit_risk_strata,
    paired_comparisons,
)
from review_tabm_errors import self_test as review_self_test


class ClassificationMetricContracts(unittest.TestCase):
    """Boundary decisions and missing observations must remain explicit."""

    def test_strict_half_threshold(self):
        result = classification_metrics(
            np.array([0, 1]), np.array([0.5, np.nextafter(0.5, 1.0)]), 0.5
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 0, 0, 1))
        self.assertEqual(result["f1"], 1.0)

    def test_no_true_or_predicted_positives(self):
        result = classification_metrics(np.array([0, 0]), np.array([0.1, 0.5]), 0.5)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["specificity"], 1.0)
        for name in ("recall", "f1", "average_precision", "roc_auc"):
            self.assertTrue(np.isnan(result[name]), name)

    def test_no_true_positives_with_false_positive(self):
        result = classification_metrics(np.array([0, 0]), np.array([0.1, 0.8]), 0.5)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["fp"], 1)
        self.assertTrue(np.isnan(result["recall"]))
        self.assertTrue(np.isnan(result["average_precision"]))

    def test_no_predicted_positives_with_true_positive(self):
        result = classification_metrics(np.array([0, 1]), np.array([0.1, 0.5]), 0.5)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["fn"], 1)

    def test_failed_score_keeps_complete_denominator(self):
        result = classification_metrics(np.array([0, 1, 1]), np.array([0.1, np.nan, 0.9]), 0.5)
        self.assertEqual(result["n_cells"], 3)
        self.assertEqual(result["n_true_positive"], 2)
        self.assertEqual(result["n_score_finite"], 2)
        self.assertEqual(result["n_score_failed"], 1)
        self.assertAlmostEqual(result["prediction_coverage"], 2 / 3)
        self.assertFalse(result["complete_set_metrics_valid"])
        for name in METRIC_COLUMNS:
            self.assertTrue(np.isnan(result[name]), name)


class CellKeyContracts(unittest.TestCase):
    def setUp(self):
        self.cells = pd.DataFrame({
            "dataset_id": ["synthetic"] * 4,
            "image_id": ["wsi_a", "wsi_a", "wsi_b", "wsi_b"],
            "cell_id": [90001, 7, 7, 90001],
            "marker": ["CD8"] * 4,
        })
        self.prediction = self.cells.assign(score=[0.11, 0.22, 0.33, 0.44])

    def test_shuffled_predictions_join_by_complete_key(self):
        shuffled = self.prediction.sample(frac=1, random_state=0).reset_index(drop=True)
        aligned = aligned_scores(self.cells, shuffled)
        pd.testing.assert_frame_equal(aligned[KEYS], self.cells[KEYS])
        np.testing.assert_allclose(aligned["score"], [0.11, 0.22, 0.33, 0.44])

    def test_duplicate_prediction_key_is_rejected(self):
        duplicated = pd.concat([self.prediction, self.prediction.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            aligned_scores(self.cells, duplicated)

    def test_missing_prediction_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            aligned_scores(self.cells, self.prediction.iloc[:-1])


class RiskStratumContracts(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "split": [TRAIN] * 4,
            "ground_truth_binary": [1, 1, 0, 0],
            "whole_mean": [1.0, 3.0, 10.0, 20.0],
            "neighbor_edge_mean_diff": [1.0, 2.0, 3.0, 4.0],
            "neighbor_edge_area_fraction": [0.2, 0.0, 0.2, 0.2],
            "local_reference_median": [1.0, 3.0, 5.0, 7.0],
            "sector_signal_cv": [0.1, 0.2, 0.3, 0.4],
            "edge_sector_cv": [900.0] * 4,
        })

    def test_fit_ignores_validation_test_and_unlabelled_training(self):
        before = fit_risk_strata(self.train)
        extra = pd.concat([
            self.train.assign(split=VALID),
            self.train.assign(split=TEST),
            self.train.assign(ground_truth_binary=np.nan),
        ], ignore_index=True)
        for column in ("whole_mean", "neighbor_edge_mean_diff", "local_reference_median", "sector_signal_cv"):
            extra[column] = 1e12
        after = fit_risk_strata(pd.concat([self.train, extra], ignore_index=True))
        self.assertEqual(before, after)
        self.assertEqual(after["weak_positive_whole_mean_q25"], 1.5)
        self.assertEqual(after["contact_bright_diff_q75"], 3.5)
        self.assertEqual(after["local_reference_q75"], 5.5)
        self.assertAlmostEqual(after["sector_cv_q75"], 0.325)
        self.assertEqual(after["metadata"]["fit_counts"]["contact_bright_diff_q75"], 3)

    def test_apply_uses_correct_signal_cv_and_strict_proxy_threshold(self):
        config = {
            "weak_positive_whole_mean_q25": 1.0,
            "contact_bright_diff_q75": 3.0,
            "local_reference_q75": 5.0,
            "sector_cv_q75": 0.3,
        }
        masks = apply_risk_strata(self.train, config)
        np.testing.assert_array_equal(masks["weak_positive"], [True, False, False, False])
        np.testing.assert_array_equal(masks["contact_bright_proxy"], [False, False, False, True])
        np.testing.assert_array_equal(masks["high_local_reference_proxy"], [False, False, False, True])
        np.testing.assert_array_equal(masks["directional_signal_proxy"], [False, False, False, True])

    def test_missing_feature_does_not_create_a_proxy(self):
        partial = self.train.drop(columns="sector_signal_cv")
        config = fit_risk_strata(partial)
        self.assertIsNone(config["sector_cv_q75"])
        self.assertNotIn("directional_signal_proxy", apply_risk_strata(partial, config))


class PairedThresholdContracts(unittest.TestCase):
    def test_gmm_reference_remains_fixed_for_selected_threshold_comparison(self):
        records = []
        settings = [
            ("TabM", "A", 0, "fixed_0.5", 0.5),
            ("TabM", "A", 0, "validation_selected", 0.32),
            ("HGB", "A", 0, "fixed_0.5", 0.5),
            ("HGB", "A", 0, "validation_selected", 0.44),
            ("GMM", "GMM", -1, "fixed_0.5", 0.5),
        ]
        for family, feature_set, seed, mode, threshold in settings:
            records.append({
                "model_family": family, "feature_set": feature_set, "seed": seed,
                "threshold_mode": mode, "threshold": threshold, "split": TEST,
                "group_type": "split", "group_id": TEST, "context": "all",
                **classification_metrics(np.array([0, 1]), np.array([0.1, 0.8]), threshold),
            })
        result = paired_comparisons(pd.DataFrame(records))
        selected = result.loc[result["threshold_mode"].eq("validation_selected")]
        gmm = selected.loc[selected["comparison"].eq("TabM_A_minus_GMM")].iloc[0]
        self.assertEqual(gmm["threshold_new"], 0.32)
        self.assertEqual(gmm["threshold_old"], 0.5)
        self.assertEqual(gmm["reference_threshold_mode"], "fixed_0.5")
        hgb = selected.loc[selected["comparison"].eq("TabM_minus_HGB_A")].iloc[0]
        self.assertEqual(hgb["threshold_old"], 0.44)
        self.assertEqual(hgb["reference_threshold_mode"], "validation_selected")


class DevelopmentReviewContracts(unittest.TestCase):
    def test_review_selector_importable_self_test(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            review_self_test()
        self.assertIn("SELF_TEST_PASS", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
