"""Synthetic interface tests; their scores are not biological model results."""
import unittest
import numpy as np
import pandas as pd
from baseline_contract import *


def table():
    return pd.DataFrame(
        dict(
            dataset_id=["PhenoBIC-Data"] * 3,
            image_id=["slide"] * 3,
            cell_id=[11, 37, 93],
            marker=["CD8"] * 3,
            ground_truth_binary=[0, 1, 0],
        )
    )


class ContractTests(unittest.TestCase):
    def test_reordered_ids(self):
        a = table()
        b = a[KEYS].iloc[[2, 0, 1]].copy()
        b["score"] = [0.1, 0.2, 0.9]
        c = exact_join(a, b).set_index("cell_id")
        self.assertEqual(c.loc[37, "score"], 0.9)

    def test_duplicate_rejected(self):
        a = table()
        with self.assertRaises(ValueError):
            validate_keys(pd.concat([a, a.iloc[:1]]))

    def test_missing_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_keys(table().drop(columns="cell_id"))

    def test_missing_cell_rejected(self):
        a = table()
        with self.assertRaises(ValueError):
            exact_join(a, a[KEYS].iloc[:2])

    def test_unknown_cell_rejected(self):
        a = table()
        b = a[KEYS].copy()
        b.loc[0, "cell_id"] = 99
        with self.assertRaises(ValueError):
            exact_join(a, b)

    def test_null_or_background_id_rejected(self):
        a = table()
        a.loc[0, "cell_id"] = 0
        with self.assertRaises(ValueError):
            validate_keys(a)
        a = table()
        a.loc[0, "image_id"] = None
        with self.assertRaises(ValueError):
            validate_keys(a)

    def test_key_hash_order_invariant(self):
        self.assertEqual(key_hash(table()), key_hash(table().iloc[::-1]))

    def test_missing_score_not_dropped(self):
        m = binary_metrics([0, 1, 0], [0.1, np.nan, 0.2], 0.5)
        self.assertEqual(m["n_total"], 3)
        self.assertEqual(m["n_failed"], 1)
        self.assertFalse(m["complete_population_metrics"])
        self.assertTrue(np.isnan(m["F1"]))

    def test_strict_and_inclusive_threshold(self):
        self.assertEqual(predictions_from_scores([0.5], 0.5, "gt")[0], 0)
        self.assertEqual(predictions_from_scores([0.5], 0.5, "ge")[0], 1)

    def test_no_positive_slice(self):
        m = binary_metrics([0, 0], [0.1, 0.2], 0.5)
        self.assertEqual(m["F1"], 0)
        self.assertEqual(m["recall"], 0)
        self.assertTrue(np.isnan(m["AP"]))
        self.assertTrue(np.isnan(m["ROC_AUC"]))

    def test_ambiguous_label_not_negative(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, 2, 3], [0.1, 0.2, 0.3], 0.5)

    def test_known_metrics(self):
        m = binary_metrics([0, 1, 1, 0], [0.8, 0.9, 0.1, 0.2], 0.5)
        self.assertEqual([m[x] for x in ["TP", "FP", "FN", "TN"]], [1, 1, 1, 1])
        self.assertEqual(m["F1"], 0.5)

    def test_metadata_and_gmm_forbidden(self):
        for x in ["patient_id", "image_id", "GMM_score", "ground_truth_binary"]:
            with self.assertRaises(ValueError):
                validate_feature_names([x], [x])
        validate_feature_names(["whole_mean"], ["whole_mean"])

    def test_model_spec_required(self):
        from evaluate_predictions import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"status": "awaiting_paper"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
