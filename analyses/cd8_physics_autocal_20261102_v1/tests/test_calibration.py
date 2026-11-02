import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np, pandas as pd
from calibration_common import intercept, shifted, logits, anchor, summary, cc
from zero_label import fit_reference, MODERATE


class CalibrationChecks(unittest.TestCase):
    def test_native_exact_zero_intercept_tie(self):
        x = np.array([0.1, 0.4828488528728485, 0.9], np.float32)
        t = float(x[1])
        self.assertTrue(np.array_equal(shifted(x, 0) > t, x > np.float32(t)))

    def test_one_class_fallback(self):
        self.assertEqual(
            intercept(np.array([-4.0, 1.0, 2.0]), np.array([0, 0, 0])), (0.0, True)
        )

    def test_regularized_intercept_score_equation(self):
        from scipy.special import expit

        z = np.array([-2.0, -0.5, 0.4, 1.0])
        y = np.array([0, 1, 1, 1])
        b, f = intercept(z, y)
        self.assertFalse(f)
        self.assertLess(abs(np.sum(expit(z + b) - y) + b), 1e-11)
        self.assertGreater(b, 0)

    def test_no_neighbor_support_fallback(self):
        q, n = anchor(np.zeros(500), np.zeros((500, 4)))
        self.assertIsNone(q)
        self.assertEqual(n, 0)

    def test_proxy_summary_finite_for_allzero(self):
        self.assertTrue(
            np.isfinite(summary(np.full(500, -20.0), np.zeros((500, 4)))).all()
        )

    def test_heldout_labels_and_features_do_not_fit(self):
        n = 400
        d = pd.DataFrame(
            {
                "split": [cc.DEV] * (3 * n),
                "image_id": np.repeat(["u1", "u2", "held"], n),
                "ground_truth_binary": np.tile(np.arange(n) % 2, 3),
            }
        )
        rng = np.random.default_rng(4)
        p = rng.uniform(0.01, 0.99, 3 * n).astype(np.float32)
        f = np.column_stack(
            [
                rng.uniform(0.1, 0.9, 3 * n),
                rng.uniform(0.1, 0.8, 3 * n),
                rng.uniform(0, 0.3, 3 * n),
                np.ones(3 * n),
            ]
        ).astype(np.float32)
        scores = {s: {"A_s0": p.copy()} for s in MODERATE}
        features = {s: f.copy() for s in MODERATE}
        a = fit_reference(d, ["u1", "u2"], "A_s0", scores, features)
        d.loc[d.image_id == "held", "ground_truth_binary"] = 1
        for s in MODERATE:
            scores[s]["A_s0"][-n:] = 0.999
            features[s][-n:] = 1000
        b = fit_reference(d, ["u1", "u2"], "A_s0", scores, features)
        self.assertTrue(np.array_equal(a["ridge"].coef_, b["ridge"].coef_))
        self.assertTrue(np.array_equal(a["scaler"].mean_, b["scaler"].mean_))
        self.assertEqual(a["trace"], b["trace"])
        self.assertEqual(a["regression_targets"], b["regression_targets"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
