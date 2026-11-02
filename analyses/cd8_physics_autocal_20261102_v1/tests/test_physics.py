import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from physics_common import transform, PARAMS
import numpy as np


class PhysicsChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1, 96 * 96, dtype=np.float32).reshape(96, 96)
        self.v = np.ones_like(self.x, bool)

    def test_identity(self):
        self.assertTrue(np.array_equal(self.x, transform(self.x, self.v, 1, "clean")))

    def test_gain_monotonic_with_clipping(self):
        self.assertTrue(
            (
                transform(self.x, self.v, 1, "gain_0.25")
                <= transform(self.x, self.v, 1, "gain_2.0")
            ).all()
        )

    def test_gamma_domain_order(self):
        self.assertTrue(
            (transform(self.x, self.v, 1, "gamma_0.7") >= self.x - 1e-7).all()
        )
        self.assertTrue(
            (transform(self.x, self.v, 1, "gamma_1.3") <= self.x + 1e-7).all()
        )

    def test_clipping_is_noninvertible(self):
        self.assertTrue(
            np.array_equal(
                transform(self.x, self.v, 1, "zero_0.95", 1),
                transform(self.x * 0.3, self.v, 1, "zero_0.95", 1),
            )
        )

    def test_noise_reproducible_but_not_row_shared(self):
        a = transform(self.x, self.v, 7, "noise_high")
        self.assertTrue(np.array_equal(a, transform(self.x, self.v, 7, "noise_high")))
        self.assertFalse(np.array_equal(a, transform(self.x, self.v, 8, "noise_high")))

    def test_invalid_support_never_gets_noise(self):
        self.v[:5] = False
        a = transform(self.x, self.v, 7, "noise_high")
        self.assertTrue((a[:5] == 0).all())
        self.assertTrue(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all())

    def test_zero_fraction_with_quantile_ties(self):
        x = np.array([0, 0, 1, 1], np.float32)
        q = float(np.quantile(x, 0.7, method="higher"))
        self.assertGreaterEqual(
            (transform(x, np.ones_like(x, bool), 1, "zero_0.7", q) == 0).mean(), 0.7
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
