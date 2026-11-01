import sys, pathlib, unittest, numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from repaired_features import describe_regions, ordinal_contrast, NAMES


class DescriptorChecks(unittest.TestCase):
    def regions(self, w=0, e=0, i=0, r=0):
        return {
            "whole": np.full(100, w, dtype=float),
            "edge": np.full(30, e, dtype=float),
            "inner": np.full(70, i, dtype=float),
            "reference": np.full(25, r, dtype=float),
        }

    def test_zero_signal_is_observed_zero_but_conditional_mean_missing(self):
        x, s = describe_regions(self.regions())
        self.assertEqual(x["nonzero_rate_whole"], 0)
        self.assertTrue(np.isnan(x["positive_mean_whole"]))
        self.assertEqual(x["ordinal_edge_vs_reference"], 0)
        self.assertTrue(s["whole"]["no_nonzero_signal"])

    def test_constant_positive_background_has_no_ordinal_contrast(self):
        x, s = describe_regions(self.regions(0.4, 0.4, 0.4, 0.4))
        self.assertEqual(x["ordinal_edge_vs_reference"], 0)
        self.assertEqual(x["nonzero_rate_edge_minus_reference"], 0)
        self.assertAlmostEqual(x["positive_mean_edge_minus_reference"], 0)

    def test_weak_positive_has_measurement_support_without_MAD(self):
        x, s = describe_regions(self.regions(0.02, 0.02, 0.02, 0))
        self.assertEqual(x["ordinal_edge_vs_reference"], 0.5)
        self.assertAlmostEqual(x["positive_mean_edge"], 0.02)
        self.assertTrue(np.isnan(x["positive_mean_edge_minus_reference"]))

    def test_single_side_bright_fraction_remains_fraction(self):
        q = self.regions()
        q["edge"][:6] = 0.8
        x, s = describe_regions(q)
        self.assertAlmostEqual(x["nonzero_rate_edge"], 0.2)
        self.assertAlmostEqual(x["positive_mean_edge"], 0.8)
        self.assertAlmostEqual(x["ordinal_edge_vs_reference"], 0.1)

    def test_missing_support_does_not_become_zero(self):
        q = self.regions()
        q["reference"] = np.zeros(19)
        x, s = describe_regions(q)
        self.assertTrue(np.isnan(x["ordinal_edge_vs_reference"]))
        self.assertTrue(np.isnan(x["nonzero_rate_reference"]))
        self.assertFalse(s["reference"]["support_valid"])

    def test_invalid_geometry_suppresses_descriptors(self):
        x, s = describe_regions(self.regions(1, 1, 1, 1), False)
        self.assertTrue(all(np.isnan(v) for v in x.values()))

    def test_ordinal_ties_match_pairwise_definition(self):
        e = np.array([0, 0.1, 0.1, 0.4])
        r = np.array([0, 0, 0.1, 0.2, 0.3])
        brute = (e[:, None] > r).mean() + 0.5 * (e[:, None] == r).mean() - 0.5
        self.assertAlmostEqual(ordinal_contrast(e, r), brute)

    def test_positive_affine_map_preserves_order(self):
        e = np.array([0, 0.1, 0.4])
        r = np.array([0, 0.2, 0.3])
        self.assertAlmostEqual(
            ordinal_contrast(e, r), ordinal_contrast(2 * e + 7, 2 * r + 7)
        )

    def test_fixed_dimension(self):
        x, _ = describe_regions(self.regions())
        self.assertEqual(len(x), 13)
        self.assertEqual(tuple(x), NAMES)

    def test_nonfinite_support_rejected(self):
        q = self.regions()
        q["whole"][0] = np.nan
        with self.assertRaises(ValueError):
            describe_regions(q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
