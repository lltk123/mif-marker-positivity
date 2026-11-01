import sys, tempfile, json, unittest
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from execution_guards import *


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent.parent)
        self.r = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_dtype(self):
        for t in [0.001, 0.853, 0.954]:
            a = np.array([t, np.nextafter(np.float32(t), np.float32(1))], np.float32)
            self.assertTrue(
                np.array_equal(
                    a > np.asarray(t, dtype=a.dtype),
                    a.astype(float) > native_threshold(t, a.dtype),
                )
            )

    def test_strict_tie(self):
        a = np.array([0.001], np.float32)
        self.assertFalse((a > 0.001)[0])
        self.assertTrue((a.astype(float) > 0.001)[0])

    def test_atomic(self):
        p = self.r / "a.npy"
        m = self.r / "a.json"
        sig = {"keys_sha256": "a", "feature_sha256": "b"}
        save_score(p, m, np.arange(3.0), sig)
        self.assertTrue(validate_score(p, m, 3, sig)[0])
        self.assertFalse(validate_score(p, m, 4, sig)[0])
        self.assertFalse(validate_score(p, m, 3, dict(sig, keys_sha256="wrong"))[0])

    def test_corrupt(self):
        p = self.r / "a.npy"
        m = self.r / "a.json"
        save_score(p, m, np.arange(3.0), {})
        p.write_bytes(b"partial")
        self.assertFalse(validate_score(p, m, 3, {})[0])

    def test_missing_meta(self):
        p = self.r / "a.npy"
        atomic_npy(p, np.arange(3.0))
        self.assertFalse(validate_score(p, self.r / "none", 3, {})[0])

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            save_score(self.r / "a", self.r / "m", np.array([np.nan]), {})

    def test_keys(self):
        a = pd.DataFrame(
            {
                "dataset_id": ["d", "d"],
                "image_id": ["a", "b"],
                "cell_id": [1, 2],
                "marker": ["CD8", "CD8"],
            }
        )
        self.assertNotEqual(canonical_keys_hash(a), canonical_keys_hash(a.iloc[::-1]))

    def test_budget(self):
        now = [100.0]
        b = BudgetLedger(self.r / "budget.json", cap=10, clock=lambda: now[0])
        b.begin("job")
        now[0] += 4
        b.checkpoint(force=True)
        self.assertEqual(b.data["consumed_seconds"], 4)
        now[0] += 7
        with self.assertRaises(BudgetExhausted):
            b.checkpoint()
        b.abort("test")
        self.assertEqual(b.data["status"], "budget_exhausted")
        self.assertEqual(b.data["consumed_seconds"], 11)

    def test_failed_retry(self):
        now = [100.0]
        p = self.r / "b.json"
        b = BudgetLedger(p, clock=lambda: now[0])
        b.begin("job")
        now[0] += 4
        b.abort("failed")
        b2 = BudgetLedger(p, clock=lambda: now[0])
        self.assertEqual(b2.data["consumed_seconds"], 4)
        self.assertEqual(b2.data["status"], "partial")

    def test_completion(self):
        self.assertEqual(
            neural_completion_state(0, None, True), "partial_budget_exhausted"
        )
        self.assertEqual(
            neural_completion_state(0, None, False), "partial_missing_completion_marker"
        )
        self.assertEqual(
            neural_completion_state(0, {"n_models": 15}, False), "completed"
        )
        self.assertEqual(neural_completion_state(1, {"n_models": 15}, False), "failed")


if __name__ == "__main__":
    unittest.main()
