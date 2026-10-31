"""Synthetic contracts for the calibrated TabM adapter; no GPU or real data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calibrated_inputs import (
    METHOD,
    RAW_RISK_FEATURES,
    assemble_cells,
    attach_bounds,
    audit_risk_frame,
    frozen_raw_risk,
    verify_source_contract,
)
from common_tabm import KEYS, TRAIN, VALID, TEST, sha256, key_hash
from evaluate_tabm import apply_risk_strata, fit_risk_strata
from review_tabm_errors import category_masks


def fixtures():
    whole = [
        "whole_mean",
        "whole_median",
        "whole_std",
        "whole_p10",
        "whole_p90",
        "whole_p95",
        "whole_IQR",
        "whole_integrated_intensity",
    ]
    morph = ["morph_%02d" % i for i in range(19)]
    extra = [x for x in RAW_RISK_FEATURES if x not in whole] + [
        "spatial_%02d" % i for i in range(58)
    ]
    requested = {"A": whole, "B": whole + morph, "C": whole + morph + extra}
    ids = [101, 7, 900, 44, 203, 19]
    splits = [TRAIN] * 4 + [VALID, TEST]
    raw = pd.DataFrame(
        {
            "dataset_id": ["PhenoBIC-Data"] * 6,
            "image_id": [
                TRAIN + "/W1",
                TRAIN + "/W1",
                TRAIN + "/W2",
                TRAIN + "/W3",
                VALID + "/W4",
                TEST + "/TMA1_CD8/C1",
            ],
            "cell_id": ids,
            "split": splits,
            "context": ["WSI"] * 5 + ["TMA1_CD8"],
        }
    )
    for j, name in enumerate(requested["C"]):
        raw[name] = np.arange(6, dtype=float) + j + 10
    raw["whole_mean"] = [100.0, 20.0, 300.0, 50.0, 900.0, 700.0]
    raw.loc[1, morph[0]] = np.nan
    raw["neighbor_edge_area_fraction"] = [0.1, 0.2, 0, 0.4, 0.2, 0.8]
    f = raw.copy()
    f["marker"] = "CD8"
    for name in whole + extra:
        f[name] = np.arange(6) * 0.1
    f["mask_type"] = "expanded_nucleus"
    f["source_image_id"] = f.image_id
    f["observed_pixel_count"] = [11, 12, 13, 14, 15, 16]
    f["observed_mean_CD8"] = [0.2, 0.4, 0.1, 0.5, 0.8, np.nan]
    f["raw_observed_mean_CD8"] = raw.whole_mean
    f["calibration_valid"] = [True] * 5 + [False]
    f.loc[5, whole + extra] = np.nan
    ann = raw[KEYS[:3]].copy()
    ann["ground_truth_original"] = ["pos", "neg", "pos", "neg", "pos", "neg"]
    ann["ground_truth_binary"] = [1, 0, 1, 0, 1, 0]
    ann["GMM_score"] = [0.6, 0.1, 0.8, 0.2, 0.7, 0.3]
    ann["GMM_prediction"] = [1, 0, 1, 0, 1, 0]
    ann["pixel_count"] = f.observed_pixel_count
    ann["mean_CD8"] = raw.whole_mean
    return f, raw, ann, requested


class CalibratedInputContracts(unittest.TestCase):
    def setUp(self):
        self.f, self.raw, self.ann, self.requested = fixtures()

    def assemble(self, f=None, raw=None, ann=None):
        return assemble_cells(
            self.f if f is None else f,
            self.raw if raw is None else raw,
            self.ann if ann is None else ann,
            self.requested,
        )

    def test_shuffled_explicit_ids_and_normalized_vs_raw_means(self):
        result = self.assemble(
            self.f.sample(frac=1, random_state=4),
            self.raw.sample(frac=1, random_state=2),
            self.ann.sample(frac=1, random_state=8),
        )
        self.assertEqual(len(result), 6)
        keyed = result.set_index("cell_id")
        self.assertEqual(keyed.loc[101, "ground_truth_binary"], 1)
        self.assertEqual(keyed.loc[7, "ground_truth_binary"], 0)
        self.assertEqual(keyed.loc[101, "observed_mean_CD8"], 0.2)
        self.assertEqual(keyed.loc[101, "raw_observed_mean_CD8"], 100.0)
        self.assertEqual(keyed.loc[101, "audit_raw_whole_mean"], 100.0)
        self.assertTrue(result.normalization_method.eq(METHOD).all())

    def test_wrong_raw_mean_rejected(self):
        self.f.loc[0, "raw_observed_mean_CD8"] = 0.2
        with self.assertRaisesRegex(ValueError, "Raw instance"):
            self.assemble()

    def test_missing_raw_audit_mean_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw_observed_mean_CD8"):
            self.assemble(self.f.drop(columns="raw_observed_mean_CD8"))

    def test_changed_key_rejected_not_aligned_by_row(self):
        self.ann.loc[0, "cell_id"] = 99999
        with self.assertRaisesRegex(ValueError, "annotation keys"):
            self.assemble()

    def test_duplicate_key_rejected(self):
        duplicate = pd.concat([self.f, self.f.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            self.assemble(duplicate)

    def test_fractional_id_rejected(self):
        self.f["cell_id"] = self.f.cell_id.astype(float)
        self.f.loc[0, "cell_id"] = 101.5
        with self.assertRaisesRegex(ValueError, "integer instance ID"):
            self.assemble()

    def test_changed_geometry_rejected(self):
        self.f.loc[0, "morph_00"] += 1
        with self.assertRaises(AssertionError):
            self.assemble()

    def test_feature_input_cannot_contain_truth(self):
        self.f["ground_truth_binary"] = self.ann.ground_truth_binary
        with self.assertRaisesRegex(ValueError, "label- and GMM-free"):
            self.assemble()

    def test_failed_calibration_kept_even_with_missing_intensities(self):
        result = self.assemble()
        invalid = result.loc[~result.calibration_valid]
        self.assertEqual(len(result), 6)
        self.assertEqual(invalid.cell_id.tolist(), [19])
        self.assertTrue(invalid[self.requested["A"]].isna().all().all())
        bounds = result[["image_id", "calibration_valid"]].drop_duplicates().copy()
        bounds["calibration_invalid"] = ~bounds.calibration_valid
        bounds["L"], bounds["U"], bounds["n_eligible"] = 1.0, 1000.0, 5
        result = attach_bounds(result, bounds)
        self.assertEqual(len(result), 6)

    def test_wrong_calibration_unit_rejected(self):
        result = self.assemble()
        bounds = pd.DataFrame(
            {
                "image_id": ["other"],
                "calibration_invalid": [False],
                "L": [0.0],
                "U": [1.0],
                "n_eligible": [5],
            }
        )
        with self.assertRaisesRegex(ValueError, "unit keys"):
            attach_bounds(result, bounds)

    def test_raw_risk_cohorts_preserved_independent_of_model_values(self):
        data = self.assemble()
        original = audit_risk_frame(
            data, {"feature_columns": {n: "audit_raw_" + n for n in RAW_RISK_FEATURES}}
        )
        rules = fit_risk_strata(original)
        rules.update(
            fit_split=TRAIN, weak_comparison="<=", contact_comparison="strict >"
        )
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "old_risk.json"
            p.write_text(json.dumps(rules), encoding="utf-8")
            reused = frozen_raw_risk(data, rules, p)
            expected = apply_risk_strata(original, rules)
            actual = apply_risk_strata(data, reused)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            before, _ = category_masks(original, rules)
            after, _ = category_masks(data, reused)
            for key in before:
                np.testing.assert_array_equal(after[key], before[key])
            # Target observations cannot influence verified training cutoffs.
            changed = data.copy()
            changed.loc[changed.split.ne(TRAIN), "audit_raw_whole_mean"] = 1e15
            verified = frozen_raw_risk(changed, rules, p)
            self.assertEqual(
                verified["weak_positive_whole_mean_q25"],
                reused["weak_positive_whole_mean_q25"],
            )
            self.assertFalse(
                any(name.startswith("audit_raw_") for name in self.requested["C"])
            )

    def test_missing_raw_risk_field_never_falls_back_to_normalized(self):
        data = self.assemble()
        config = {"feature_columns": {"whole_mean": "audit_raw_whole_mean"}}
        with self.assertRaisesRegex(ValueError, "Missing original raw"):
            audit_risk_frame(data.drop(columns="audit_raw_whole_mean"), config)

    def test_architecture_and_training_budget_unchanged(self):
        import ast
        from common_tabm import COMMIT

        here = Path(__file__).resolve()
        old = here.parents[2] / "cd8_tabm_20261029/scripts/train_tabm.py"

        def config_at(file):
            module = ast.parse(file.read_text(encoding="utf-8-sig"))
            function = next(
                n
                for n in module.body
                if isinstance(n, ast.FunctionDef) and n.name == "configuration"
            )
            namespace = dict(
                COMMIT=COMMIT, TRAIN=TRAIN, VALID=VALID, TEST=TEST, METHOD=METHOD
            )
            exec(
                compile(
                    ast.Module(body=[function], type_ignores=[]), str(file), "exec"
                ),
                namespace,
            )
            return namespace["configuration"]()

        previous, current = config_at(old), config_at(here.parent / "train_tabm.py")
        for key in ["features", "normalization", "normalization_method"]:
            previous.pop(key, None)
            current.pop(key, None)
        self.assertEqual(previous, current)

    def test_shared_manifest_rejects_changed_calibration_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = {
                name: base / filename
                for name, filename in {
                    "features": "features.parquet",
                    "manifest": "extraction_manifest.json",
                    "protocol": "PROTOCOL.json",
                    "bounds": "unit_bounds.csv",
                    "raw_features": "raw.parquet",
                    "labels": "labels.parquet",
                    "risk": "risk.json",
                    "dictionary": "dictionary.csv",
                    "feature_code": "features.py",
                    "experiment_protocol": "RETRAIN_PROTOCOL.json",
                }.items()
            }
            for name, file in paths.items():
                if name != "manifest":
                    file.write_text("synthetic test fixture", encoding="utf-8")
            pd.DataFrame(
                {
                    "image_id": ["unit_" + str(i) for i in range(48)],
                    "L": np.zeros(48),
                    "U": np.ones(48),
                    "calibration_invalid": np.zeros(48, bool),
                    "n_eligible": np.full(48, 5),
                }
            ).to_csv(paths["bounds"], index=False)
            manifest = dict(
                status="complete",
                method_id=METHOD,
                rows=1817783,
                feature_file=paths["features"].name,
                calibration_file="calibration/unit_bounds.csv",
                feature_columns=self.requested["C"],
                all_keys_sha256="synthetic_no_real_data",
            )
            fields = {
                "feature_sha256": "features",
                "protocol_sha256": "protocol",
                "calibration_sha256": "bounds",
                "source_feature_sha256": "raw_features",
                "dictionary_sha256": "dictionary",
            }
            manifest.update(
                {field: sha256(paths[name]) for field, name in fields.items()}
            )
            paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
            _, bounds, signature = verify_source_contract(paths, self.requested)
            self.assertEqual(len(bounds), 48)
            self.assertEqual(signature["method_id"], METHOD)
            paths["protocol"].write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol_sha256"):
                verify_source_contract(paths, self.requested)


if __name__ == "__main__":
    unittest.main(verbosity=2)
