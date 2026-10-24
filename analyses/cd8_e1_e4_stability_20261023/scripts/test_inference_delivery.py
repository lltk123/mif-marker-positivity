"""No model fitting: verify delivered inference API against frozen cell predictions."""
import numpy as np, pandas as pd
from common import *
from inference import calibrate_instances, predict_features


def main():
    f = pd.read_parquet(
        A / "PhenoBIC_E1_E4_features.parquet",
        columns=KEYS + I8 + S9 + ["intensity_input_valid", "observed_raw_mean"],
    ).head(32)
    p = pd.read_parquet(
        A / "predictions/final_all_cells_fixed_0.5.parquet",
        columns=KEYS + ["GMM_score"] + [n + "_score" for n in SETS],
    )
    f = f.merge(p, on=KEYS, validate="one_to_one")
    checks = []
    for mode in ["fixed_0.5", "OOF_threshold"]:
        for name in SETS:
            o = predict_features(f, name, mode)
            err = float(np.max(abs(o.score - f[name + "_score"])))
            assert err < 1e-12
            checks.append(
                dict(
                    test="real_cell_inference",
                    model=name,
                    threshold_mode=mode,
                    n=len(f),
                    max_score_error=err,
                    passed=True,
                )
            )
    bad = f.head(1).copy()
    bad[I8 + S9] = np.nan
    bad["intensity_input_valid"] = False
    for name in SETS:
        o = predict_features(bad, name)
        assert not o.input_valid.any() and o.score_finite.all()
        checks.append(
            dict(
                test="invalid_input_is_separate_from_finite_output",
                model=name,
                passed=True,
            )
        )
    c = calibrate_instances([(1, np.arange(5)), (2, np.arange(10, 60, 10))])
    assert abs(c["L"] - 0.4) < 1e-12 and abs(c["U"] - 46) < 1e-12
    assert (
        calibrate_instances([])["calibration_invalid"]
        and calibrate_instances([(1, np.zeros(5))])["calibration_invalid"]
    )
    checks.append(
        dict(test="common_calibration_formula_empty_and_degenerate", passed=True)
    )
    pd.DataFrame(checks).to_csv(A / "tests/API_INTERFACE_CHECKS.csv", index=False)
    print("API CHECKS", len(checks))


if __name__ == "__main__":
    main()
