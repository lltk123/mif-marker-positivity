"""Descriptive source/core aggregation from completed evaluation tables only."""
from pathlib import Path
import pandas as pd, numpy as np

R = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(R / "results/ALL_EXTERNAL_SIX_METRICS.csv")
    core = d[d.level == "core"].copy()
    core["source"] = core.unit.str.split("/").str[1]
    known = set(d.loc[d.level == "source", "unit"])
    assert set(core.source) == known
    keys = ["model", "method", "threshold_mode"]
    rows = []
    for k, g in core.groupby(keys):
        assert len(g) == 42
        counts = g.groupby("source").size().to_dict()
        assert sorted(counts.values()) == [12, 12, 18]
        for source, z in g.groupby("source"):
            rows.append(
                {
                    **dict(zip(keys, k)),
                    "source": source,
                    "n_cores": len(z),
                    "n_F1_defined": int(z.F1.notna().sum()),
                    "core_F1_median": float(z.F1.median()),
                    "core_F1_IQR": float(z.F1.quantile(0.75) - z.F1.quantile(0.25)),
                    "mean_core_prevalence_MAE": float(
                        z.positive_fraction_absolute_error.mean()
                    ),
                    "mean_core_AP": float(z.AP.mean()),
                    "core_not_independent_patient": True,
                }
            )
    pd.DataFrame(rows).to_csv(R / "results/SOURCE_CORE_MACRO_SUMMARY.csv", index=False)
    print("SOURCE_CORE_MACRO_READY", len(rows))


if __name__ == "__main__":
    main()
