"""Describe measured support and image-crop QC without changing any model."""
import json
import numpy as np, pandas as pd
from experiment_common import *


def main():
    d, man = load()
    records = []
    for (split, context), g in d.groupby(["split", "context"], sort=True):
        records.append(
            {
                "split": split,
                "context": context,
                "n_labelled_cells": len(g),
                "n_target_absent_in_crop": int((g.crop_target_pixels == 0).sum()),
                "n_target_partially_visible": int((g.crop_target_fraction < 1).sum()),
                "n_context_partly_unobserved": int(
                    (g.crop_context_valid_fraction < 1).sum()
                ),
                "n_geometry_input_invalid": int((~g.repair_geometry_valid).sum()),
                "n_valid_whole_no_nonzero_signal": int(
                    g.repair_whole_no_nonzero_signal.sum()
                ),
                "n_valid_reference_no_nonzero_signal": int(
                    g.repair_reference_no_nonzero_signal.sum()
                ),
                "n_reference_below_20_observed_pixels": int(
                    (g.repair_reference_n_observed < 20).sum()
                ),
                "network_unobserved_context_encoding": "zero sentinel; valid support kept separately; not measured zero",
            }
        )
    pd.DataFrame(records).to_csv(RUN / "results/CROP_SUPPORT_QC.csv", index=False)
    savej(
        RUN / "results/CROP_SUPPORT_QC_COMPLETE.json",
        {
            "rows": len(d),
            "finite_predictions_not_equivalent_to_observed_support": True,
            "features_models_and_predictions_changed": False,
        },
    )


if __name__ == "__main__":
    main()
