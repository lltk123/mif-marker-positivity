"""Measurement descriptors for sparse mIF pixels; never infer biological positivity.
Region arrays must contain observed finite pixels only. Caller supplies geometry QC.
No learned threshold, MAD division, pseudocount or epsilon creates signal.
"""
import numpy as np

REGIONS = ("whole", "edge", "inner", "reference")
MIN_SUPPORT = {"whole": 5, "edge": 5, "inner": 5, "reference": 20}
NAMES = tuple(
    ["nonzero_rate_" + r for r in REGIONS]
    + ["positive_mean_" + r for r in REGIONS]
    + [
        "nonzero_rate_edge_minus_reference",
        "nonzero_rate_edge_minus_inner",
        "positive_mean_edge_minus_reference",
        "positive_mean_edge_minus_inner",
        "ordinal_edge_vs_reference",
    ]
)


def ordinal_contrast(edge, reference):
    e = np.asarray(edge, dtype=np.float64)
    r = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(e) or not len(r):
        return np.nan
    less = np.searchsorted(r, e, side="left")
    leq = np.searchsorted(r, e, side="right")
    return float(
        np.sum(less + 0.5 * (leq - less), dtype=np.float64) / (len(e) * len(r)) - 0.5
    )


def describe_regions(regions, geometry_valid=True):
    values = {name: np.nan for name in NAMES}
    state = {}
    arrays = {}
    for region in REGIONS:
        a = np.asarray(regions.get(region, []), dtype=np.float64).reshape(-1)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite/unobserved support must be excluded explicitly")
        arrays[region] = a
        ok = bool(geometry_valid and len(a) >= MIN_SUPPORT[region])
        positive = a[a > 0]
        state[region] = {
            "n_observed": len(a),
            "n_strictly_positive": len(positive),
            "support_valid": ok,
            "no_nonzero_signal": bool(ok and len(positive) == 0),
        }
        if ok:
            values["nonzero_rate_" + region] = float(len(positive) / len(a))
            if len(positive):
                values["positive_mean_" + region] = float(positive.mean())
    for other in ["reference", "inner"]:
        for kind in ["nonzero_rate", "positive_mean"]:
            a = values[kind + "_edge"]
            b = values[kind + "_" + other]
            if np.isfinite(a) and np.isfinite(b):
                values[kind + "_edge_minus_" + other] = a - b
    if state["edge"]["support_valid"] and state["reference"]["support_valid"]:
        values["ordinal_edge_vs_reference"] = ordinal_contrast(
            arrays["edge"], arrays["reference"]
        )
    state["geometry_valid"] = bool(geometry_valid)
    return values, state
