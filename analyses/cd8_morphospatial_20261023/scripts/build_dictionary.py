"""Document feature formulas, units, valid conditions and missing-value causes."""

from pathlib import Path
import json, csv, sys
from mif_features import *

A = Path(__file__).resolve().parents[1]
D = {}


def add(
    name,
    formula,
    unit="dimensionless",
    condition="Valid, complete connected instance with >=5 finite pixels",
    nan="Truncated/disconnected instance, insufficient pixels, or invalid target pixels",
    role="numeric_feature",
    source="custom",
):
    D[name] = dict(
        name=name,
        role=role,
        definition=formula,
        unit=unit,
        applicable_when=condition,
        nan_reason=nan,
        implementation=source,
        used_by_full_model=name in FEATURES,
    )


md = {
    "Orientation": (
        "Major-axis orientation from row (vertical) axis, [-pi/2, pi/2]; actual skimage implementation, not the horizontal-axis wording in HTK docstring",
        "radian",
    ),
    "Area": ("Number of instance pixels", "px^2"),
    "ConvexHullArea": ("Number of pixels in binary convex hull", "px^2"),
    "MajorAxisLength": ("Major-axis length of equal-second-moment ellipse", "px"),
    "MinorAxisLength": ("Minor-axis length of equal-second-moment ellipse", "px"),
    "Perimeter": ("4-neighborhood border-pixel-center perimeter estimator", "px"),
    "Circularity": (
        "4*pi*Area/Perimeter^2; not clipped to [0,1] because digital perimeter can yield >1",
        "dimensionless",
    ),
    "Eccentricity": ("Eccentricity of equal-second-moment ellipse", "dimensionless"),
    "EquivalentDiameter": ("sqrt(4*Area/pi)", "px"),
    "Extent": ("Area / area of axis-aligned bounding box", "dimensionless"),
    "MinorMajorAxisRatio": ("MinorAxisLength / MajorAxisLength", "dimensionless"),
    "Solidity": ("Area / ConvexHullArea", "dimensionless"),
}
for n in MORPH:
    formula, unit = md.get(
        n,
        (
            "Unweighted binary-mask Hu moment "
            + n.replace("HuMoments", "")
            + " from normalized central moments; raw signed value, not log transformed",
            "dimensionless",
        ),
    )
    add(
        n,
        formula,
        unit,
        source="scikit-image 0.21.0 regionprops; HTK 1.4.0 column "
        + HTK_COLUMNS[MORPH.index(n)],
    )
for n, desc in {
    "whole_mean": "Arithmetic mean",
    "whole_median": "Median",
    "whole_std": "Population standard deviation (ddof=0)",
    "whole_p10": "10th percentile",
    "whole_p90": "90th percentile",
    "whole_p95": "95th percentile",
    "whole_IQR": "75th minus 25th percentile",
    "integrated_intensity": "Sum in float64",
}.items():
    add(
        n,
        desc
        + " of original CD8 pixels belonging to target instance only. Percentiles: numpy linear interpolation.",
        "intensity*px" if n == "integrated_intensity" else "original intensity unit",
    )
for part in ["edge", "inner"]:
    for stat in ["mean", "median", "p90"]:
        add(
            part + "_" + stat,
            stat + " of original CD8 in " + part + "; edge: EDT(M)<=w, inner: EDT(M)>w",
            "original intensity unit",
            "Complete finite instance; region >=5 valid pixels",
        )
add(
    "edge_inner_mean_diff",
    "edge_mean - inner_mean",
    "original intensity unit",
    "Both regions valid",
)
add(
    "edge_inner_normalized_diff",
    "(edge_mean-inner_mean)/(abs(edge_mean)+abs(inner_mean)+epsilon)",
    "dimensionless",
    "Both regions valid; abs(edge_mean)+abs(inner_mean)>epsilon",
    "Invalid region or both means effectively zero",
)
add(
    "local_reference_median",
    "Median of valid original CD8 in Euclidean annulus r_inner<distance_to_M<=r_outer; exclude ALL segmented IDs and supplied tissue-mask exterior",
    "original intensity unit",
    "At least 20 finite, observed, nonsegmented reference pixels",
    "Insufficient reference pixels; missing tissue mask is flagged, not inferred",
)
add(
    "local_reference_MAD",
    "Median(abs(reference_pixels-local_reference_median)); unscaled MAD",
    "original intensity unit",
    "Reference valid",
    "Reference insufficient; a true MAD of zero is retained and flagged",
)
for part in ["whole", "edge", "inner"]:
    add(
        part + "_mean_minus_reference",
        part + "_mean - local_reference_median",
        "original intensity unit",
        part + " and local reference valid",
    )
add(
    "edge_inner_log2_ratio",
    "log2((mean(S_edge)+epsilon)/(mean(S_inner)+epsilon)); S=max(CD8-reference_median,0)",
    "dimensionless",
    "Both regions and reference valid; total S > epsilon*whole_area",
    "Low total signal or invalid regions/reference",
)
add(
    "edge_signal_enrichment",
    "[sum(S_edge)/sum(S_whole)]/[edge_area/whole_area]; uniform S yields 1",
    "dimensionless",
    "Both regions and reference valid; total S > epsilon*whole_area",
    "Low total signal or invalid regions/reference",
)
for k in range(12):
    origin = (
        "Sector "
        + str(k)
        + ": ["
        + str(30 * k)
        + ","
        + str(30 * (k + 1))
        + ") degrees; 0 right, counter-clockwise in x-right/y-up coordinates about geometric centroid."
    )
    add(
        f"edge_sector_{k:02d}_mean",
        origin + " Mean original CD8.",
        "original intensity unit",
        "Complete finite instance and >=2 valid edge pixels in sector",
        "Invalid instance or sector has <2 valid pixels",
    )
    add(
        f"edge_sector_{k:02d}_signal_mean",
        origin + " Mean S=max(CD8-reference_median,0).",
        "original intensity unit",
        "Sector valid and reference valid",
        "Invalid/insufficient sector or reference",
    )
    add(
        f"edge_sector_{k:02d}_high_fraction",
        origin
        + " Fraction of valid edge pixels with CD8>T. T=reference_median+k*1.4826*MAD.",
        "fraction",
        "Sector and reference valid; MAD>epsilon",
        "MAD zero, insufficient reference or sector; NOT a cell positivity threshold",
    )
    add(
        f"edge_sector_{k:02d}_pixel_count",
        origin + " Number of finite observed edge pixels.",
        "pixels",
        "Always recorded",
        "Not NaN; zero means no valid pixels",
        role="quality_count",
    )
for pre, suffix in [
    ("edge_sector", "original per-sector CD8 means"),
    ("sector_signal", "reference-subtracted per-sector S means"),
]:
    add(
        pre + "_cv",
        "Population std / mean(abs(values)) across valid " + suffix,
        "dimensionless",
        ">=6 valid sectors and mean absolute sector signal>epsilon",
        "Too few sectors or effectively zero signal",
    )
    add(
        pre + "_max_minus_median",
        "max(values)-median(values) across valid " + suffix,
        "original intensity unit",
        ">=6 valid sectors and nonzero denominator as for CV",
        "Too few sectors or effectively zero signal",
    )
add(
    "high_signal_edge_coverage",
    "Fraction of all valid edge pixels with original CD8>T; T=reference_median+3*1.4826*MAD by default",
    "fraction",
    "Complete instance, >=5 edge pixels, valid reference and MAD>epsilon",
    "Reference insufficient, MAD zero, or invalid edge. k=3 is an engineering starting value only.",
)
add(
    "signal_centroid_offset_px",
    "Euclidean distance between geometric centroid and S-weighted centroid",
    "px",
    "Valid instance/reference; sum(S)>epsilon*area",
    "Insufficient positive reference-subtracted signal or invalid instance/reference",
)
add(
    "signal_centroid_offset_normalized",
    "signal_centroid_offset_px / sqrt(Area/pi)",
    "dimensionless",
    "Same as signal_centroid_offset_px",
    "Same as signal_centroid_offset_px",
)
for part in ["neighbor", "nonneighbor"]:
    add(
        part + "_edge_mean",
        "Mean original CD8 on edge "
        + ("within" if part == "neighbor" else "outside")
        + " configured distance to any OTHER positive mask ID",
        "original intensity unit",
        "Complete finite instance; side has >=5 pixels",
        "Side absent/too small or invalid target",
    )
add(
    "neighbor_edge_mean_diff",
    "neighbor_edge_mean - nonneighbor_edge_mean",
    "original intensity unit",
    "Both side regions valid",
    "Missing/too-small side; not proof of contamination",
)
add(
    "neighbor_edge_area_fraction",
    "area(edge within neighbor distance) / area(edge)",
    "fraction",
    "Complete finite instance with nonempty edge",
    "Invalid target; 0 is valid when no local neighbor contact",
)
add(
    "nonneighbor_edge_area_fraction",
    "1 - neighbor_edge_area_fraction",
    "fraction",
    "Same as neighbor_edge_area_fraction",
    "Same as neighbor_edge_area_fraction",
)
for n in COUNTS:
    if n not in D:
        add(
            n,
            "Count for " + n.replace("_", " "),
            "pixels" if "pixel" in n else "count",
            "Always recorded",
            "No NaN; zero is an explicit count",
            role="quality_count",
        )
for n in META_FLOAT:
    if n not in D:
        definition = "Audit/coordinate/config metadata: " + n
        if n == "observed_mean_CD8":
            definition = "Mean of finite observed target pixels, retained only for input/GMM intensity correspondence audit even if shape is truncated. NOT a classifier input."
        if n == "edge_width_px":
            definition = "Actual w=max(1,edge_fraction*max(EDT(M))) unless explicit override; default fraction 0.2"
        if n == "epsilon_intensity":
            definition = "Configured absolute epsilon=1e-8 in original intensity units; low total S threshold is epsilon*area"
        if n == "high_signal_threshold":
            definition = "Reference median + high_signal_k*1.4826*reference MAD; NOT cell positivity threshold"
        if n.endswith("_um") or n.endswith("_um2"):
            definition = "Optional isotropic physical-scale feature: corresponding pixel feature times pixel_size_um (length) or its square (area)"
        add(
            n,
            definition,
            "um^2"
            if n.endswith("_um2")
            else "um"
            if n.endswith("_um")
            else "see definition",
            "Relevant inputs and optional confirmed isotropic spacing available",
            "Optional input absent or associated computation invalid",
            role="metadata",
        )
for n in META_STR + ["cell_id"]:
    add(
        n,
        "Explicit identifier or metadata; cell_id is original integer mask value, never a table row index",
        "identifier",
        "All rows",
        "Empty string indicates unavailable patient metadata",
        role="metadata",
    )
for n in QC:
    add(
        n,
        "Boolean condition: " + n[3:].replace("_", " "),
        "boolean",
        "All rows",
        "False/True only",
        role="QC_flag",
    )
D["qc_no_signal"][
    "definition"
] = "One or more positional summaries have near-zero signal/denominator. Does not assign biological negativity."
D["qc_no_neighbors"][
    "definition"
] = "No other positive instance ID in the finite local extraction context; does not claim there are no cells elsewhere in the slide."
D["qc_reference_tissue_unknown"][
    "definition"
] = "No tissue mask supplied; reference excludes segmented IDs, invalid and unobserved pixels but tissue membership cannot be confirmed."
D["qc_truncated"][
    "definition"
] = "Instance touches source image boundary or an unobserved spatial-support boundary. Morphology, whole-instance and spatial target features are NaN; observed counts/mean remain audit metadata."
assert set(D) == set(COLUMNS), (set(COLUMNS) - set(D), set(D) - set(COLUMNS))
ordered = [D[c] for c in COLUMNS]
with (A / "FEATURE_DICTIONARY.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ordered[0]))
    w.writeheader()
    w.writerows(ordered)
(A / "FEATURE_DICTIONARY.json").write_text(json.dumps(ordered, indent=2))
(A / "provenance/mask_semantics.json").write_text(
    json.dumps(
        dict(
            mask_type="expanded_nucleus",
            applies_to="PhenoBIC in-house training/internal WSI and external TMA",
            source="https://pmc.ncbi.nlm.nih.gov/articles/PMC13308180/",
            section="Cell segmentation; Figure 2",
            summary="Authors approximate cells by expanding convex hulls of predicted nuclei, resolving overlaps by Voronoi partitioning. The released mask edge is an approximation, not a measured membrane. Original nuclear masks and tissue masks are not supplied for this run.",
            pixel_units="px; physical spacing not confirmed",
            DAPI_requirement="Not required with existing corresponding instance masks",
            fallback="HistomicsTK 1.4 requires Python>=3.9; existing validated server environment is Python3.8. scikit-image0.21 implementation validated against the released standalone function on nondegenerate masks; degenerate/truncated NaN behavior intentionally differs.",
        ),
        indent=2,
    )
)
print("DICTIONARY", len(ordered), "MODEL FEATURES", len(FEATURES), "MORPH", len(MORPH))
