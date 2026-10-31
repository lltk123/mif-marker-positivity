"""Full-key coverage and model-seam audit, independent of labels and scores."""
from common import *
from normalization import bounds_for


def raw_seam_indices(low_n, raw_n, lines):
    out = set()
    for x in lines:
        if x <= 0 or x >= low_n:
            continue
        boundary = int(np.ceil(x * raw_n / low_n))
        for t in range(max(0, boundary - 2), min(raw_n, boundary + 3)):
            out.add(t)
    return sorted(out)


def lines(n):
    step = 832
    tile = 1024
    margin = 96
    total = tile - n % step
    before = total - total // 2
    adjust = before - margin
    return [j * step - adjust for j in range(1, (n + total - tile) // step + 1)]


def near_seam_ids(image):
    from prepare_data import raw_unit

    a, m, v = raw_unit(image)
    h, w = m.shape
    hs, ws = h // 2, w // 2
    # All three actual unique TTA views: 270deg, vertical, horizontal.
    yl = set(lines(hs)) | {hs - t for t in lines(hs)}
    xl = set(lines(ws)) | {ws - t for t in lines(ws)}
    ys = raw_seam_indices(hs, h, yl)
    xs = raw_seam_indices(ws, w, xl)
    vals = []
    if ys:
        vals.append(np.unique(m[np.asarray(ys)]))
    # Slice columns in row chunks to bound temporary memory.
    if xs:
        for y in range(0, h, 512):
            vals.append(np.unique(m[y : y + 512, np.asarray(xs)]))
    ids = np.unique(np.concatenate(vals)) if vals else np.array([], np.uint32)
    return ids[ids > 0]


def main():
    cells = load_cells()
    rows = []
    seamrows = []
    countrows = []
    for i in units():
        c = cells.loc[cells.image_id == i]
        invalid = bool(bounds_for(i)["calibration_invalid"])
        sid = near_seam_ids(i) if not invalid else np.array([], dtype=int)
        counts = cached(i, "counts") if not invalid else None
        if "pixel_count" in c and not invalid:
            diff = counts[c.cell_id.to_numpy(int)] - c.pixel_count.to_numpy()
            countrows.append(
                dict(
                    image_id=i,
                    n_cells=len(c),
                    n_pixel_count_differences=int((diff != 0).sum()),
                    max_absolute_count_difference=float(np.max(np.abs(diff))),
                )
            )
        for model in ["Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned"]:
            f = R / "predictions" / model / (unit_code(i) + ".parquet")
            d = pd.read_parquet(f)
            exact_keys(d)
            assert key_hash(d) == key_hash(c)
            if invalid:
                assert d.score.isna().all() and not d.input_valid.any()
            else:
                assert d.score.between(0, 1).all()
                assert d.invalid_predicted_pixel_count.eq(0).all()
            rows.append(
                dict(
                    image_id=i,
                    model=model,
                    n_expected=len(c),
                    n_predictions=len(d),
                    duplicate_keys=int(d.duplicated(KEYS).sum()),
                    nonfinite_scores=int((~np.isfinite(d.score)).sum()),
                    unpredicted_original_pixels=int(
                        d.invalid_predicted_pixel_count.sum() if not invalid else 0
                    ),
                    key_sha256=key_hash(d),
                )
            )
            at = d.loc[d.cell_id.isin(sid)]
            seamrows.append(
                dict(
                    image_id=i,
                    model=model,
                    n_near_seam_cells=len(at),
                    n_finite_scores=int(np.isfinite(at.score).sum()),
                    n_missing_seam_IDs=int(
                        len(set(map(int, sid)) - set(map(int, at.cell_id)))
                    ),
                    band_definition="within2raw pixels of mapped output grid boundaries across actual TTA views",
                )
            )
    pd.DataFrame(rows).to_csv(R / "results/prediction_integrity.csv", index=False)
    pd.DataFrame(seamrows).to_csv(R / "results/seam_cell_audit.csv", index=False)
    pd.DataFrame(countrows).to_csv(
        R / "results/mask_pixel_count_comparison.csv", index=False
    )
    assert pd.DataFrame(seamrows).n_missing_seam_IDs.eq(0).all()
    dump(
        R / "provenance/prediction_integrity_passed.json",
        dict(
            total_cells_per_model=len(cells),
            models=2,
            units=len(units()),
            key_sha256=key_hash(cells),
            all_finite=bool(pd.DataFrame(rows).nonfinite_scores.eq(0).all()),
            total_unscorable=sum(r["nonfinite_scores"] for r in rows),
        ),
    )
    print("PREDICTION_INTEGRITY_PASSED", flush=True)


if __name__ == "__main__":
    main()
