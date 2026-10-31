"""Full-image adapter. Global resizing, overlap union and unchanged integer IDs."""
from common import *
import cv2, gc, argparse
from scipy import ndimage
from nimbus_inference.utils import prepare_binary_mask
from normalization import (
    bounds_for,
    calibration,
    calibration_signature,
    normalize_pixels,
)


def raw_unit(i):
    """Reconstruct a complete image unit with original intensity, IDs and valid support."""
    es = units()[i]
    out = unit_dir(i)
    out.mkdir(parents=True, exist_ok=True)
    if len(es) == 1:
        e = es[0]
        a = read_tiff(D / e["source"]["image"])
        m = read_tiff(D / e["source"]["mask"])
        assert a.shape == m.shape == tuple(e["shape"])
        return a, m, None
    # Same overlap consistency and nonzero-ID union as existing io_adapter.roi.
    shape = (
        max(e["offset_y"] + e["shape"][0] for e in es),
        max(e["offset_x"] + e["shape"][1] for e in es),
    )
    ready = out / "mosaic.json"
    if ready.exists():
        assert json.loads(ready.read_text())["input_provenance"] == verify_inputs()
        return cached(i, "raw"), cached(i, "mask"), cached(i, "valid")
    a = allocate(out / "raw.npy", shape, np.float32)
    m = allocate(out / "mask.npy", shape, np.uint32)
    v = allocate(out / "valid.npy", shape, bool)
    overlap_pixels = 0
    for e in es:
        yy, xx = e["offset_y"], e["offset_x"]
        h, w = e["shape"]
        s = np.s_[yy : yy + h, xx : xx + w]
        q = read_tiff(D / e["source"]["image"])
        qm = read_tiff(D / e["source"]["mask"])
        seen = v[s]
        if seen.any():
            assert np.array_equal(a[s][seen], q[seen], equal_nan=True)
            assert not ((m[s] > 0) & (qm > 0) & (m[s] != qm)).any()
            overlap_pixels += int(seen.sum())
        a[s] = q
        m[s] = np.where(qm > 0, qm, m[s])
        v[s] = np.isfinite(q)
        del q, qm
    a.flush()
    m.flush()
    v.flush()
    dump(
        ready,
        dict(
            shape=shape,
            overlap_pixels_deduplicated=overlap_pixels,
            input_provenance=verify_inputs(),
        ),
    )
    return a, m, v


def fit_normalization():
    """Load the shared full-unit calibration; this adapter never fits a crop scale."""
    table, signature, manifest = calibration()
    state = dict(
        method_id="E1_outer_P1_P99",
        shared_calibration=signature,
        input_provenance=verify_inputs(),
        n_units=len(table),
    )
    path = R / "models/normalization.json"
    if path.exists():
        assert json.loads(path.read_text()) == state, "shared calibration changed"
    else:
        dump(path, state)
    table.to_csv(R / "results/normalization_parameters.csv", index=False)
    return table


def prepare_unit(i, calibration_table=None):
    """Use full-unit shared L/U, then preserve the original official resizing path."""
    bound = bounds_for(i)
    lower, upper = bound["L"], bound["U"]
    out = unit_dir(i)
    out.mkdir(parents=True, exist_ok=True)
    sig = {
        "input_provenance": verify_inputs(),
        "normalization_sha256": sha(R / "models/normalization.json"),
        "protocol_sha256": sha(R / "NIMBUS_PROTOCOL.json"),
        "preparation_code_sha256": sha(__file__),
        "normalization_code_sha256": sha(R / "scripts/normalization.py"),
        "shared_calibration": calibration_signature(),
    }
    done = out / "prepared.json"
    if done.exists():
        q = json.loads(done.read_text())
        assert q["signature"] == sig, "incompatible preparation cache"
        return q
    start = time.perf_counter()
    if bool(bound["calibration_invalid"]):
        q = dict(
            image_id=i,
            calibration_invalid=True,
            L=lower if np.isfinite(lower) else None,
            U=upper if np.isfinite(upper) else None,
            n_cells=int(bound["n_instances"]),
            signature=sig,
        )
        dump(done, q)
        return q
    a, m, valid = raw_unit(i)
    h, w = a.shape
    hs, ws = int(h * 0.5), int(w * 0.5)
    assert m.dtype.kind in "iu" and int(m.max()) < 2**24
    norm = allocate(out / "normalized_full.npy", (h, w), np.float32)
    binary = allocate(out / "binary_full.npy", (h, w), np.uint8)
    support = allocate(out / "observed_full.npy", (h, w), bool)
    counts = np.zeros(int(m.max()) + 1, np.int64)
    n_finite = n_positive = n_clipped = n_clipped_lower = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    for y in range(0, h, 256):
        z = min(h, y + 256)
        aa = np.asarray(a[y:z], dtype=np.float64)
        vv = np.isfinite(aa) if valid is None else np.isfinite(aa) & valid[y:z]
        if np.any((m[y:z] > 0) & ~vv):
            raise ValueError("instance contains invalid raw intensity")
        norm[y:z] = normalize_pixels(aa, lower, upper, vv).astype(np.float32)
        support[y:z] = vv
        n_finite += int(vv.sum())
        n_positive += int((aa[vv] > 0).sum())
        n_clipped += int((aa[vv] > upper).sum())
        n_clipped_lower += int((aa[vv] < lower).sum())
        if vv.any():
            raw_min = min(raw_min, float(aa[vv].min()))
            raw_max = max(raw_max, float(aa[vv].max()))
        y0 = max(0, y - 1)
        y1 = min(h, z + 1)
        binary[y:z] = prepare_binary_mask(m[y0:y1])[y - y0 : z - y0]
        counts += np.bincount(m[y:z].ravel().astype(np.int64), minlength=len(counts))
    norm.flush()
    binary.flush()
    support.flush()
    x = allocate(out / "input.npy", (1, 2, hs, ws), np.float32)
    # Missing FOV support remains NaN in normalized_full and false in observed_full.
    # The CNN needs finite tensors: use a documented zero sentinel only outside observed
    # support; no original instance may occupy unsupported pixels (checked above).
    network_signal = np.nan_to_num(norm, nan=0.0)
    cv2.resize(network_signal, (ws, hs), dst=x[0, 0], interpolation=cv2.INTER_LINEAR)
    del network_signal
    cv2.resize(
        binary.astype(np.float32),
        (ws, hs),
        dst=x[0, 1],
        interpolation=cv2.INTER_NEAREST,
    )
    ids = allocate(out / "ids_scaled.npy", (hs, ws), np.uint32)
    yy = np.floor(np.arange(hs) * h / hs).astype(int)
    xx = np.floor(np.arange(ws) * w / ws).astype(int)
    for y in range(0, hs, 256):
        ids[y : y + 256] = m[yy[y : y + 256, None], xx[None, :]]
    needed = allocate(out / "needed.npy", (hs, ws), bool)
    # Every raw foreground pixel contributes to the exact nearest-upsampled score index.
    # reduceat uses the first raw column mapping to each low-resolution column.
    startx = np.ceil(np.arange(ws) * w / ws).astype(int)
    for y in range(0, h, 128):
        z = min(h, y + 128)
        p = np.logical_or.reduceat(m[y:z] > 0, startx, axis=1)
        iy = np.floor(np.arange(y, z) * hs / h).astype(int)
        np.logical_or.at(needed, iy, p)
    x.flush()
    ids.flush()
    needed.flush()
    actual = np.flatnonzero(counts)
    actual = actual[actual > 0]
    cells = load_cells()
    expected = cells.loc[cells.image_id == i, "cell_id"].to_numpy()
    assert np.array_equal(np.sort(expected), actual), f"ID mismatch {i}"
    np.save(out / "counts.npy", counts)
    q = dict(
        image_id=i,
        shape=[h, w],
        scaled_shape=[hs, ws],
        n_cells=len(actual),
        max_id=int(m.max()),
        normalization_method="E1_outer_P1_P99",
        L=lower,
        U=upper,
        calibration_invalid=False,
        raw_min=raw_min,
        raw_max=raw_max,
        n_finite_pixels=n_finite,
        n_positive_pixels=n_positive,
        n_pixels_above_U=n_clipped,
        n_pixels_below_L=n_clipped_lower,
        fraction_clipped_to_zero=n_clipped_lower / n_finite,
        fraction_clipped_to_one=n_clipped / n_finite,
        missing_support_pixels=int((~support).sum()),
        seconds=time.perf_counter() - start,
        rss_gib=rss_gib(),
        signature=sig,
    )
    dump(done, q)
    print("PREPARED", q, flush=True)
    del a, m, norm, binary, support, x, ids, needed
    gc.collect()
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id")
    args = ap.parse_args()
    calibration_table = fit_normalization()
    selected = [args.image_id] if args.image_id else list(units())
    rows = [prepare_unit(i, calibration_table) for i in selected]
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "signature"} for r in rows]
    ).to_csv(R / "results/preparation.csv", index=False)


if __name__ == "__main__":
    main()
