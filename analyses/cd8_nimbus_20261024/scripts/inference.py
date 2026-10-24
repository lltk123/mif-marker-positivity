"""Exact official UNet inference with bounded-memory tiling and full-instance aggregation."""
from common import *
import torch, gc, argparse
from nimbus_inference.nimbus import Nimbus


def reflect_indices(ix, n):
    """Map integer coordinates into NumPy-compatible reflected image support."""
    if n == 1:
        return np.zeros_like(ix)
    j = ix % (2 * (n - 1))
    return np.where(j < n, j, 2 * (n - 1) - j)


class StreamingNimbus(Nimbus):
    def __init__(self, *a, **kw):
        self.needed = None
        self.output_path = None
        self.tile_audit = []
        super().__init__(*a, **kw)

    def _tile_and_stitch(self, input_data):
        x = np.asarray(input_data)
        assert x.shape[:2] == (1, 2)
        h, w = x.shape[-2:]
        tile = np.array(self.input_shape)
        with torch.inference_mode():
            raw_shape = np.array(
                self.model(torch.zeros(1, 2, *tile, device=self.device)).shape[-2:]
            )
        crop = raw_shape % 32
        step = raw_shape - crop
        margin = (tile - step) // 2
        total = tile - (np.array([h, w]) % step)
        before = total - total // 2
        adjust = before - margin
        ny, nx = (np.array([h, w]) + total - tile) // step + 1
        out = (
            allocate(self.output_path, (1, 1, h, w), np.float64)
            if self.output_path
            else np.zeros((1, 1, h, w), np.float64)
        )
        out[:] = np.nan
        done = skipped = 0
        for iy in range(int(ny)):
            oy = int(iy * step[0] - adjust[0])
            y0 = max(0, oy)
            y1 = min(h, oy + step[0])
            ys = reflect_indices(
                np.arange(iy * step[0] - before[0], iy * step[0] - before[0] + tile[0]),
                h,
            )
            for ix in range(int(nx)):
                ox = int(ix * step[1] - adjust[1])
                x0 = max(0, ox)
                x1 = min(w, ox + step[1])
                if y0 >= y1 or x0 >= x1:
                    continue
                if self.needed is not None and not self.needed[y0:y1, x0:x1].any():
                    skipped += 1
                    continue
                xs = reflect_indices(
                    np.arange(
                        ix * step[1] - before[1], ix * step[1] - before[1] + tile[1]
                    ),
                    w,
                )
                batch = np.array(
                    x[..., ys[:, None], xs[None, :]], dtype=np.float32, copy=True
                )
                with torch.inference_mode():
                    p = (
                        self.model(torch.from_numpy(batch).to(self.device))
                        .cpu()
                        .numpy()
                    )
                p = p[
                    ...,
                    crop[0] // 2 : raw_shape[0] - crop[0] // 2,
                    crop[1] // 2 : raw_shape[1] - crop[1] // 2,
                ]
                out[..., y0:y1, x0:x1] = p[..., y0 - oy : y1 - oy, x0 - ox : x1 - ox]
                done += 1
            if iy % 20 == 0:
                print(
                    "TILES",
                    iy,
                    int(ny),
                    "computed",
                    done,
                    "skipped",
                    skipped,
                    flush=True,
                )
        if hasattr(out, "flush"):
            out.flush()
        self.tile_audit.append(
            dict(
                shape=[h, w],
                tile=tile.tolist(),
                output_step=step.tolist(),
                crop=crop.tolist(),
                computed=done,
                skipped_background_only=skipped,
            )
        )
        return out


def make_model(checkpoint="V1.pt", streaming=True):
    """Construct the pinned official Nimbus model or its equivalent streaming adapter."""
    return (StreamingNimbus if streaming else Nimbus)(
        dataset=None,
        output_dir="",
        save_predictions=False,
        batch_size=1,
        test_time_aug=True,
        input_shape=[protocol()["inference"]["tile_size"]] * 2,
        checkpoint=checkpoint,
        device="cuda",
        mixed_precision=False,
        compile_model=False,
    )


def official_tta_stream(app, x, needed, outdir):
    """Preserves actual pinned upstream late-binding behavior (4 duplicate rotations)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    acc = None
    for name, weight in [("rot270", 4), ("flip_y", 1), ("flip_x", 1)]:
        if name == "rot270":
            tx = np.rot90(x, 3, axes=(-2, -1))
            tn = np.rot90(needed, 3)
            inverse = lambda p: np.rot90(p, 1, axes=(-2, -1))
        else:
            axis = -2 if name == "flip_y" else -1
            tx = np.flip(x, axis=axis)
            tn = np.flip(needed, axis=axis)
            inverse = lambda p, axis=axis: np.flip(p, axis=axis)
        app.needed = tn
        app.output_path = None
        p = app.predict_segmentation(tx)
        if isinstance(p, torch.Tensor):
            p = p.numpy()
        back = inverse(np.squeeze(p))
        if acc is None:
            acc = np.zeros(back.shape, np.float64)
        assert back.shape == acc.shape
        for y in range(0, acc.shape[0], 256):
            acc[y : y + 256] += weight * back[y : y + 256]
        del p, back
        gc.collect()
    for y in range(0, acc.shape[0], 256):
        acc[y : y + 256] /= 6
    return acc


def aggregate(i, low_prediction):
    """Average restored pixel predictions over every original instance pixel."""
    from prepare_data import raw_unit

    a, m, valid = raw_unit(i)
    h, w = m.shape
    hs, ws = low_prediction.shape
    counts = cached(i, "counts")
    sums = np.zeros(len(counts), np.float64)
    bad = np.zeros(len(counts), np.int64)
    xx = np.floor(np.arange(w) * ws / w).astype(int)
    for y in range(0, h, 128):
        z = min(h, y + 128)
        yy = np.floor(np.arange(y, z) * hs / h).astype(int)
        v = low_prediction[yy[:, None], xx[None, :]]
        ids = np.asarray(m[y:z]).ravel().astype(np.int64)
        vv = v.ravel()
        fg = ids > 0
        sums += np.bincount(
            ids[fg], weights=np.nan_to_num(vv[fg], nan=0.0), minlength=len(sums)
        )
        bad += np.bincount(
            ids[fg], weights=(~np.isfinite(vv[fg])).astype(int), minlength=len(sums)
        ).astype(np.int64)
    found = np.flatnonzero(counts)
    found = found[found > 0]
    df = pd.DataFrame(
        dict(
            dataset_id="PhenoBIC-Data",
            image_id=i,
            cell_id=found,
            marker="CD8",
            score=sums[found] / counts[found],
            pixel_count=counts[found],
            invalid_predicted_pixel_count=bad[found],
        )
    )
    df.loc[df.invalid_predicted_pixel_count > 0, "score"] = np.nan
    assert df.score.notna().all(), "unpredicted original instance pixels"
    expected = load_cells().query("image_id==@i")[KEYS]
    assert len(expected) == len(df) and key_hash(expected) == key_hash(df)
    exact_keys(df)
    return df


def infer_one(i, app, model_name):
    """Score one complete calibration unit, preserving keys and validating cache provenance."""
    t = time.perf_counter()
    u = unit_dir(i)
    dest = R / "predictions" / model_name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / (unit_code(i) + ".parquet")
    info = path.with_suffix(".json")
    sig = dict(
        preparation_sha256=sha(u / "prepared.json"),
        model_sha256=sha(app.checkpoint_path),
        inference_code_sha256=sha(__file__),
        protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
    )
    if path.exists():
        assert (
            json.loads(info.read_text())["signature"] == sig
        ), "incompatible prediction cache"
        return
    x = np.array(cached(i, "input"), copy=True)
    need = np.array(cached(i, "needed"), copy=True)
    torch.cuda.reset_peak_memory_stats()
    app.tile_audit = []
    tmp = R / "cache/pixel_predictions" / model_name / unit_code(i)
    p = official_tta_stream(app, x, need, tmp)
    df = aggregate(i, p)
    df["model_version"] = model_name + "_" + sig["model_sha256"][:12]
    df["input_valid"] = df.pixel_count.gt(0)
    df["score_finite"] = np.isfinite(df.score)
    df.to_parquet(path, index=False)
    record = dict(
        image_id=i,
        model=model_name,
        n_cells=len(df),
        score_min=float(df.score.min()),
        score_max=float(df.score.max()),
        seconds=time.perf_counter() - t,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        process_peak_rss_gib=__import__("resource")
        .getrusage(__import__("resource").RUSAGE_SELF)
        .ru_maxrss
        / 1024**2,
        tile_audit=app.tile_audit,
        signature=sig,
    )
    dump(info, record)
    print("INFERENCE_DONE", record, flush=True)


def verify_ram_tta(app):
    marker = R / "tests/ram_tta_verified.json"
    if marker.exists():
        assert json.loads(marker.read_text())["inference_code_sha256"] == sha(__file__)
        return
    import cv2
    from nimbus_inference.utils import (
        prepare_binary_mask,
        prepare_input_data,
        test_time_aug,
    )

    z = np.load(R / "tests/pilot_raw_a07440df4b68ff270ce3.npz")
    raw = z["image"]
    mask = z["mask"]
    factor = json.loads((R / "models/normalization.json").read_text())["CD8"]
    shape = (raw.shape[1] // 2, raw.shape[0] // 2)
    inp = prepare_input_data(
        cv2.resize(np.clip(raw.astype(np.float32) / factor, 0, 1), shape),
        binary_mask=cv2.resize(prepare_binary_mask(mask), shape, interpolation=0),
    )
    ref = make_model("V1.pt", streaming=False)
    expected = test_time_aug(inp, "CD8", ref, {"CD8": factor})
    actual = official_tta_stream(
        app, inp, np.ones(inp.shape[-2:], bool), R / "tests/ram_tta"
    )
    err = float(np.max(np.abs(expected - actual)))
    assert err < 2e-6
    info = dict(
        inference_code_sha256=sha(__file__),
        max_abs_error=err,
        actual_real_crop="training WSI1 same pilot pixels",
        preserves_frozen_preprocessing=True,
    )
    dump(marker, info)
    pd.DataFrame([info]).to_csv(R / "tests/ram_tta_equivalence.csv", index=False)
    del ref, expected, actual, inp
    gc.collect()
    torch.cuda.empty_cache()
    print("RAM_TTA_EQUIVALENCE_PASSED", err, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=["Nimbus_pretrained", "Nimbus_finetuned"],
        default="Nimbus_pretrained",
    )
    ap.add_argument("--image-id")
    ap.add_argument("--split")
    a = ap.parse_args()
    import fcntl

    lock = (R / "logs/gpu_inference.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    seed_all()
    checkpoint = "V1.pt" if a.model == "Nimbus_pretrained" else "Nimbus_finetuned.pt"
    app = make_model(checkpoint)
    if a.model == "Nimbus_pretrained":
        verify_ram_tta(app)
    for i in units():
        if a.image_id and i != a.image_id:
            continue
        if a.split and not i.startswith(a.split + "/"):
            continue
        infer_one(i, app, a.model)


if __name__ == "__main__":
    main()
