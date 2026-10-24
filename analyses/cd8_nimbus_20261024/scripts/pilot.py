"""Reference comparisons on real image crops before full inference."""
from common import *
import cv2, torch, gc
from nimbus_inference.utils import (
    prepare_binary_mask,
    prepare_input_data,
    segment_mean,
    test_time_aug,
)
from inference import make_model, official_tta_stream, reflect_indices


def main():
    seed_all()
    p = protocol()
    factor = json.loads((R / "models/normalization.json").read_text())["CD8"]
    reference = make_model(streaming=False)
    adapt = make_model(streaming=True)
    rows = []
    selected = []
    es = entries()
    for split, context in [
        ("training_WSIs", "WSI1"),
        ("internal_val_WSIs", "WSI1"),
        ("external_val_TMA_cores", "TMA1_CD8"),
        ("external_val_TMA_cores", "TMA2_CD8"),
        ("external_val_TMA_cores", "TMA3_CD8"),
    ]:
        e = next(
            e
            for e in es
            if e["split"] == split and e["context"] == context and e["n_cells"] > 0
        )
        selected.append(e)
    for e in selected:
        z = np.load(OLD / "bboxes" / (e["key"] + ".npz"))
        ids = z["cell_id"]
        bb = z["bbox"][len(ids) // 2]
        cy = int((bb[0] + bb[1]) / 2)
        cx = int((bb[2] + bb[3]) / 2)
        a = read_tiff(D / e["source"]["image"])
        m = read_tiff(D / e["source"]["mask"])
        h, w = a.shape
        y = max(0, min(h - 2400, cy - 1200))
        x = max(0, min(w - 2300, cx - 1150))
        aa = np.array(a[y : y + 2400, x : x + 2300])
        mm = np.array(m[y : y + 2400, x : x + 2300])
        norm = np.clip(aa.astype(np.float32) / factor, 0, 1)
        binary = prepare_binary_mask(mm)
        sh = (int(norm.shape[1] * 0.5), int(norm.shape[0] * 0.5))
        inp = prepare_input_data(
            cv2.resize(norm, sh), binary_mask=cv2.resize(binary, sh, interpolation=0)
        )
        # Assert streaming matches full official tile grid and actual TTA on all pixels.
        expected = test_time_aug(inp, "CD8", reference, {"CD8": factor})
        needed = np.ones(inp.shape[-2:], bool)
        actual = official_tta_stream(
            adapt, inp, needed, R / "tests" / ("pilot_" + e["key"])
        )
        err = float(np.max(np.abs(expected - actual)))
        assert err < 2e-6, err
        full = cv2.resize(expected, mm.shape[::-1], interpolation=0)
        props = segment_mean(mm, full)
        f = pd.DataFrame(props)
        # Explicit ID, never row-position pairing. Verify all original positive IDs exactly once.
        assert np.array_equal(
            np.sort(np.unique(mm)[np.unique(mm) > 0]), f.label.to_numpy()
        )
        for label in f.label.to_numpy()[:: max(1, len(f) // 10)]:
            assert (
                abs(
                    float(full[mm == label].mean())
                    - float(f.loc[f.label == label, "intensity_mean"].iloc[0])
                )
                < 1e-12
            )
        row = dict(
            split=e["split"],
            context=e["context"],
            source_key=e["key"],
            crop_y=y,
            crop_x=x,
            height=aa.shape[0],
            width=aa.shape[1],
            n_instances=len(f),
            n_ids_above_uint16=int((f.label > 65535).sum()),
            max_abs_official_streaming_error=err,
            score_min=float(f.intensity_mean.min()),
            score_max=float(f.intensity_mean.max()),
        )
        rows.append(row)
        print("PILOT", row, flush=True)
        f.rename(columns={"label": "cell_id", "intensity_mean": "score"}).to_csv(
            R / "tests" / ("pilot_cells_" + e["key"] + ".csv"), index=False
        )
        np.savez_compressed(
            R / "tests" / ("pilot_raw_" + e["key"] + ".npz"),
            image=aa,
            mask=mm,
            score=full,
        )
        del a, m, aa, mm, expected, actual, full, inp
        gc.collect()
    pd.DataFrame(rows).to_csv(R / "tests/pilot_reference_comparison.csv", index=False)
    # Exact reflection mapping versus numpy, including corners.
    for n in [1, 2, 11]:
        ix = np.arange(-13, n + 13)
        assert np.array_equal(
            np.pad(np.arange(n), (13, 13), mode="reflect"),
            np.arange(n)[reflect_indices(ix, n)],
        )

    # Capture upstream actual TTA transform identity on asymmetric input.
    class Spy:
        def __init__(self):
            self.inputs = []

        def predict_segmentation(self, x):
            self.inputs.append(x.copy())
            return x[:, 0:1]

    spy = Spy()
    a = np.arange(70, dtype=np.float32).reshape(1, 2, 5, 7)
    _ = test_time_aug(a, "CD8", spy, {"CD8": 1})
    assert all(np.array_equal(x, np.rot90(a, 3, axes=(-2, -1))) for x in spy.inputs[:4])
    dump(
        R / "tests/PILOT_PASSED.json",
        dict(
            real_source_groups=5,
            official_comparison_tolerance=2e-6,
            actual_tta_rotations=[270] * 4,
            flips=["y", "x"],
            key_contract="explicit integer label IDs",
            upstream_commit=COMMIT,
        ),
    )
    print("PILOT_PASSED", flush=True)


if __name__ == "__main__":
    main()
