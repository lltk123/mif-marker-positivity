"""Research validate sparse tiles for the frozen PhenoBIC CD8 experiment."""

from common import *
import torch, time
from nimbus_inference.nimbus import Nimbus
from inference import StreamingNimbus


def main():
    seed_all()
    torch.set_num_threads(2)
    start = time.perf_counter()
    kw = dict(
        dataset=None,
        output_dir="",
        save_predictions=False,
        batch_size=1,
        input_shape=[512, 512],
        checkpoint="V1.pt",
        device="cpu",
        compile_model=False,
        mixed_precision=False,
    )
    ref = Nimbus(**kw)
    app = StreamingNimbus(**kw)
    x = np.random.default_rng(0).random((1, 2, 513, 667), dtype=np.float32)
    needed = np.zeros((513, 667), bool)
    needed[:12, :13] = True
    needed[240:275, 310:335] = True
    needed[-8:, -9:] = True
    expected = ref.predict_segmentation(x)
    app.needed = needed
    app.output_path = R / "tests/sparse_tiling.npy"
    actual = app.predict_segmentation(x)
    a = np.asarray(actual).squeeze()
    b = np.asarray(expected).squeeze()
    error = float(np.max(np.abs(a[needed] - b[needed])))
    assert error < 2e-6 and np.isfinite(a[needed]).all()
    assert app.tile_audit[-1]["skipped_background_only"] > 0
    pd.DataFrame(
        [
            dict(
                test="skip_unused_output_tiles_preserves_requested_pixels",
                required_pixels=int(needed.sum()),
                max_abs_error=error,
                skipped_tiles=app.tile_audit[-1]["skipped_background_only"],
                computed_tiles=app.tile_audit[-1]["computed"],
                seconds=time.perf_counter() - start,
                device="CPU",
                passed=True,
            )
        ]
    ).to_csv(R / "tests/sparse_tile_equivalence.csv", index=False)
    print("SPARSE_TILE_EQUIVALENCE_PASSED", error, flush=True)


if __name__ == "__main__":
    main()
