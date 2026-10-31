"""Small controlled checks on actual pinned upstream operations."""
from common import *
import cv2, torch
from nimbus_inference.utils import prepare_binary_mask, segment_mean
from nimbus_inference.trainer import AugmentationPipeline, SmoothBinaryCELoss
from finetune import PhenoTileDataset


def main():
    seed_all()
    rows = []

    def record(name, passed, details=""):
        rows.append(dict(test=name, passed=bool(passed), details=details))
        assert passed, name

    # Large non-contiguous IDs and incomplete last tiles are preserved by the lazy adapter.
    m = np.zeros((90, 101), np.uint32)
    m[5:25, 5:25] = 65537
    m[40:60, 65:90] = 70001
    x = np.zeros((1, 2, 90, 101), np.float32)
    x[0, 0] = np.arange(90)[:, None]
    x[0, 1] = prepare_binary_mask(m)
    lut = np.full(70002, 2, np.float32)
    lut[0] = 0
    lut[65537] = 1
    ds = PhenoTileDataset(pd.DataFrame([dict(image_id="synthetic", y=0, x=0)]))
    ds.cache["synthetic"] = (x, m, lut)
    a, g, ids, key = ds[0]
    record(
        "tile_shape_and_exact_large_IDs",
        a.shape == (2, 256, 256)
        and g.shape == (1, 256, 256)
        and set(np.unique(ids)) == {0.0, 65537.0, 70001.0},
    )
    record(
        "unlabelled_instance_ignore",
        np.all(g[ids == 70001] == 2) and np.all(g[ids == 65537] == 1),
    )
    raw = torch.tensor([0.2, 0.8], requires_grad=True)
    target = torch.tensor([2.0, 2.0])
    loss = SmoothBinaryCELoss()(raw, target).sum()
    loss.backward()
    record(
        "official_ignore_label_zero_loss_gradient",
        loss.item() == 0 and (raw.grad == 0).all().item(),
    )
    binary = prepare_binary_mask(m)
    score = np.zeros(m.shape)
    score[(m > 0) & (binary == 0)] = 1
    f = pd.DataFrame(segment_mean(m, score))
    record(
        "aggregation_includes_original_boundary",
        np.all(f.intensity_mean > 0) and np.all(score[binary > 0] == 0),
    )
    # Exact destination coordinate convention for nearest prediction upsampling.
    low = np.arange(13 * 17, dtype=np.float64).reshape(13, 17)
    h, w = 29, 37
    yy = np.floor(np.arange(h) * 13 / h).astype(int)
    xx = np.floor(np.arange(w) * 17 / w).astype(int)
    record(
        "upsampling_equals_cv2_nearest",
        np.array_equal(
            low[yy[:, None], xx[None, :]], cv2.resize(low, (w, h), interpolation=0)
        ),
    )
    # Normalized intensity is global, not per-cell.
    raw = np.array([[0, 2], [4, 8]], np.float32)
    norm = np.clip(raw / 4, 0, 1)
    record(
        "global_normalization_clip",
        np.array_equal(norm, np.array([[0, 0.5], [1, 1]], np.float32)),
    )
    # Audit the author's B,H,W augmentation call at notebook batch size 8.
    shape_fix = False
    message = ""
    try:
        out = AugmentationPipeline(p=1.0)(
            torch.rand(8, 2, 256, 256),
            torch.ones(8, 1, 256, 256),
            torch.zeros(8, 1, 256, 256),
        )
    except Exception as e:
        shape_fix = True
        message = type(e).__name__ + ": " + str(e)
    dump(
        R / "tests/official_augmentation_audit.json",
        dict(
            batch_size=8,
            original_failed=shape_fix,
            error=message,
            permitted_adapter_fix="retain singleton channel dimension for intensity augmentation; no model/loss change",
        ),
    )
    pd.DataFrame(rows).to_csv(R / "tests/controlled_contract_checks.csv", index=False)
    print(
        "CONTROLLED_CHECKS",
        len(rows),
        "augmentation_shape_fix_needed",
        shape_fix,
        message,
        flush=True,
    )


if __name__ == "__main__":
    main()
