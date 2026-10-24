"""Reusable CD8 array API. Caller supplies the frozen marker normalization and loaded model."""
from common import KEYS
from inference import official_tta_stream
import numpy as np, pandas as pd, cv2
from nimbus_inference.utils import prepare_binary_mask, prepare_input_data


def predict_cell_scores(
    image,
    instance_mask,
    *,
    app,
    normalization_factor,
    dataset_id,
    image_id,
    workdir,
    mask_type="expanded_nucleus",
):
    """Return one CD8 score per original instance ID using the frozen marker normalization."""
    image = np.asarray(image)
    mask = np.asarray(instance_mask)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("CD8/mask must be matching 2D arrays")
    if mask.dtype.kind not in "iu" or np.any(mask < 0):
        raise ValueError("integer nonnegative instance IDs required")
    columns = KEYS + ["score", "pixel_count", "mask_type"]
    fg = mask > 0
    if not fg.any():
        return pd.DataFrame(columns=columns)
    if not np.isfinite(image).all():
        raise ValueError(
            "complete finite image support required; do not treat missing context as background"
        )
    if not np.isfinite(normalization_factor) or normalization_factor <= 0:
        raise ValueError("invalid frozen normalization factor")
    h, w = image.shape
    shape = (w // 2, h // 2)
    if min(shape) < 32:
        raise ValueError("image context too small for official 0.5-scale UNet")
    # This copy preserves the caller's original dtype, range and IDs.
    signal = np.clip(image.astype(np.float32) / normalization_factor, 0, 1)
    inp = prepare_input_data(
        cv2.resize(signal, shape),
        binary_mask=cv2.resize(prepare_binary_mask(mask), shape, interpolation=0),
    )
    low = official_tta_stream(app, inp, np.ones(inp.shape[-2:], bool), workdir)
    full = cv2.resize(low, (w, h), interpolation=0)
    # Temporary compact indexing only for sums; the returned IDs are original IDs.
    ids, inverse = np.unique(mask[fg], return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=full[fg])
    d = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "image_id": image_id,
            "cell_id": ids,
            "marker": "CD8",
            "score": total / count,
            "pixel_count": count,
            "mask_type": mask_type,
        }
    )
    assert not d.duplicated(KEYS).any()
    return d[columns]
