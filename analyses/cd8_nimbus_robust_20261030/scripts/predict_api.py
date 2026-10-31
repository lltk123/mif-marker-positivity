"""Array inference with explicit full-unit L/U; never estimate bounds on a crop."""
import numpy as np
import pandas as pd
import cv2
from common import KEYS
from normalization import normalize_pixels


def predict_cell_scores(
    image,
    instance_mask,
    *,
    app,
    lower,
    upper,
    dataset_id,
    image_id,
    workdir,
    valid=None,
    mask_type="expanded_nucleus"
):
    from inference import official_tta_stream
    from nimbus_inference.utils import prepare_binary_mask, prepare_input_data

    image = np.asarray(image)
    mask = np.asarray(instance_mask)
    if image.ndim != 2 or image.shape != mask.shape or mask.dtype.kind not in "iu":
        raise ValueError("matching 2D intensity and integer-instance arrays required")
    if np.any(mask < 0):
        raise ValueError("instance IDs must be nonnegative")
    columns = KEYS + [
        "score",
        "pixel_count",
        "input_valid",
        "score_finite",
        "mask_type",
    ]
    fg = mask > 0
    if not fg.any():
        return pd.DataFrame(columns=columns)
    ids, inverse = np.unique(mask[fg], return_inverse=True)
    counts = np.bincount(inverse)
    values = normalize_pixels(image, lower, upper, valid)
    data = dict(
        dataset_id=dataset_id,
        image_id=image_id,
        cell_id=ids,
        marker="CD8",
        score=np.full(len(ids), np.nan),
        pixel_count=counts,
        input_valid=False,
        score_finite=False,
        mask_type=mask_type,
    )
    if not np.isfinite(values).any():
        return pd.DataFrame(data)[columns]
    if not np.isfinite(values[fg]).all():
        raise ValueError("an original instance intersects invalid image support")
    h, w = image.shape
    shape = (w // 2, h // 2)
    if min(shape) < 32:
        raise ValueError("insufficient context for official scale=0.5 UNet input")
    signal = np.nan_to_num(values, nan=0.0).astype(np.float32)
    inp = prepare_input_data(
        cv2.resize(signal, shape),
        binary_mask=cv2.resize(prepare_binary_mask(mask), shape, interpolation=0),
    )
    low = official_tta_stream(app, inp, np.ones(inp.shape[-2:], bool), workdir)
    full = cv2.resize(low, (w, h), interpolation=0)
    scores = np.bincount(inverse, weights=full[fg]) / counts
    data.update(score=scores, input_valid=True, score_finite=np.isfinite(scores))
    result = pd.DataFrame(data)[columns]
    assert not result.duplicated(KEYS).any()
    return result
