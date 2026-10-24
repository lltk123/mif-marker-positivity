"""Array API contract test with an explicit echo stub; not a Nimbus performance result."""
from common import *
import cv2, tempfile
from predict_api import predict_cell_scores


class Echo:
    def predict_segmentation(self, x):
        return x[:, 0:1].copy()


def main():
    image = np.arange(96 * 101, dtype=np.float32).reshape(96, 101) / 1000
    mask = np.zeros(image.shape, np.uint32)
    mask[4:35, 9:46] = 70001
    mask[56:89, 60:90] = 110009
    factor = 8.0
    app = Echo()
    d = predict_cell_scores(
        image,
        mask,
        app=app,
        normalization_factor=factor,
        dataset_id="synthetic",
        image_id="api_test",
        workdir=R / "tests/api_tmp",
    )
    # Match the documented official linear-down / nearest-up interpolation explicitly.
    expected = cv2.resize(
        cv2.resize(np.clip(image / factor, 0, 1), (101 // 2, 96 // 2)),
        (101, 96),
        interpolation=0,
    )
    for row in d.itertuples():
        assert row.pixel_count == int((mask == row.cell_id).sum())
        assert (
            abs(
                row.score
                - float(np.mean(expected[mask == row.cell_id], dtype=np.float64))
            )
            < 1e-7
        )
    assert set(d.cell_id) == set([70001, 110009])
    empty = predict_cell_scores(
        image,
        np.zeros_like(mask),
        app=app,
        normalization_factor=factor,
        dataset_id="synthetic",
        image_id="empty",
        workdir=R / "tests/api_tmp",
    )
    assert len(empty) == 0 and list(empty.columns) == KEYS + [
        "score",
        "pixel_count",
        "mask_type",
    ]
    pd.DataFrame(
        [
            dict(
                test="nonconsecutive_ID_and_official_resampling", passed=True, stub=True
            ),
            dict(test="empty_mask_fixed_schema", passed=True, stub=True),
        ]
    ).to_csv(R / "tests/array_api_contract.csv", index=False)
    print("ARRAY_API_CONTRACT_PASSED (echo stub, no model performance)", flush=True)


if __name__ == "__main__":
    main()
