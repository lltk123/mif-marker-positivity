"""Validate real fine-tuning tiles without fitting or changing model weights."""
from common import *
from finetune import PhenoTileDataset


def main():
    index = pd.read_csv(R / "provenance/finetuning_tile_index.csv")
    rows = []
    for image, g in index.groupby("image_id"):
        subset = g.iloc[np.unique([0, len(g) // 2, len(g) - 1])]
        ds = PhenoTileDataset(subset)
        for n in range(len(ds)):
            a, y, m, key = ds[n]
            assert a.shape == (2, 256, 256) and y.shape == m.shape == (1, 256, 256)
            assert np.isfinite(a).all() and a[0].min() >= 0 and a[0].max() <= 1
            assert set(np.unique(a[1])) <= set([0, 1])
            assert set(np.unique(y)) <= set([0, 1, 2])
            assert np.equal(m, np.round(m)).all()
            original = cached(image, "label_lookup")
            assert np.array_equal(y, original[m.astype(np.int64)])
            rows.append(
                dict(
                    image_id=image,
                    tile_key=key,
                    passed=True,
                    max_original_id=int(m.max()),
                    n_positive_pixels=int((y == 1).sum()),
                    n_ignored_pixels=int((y == 2).sum()),
                    n_background_pixels=int((m == 0).sum()),
                    input_min=float(a[0].min()),
                    input_max=float(a[0].max()),
                )
            )
    pd.DataFrame(rows).to_csv(R / "tests/real_finetuning_tile_checks.csv", index=False)
    print("REAL_TILE_CONTRACT_PASSED", len(rows))


if __name__ == "__main__":
    main()
