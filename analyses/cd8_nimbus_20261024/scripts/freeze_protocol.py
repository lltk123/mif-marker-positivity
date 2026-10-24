"""Research freeze protocol for the frozen PhenoBIC CD8 experiment."""

from common import *


def main():
    assert not (R / "NIMBUS_PROTOCOL.json").exists()
    c = load_cells()
    exact_keys(c)
    assert len(c) == 1817783
    p = dict(
        run="cd8_nimbus_20261024",
        model_names=["Nimbus_pretrained", "Nimbus_finetuned"],
        seed=0,
        official_inference_commit=COMMIT,
        official_training_commit="6942b10cc106654c154bc057af36bcf0f0b3604d",
        weight_repo="JLrumberger/Nimbus-Inference",
        weight_revision="3affc9e774d68408701e44ed6136b1e13503c639",
        weight_file="V1.pt",
        weight_sha256=WEIGHT_SHA,
        dataset="PhenoBIC-Data",
        dataset_revision="aeffd288d922ee5df4cbb414f3c1f77507ba0001",
        keys=KEYS,
        mask_type="expanded_nucleus",
        pixel_size_known=False,
        dataset_magnification_assumed=20,
        model_magnification=10,
        scale=0.5,
        normalization=dict(
            quantile=0.999,
            foreground="raw intensity > 0",
            fit_split="training_WSIs",
            aggregate="mean of per-original-WSI quantiles",
            clip=[0, 1],
        ),
        inference=dict(
            tile_size=1024,
            batch_size=1,
            mixed_precision=False,
            compile=False,
            tta="pinned official late-bound rotation behavior: 4 x 270deg + vertical/horizontal flips",
            streaming="same global tile grid and reflection, skip output tiles with no pixels used in any original instance",
        ),
        splits=dict(
            train="training_WSIs",
            validation="internal_val_WSIs",
            test="external_val_TMA_cores",
        ),
        patient_id_missing=True,
        grouping="original WSI; external TMA slide, not cores as patients",
        pan_m="sealed, no read",
        training=dict(
            epochs=10,
            tile_size=256,
            batch_size=8,
            learning_rate=3e-6,
            weight_decay=1e-4,
            initial_regularization=1e-4,
            label_smoothing=0.05,
            patience=5,
            augmentation_p=0.5,
            warmup_steps=100,
            num_workers=1,
            optimizer="official Trainer AdamW",
            tile_inclusion="at least one explicitly labelled instance pixel; no unlabelled instance becomes negative",
            checkpoint_validation="seed0 uniform sample up to 128 eligible tiles per internal WSI; official tile-level interior score F1",
            unlabelled_activity=2,
        ),
        thresholds=dict(
            official_default=0.5,
            operator=">",
            supplementary_grid=[round(t / 100, 2) for t in range(1, 100)],
            objective="internal validation WSI equal-weight mean F1",
            ties="closest to .5 then higher",
            test_used_for_selection=False,
        ),
        historical_evaluation_exposure="PhenoBIC internal/external already evaluated in previous GMM/E1-E4; not a prospectively untouched cohort",
        input_provenance=verify_inputs(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    dump(R / "NIMBUS_PROTOCOL.json", p)
    (R / "provenance").mkdir(exist_ok=True)
    c[
        KEYS
        + [
            "split",
            "context",
            "ground_truth_original",
            "ground_truth_binary",
            "group_id",
            "grouping_level",
        ]
    ].to_parquet(R / "provenance/cell_keys.parquet", index=False)
    rows = []
    for s, g in c.groupby("split"):
        g[KEYS].sort_values(KEYS).to_csv(
            R / "provenance" / ("keys_" + s + ".csv.gz"), index=False
        )
        rows.append(
            dict(
                split=s,
                n_cells=len(g),
                n_labelled=int(g.ground_truth_binary.isin([0, 1]).sum()),
                key_sha256=key_hash(g),
            )
        )
    pd.DataFrame(rows).to_csv(R / "provenance/key_counts_hashes.csv", index=False)
    dump(
        R / "RUN_CONFIG.json",
        dict(
            protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
            code_hash_at_start=code_hash(),
            server_root=str(R),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            thread_limits=4,
        ),
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
