"""Freeze the new experiment only after shared calibration and original keys verify."""
from common import *
from normalization import calibration, calibration_signature, SHARED


def main():
    table, signature, manifest = calibration()
    cells = load_cells()
    exact_keys(cells)
    assert len(cells) == 1817783
    assert (
        key_hash(cells) == manifest["all_keys_sha256"]
    ), "shared and Nimbus cell keys differ"
    assert set(cells.image_id) == set(table.index)
    assert sha(R / "weights/V1.pt") == WEIGHT_SHA
    state = json.loads((PREVIOUS_NIMBUS / "NIMBUS_PROTOCOL.json").read_text())
    state.update(
        run=R.name,
        model_names=[MODEL_PRETRAINED, MODEL_FINETUNED],
        normalization={
            "method_id": "E1_outer_P1_P99",
            "shared_run": str(SHARED),
            "inner_percentiles": [10, 90],
            "outer_percentiles": [1, 99],
            "lower": "P1 of original-instance P10",
            "upper": "P99 of original-instance P90",
            "eligibility": ">=5 finite valid original-instance pixels; labelled or unlabelled",
            "unit": "original WSI; deduplicated internal WSI; individual external core",
            "mapping": "clip((I-L)/(U-L),0,1)",
            "calculation_dtype": "float64",
            "network_dtype": "float32",
            "zero_is_valid": True,
            "invalid": "NaN plus valid support; invalid calibration produces unscorable cells",
            "official_input_adaptation": True,
            "additional_marker_division": False,
            **signature,
        },
        input_provenance=verify_inputs(),
        all_keys_sha256=key_hash(cells),
        previous_protocol_sha256=sha(PREVIOUS_NIMBUS / "NIMBUS_PROTOCOL.json"),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    state["training"][
        "initial_checkpoint"
    ] = "original V1.pt, not previous fine-tuned weights"
    path = R / "NIMBUS_PROTOCOL.json"
    if path.exists():
        existing = json.loads(path.read_text())
        state["created_utc"] = existing["created_utc"]
        assert existing == state, "existing frozen protocol differs"
    else:
        dump(path, state)
    cells[
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
    for split, group in cells.groupby("split"):
        rows.append(
            dict(
                split=split,
                n_cells=len(group),
                n_labelled=int(group.ground_truth_binary.isin([0, 1]).sum()),
                key_sha256=key_hash(group),
            )
        )
    pd.DataFrame(rows).to_csv(R / "provenance/key_counts_hashes.csv", index=False)
    config = dict(
        protocol_sha256=sha(path),
        code_sha256=code_hash(),
        server_root=str(R),
        python=sys.executable,
        thread_limits=4,
        shared_calibration=signature,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    config_path = R / "RUN_CONFIG.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "run signature changed"
    else:
        dump(config_path, config)
    print("FROZEN", rows)


if __name__ == "__main__":
    main()
