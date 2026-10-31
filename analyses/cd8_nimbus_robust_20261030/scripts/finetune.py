"""Official Trainer with a lazy sparse-label/uint32 PhenoBIC Dataset adapter."""
from common import *
import torch, gc, shutil, argparse
from inference import reflect_indices, make_model
from nimbus_inference.trainer import Trainer, AugmentationPipeline
from normalization import bounds_for, calibration_signature


def build_indices():
    cells = load_cells()
    rows = []
    summary = []
    missing = []
    for image, es in units().items():
        split = es[0]["split"]
        if split not in ["training_WSIs", "internal_val_WSIs"]:
            continue
        u = unit_dir(image)
        if bounds_for(image)["calibration_invalid"]:
            c = cells.loc[cells.image_id == image]
            known = c.ground_truth_binary.isin([0, 1])
            missing.append(c.loc[known, KEYS].assign(reason="calibration_invalid"))
            summary.append(
                dict(
                    image_id=image,
                    split=split,
                    eligible_tiles=0,
                    used_tiles=0,
                    labelled_instances=int(known.sum()),
                    instances_absent_from_downsampled_support=int(known.sum()),
                    calibration_invalid=True,
                )
            )
            continue
        ids = cached(image, "ids_scaled")
        h, w = ids.shape
        c = cells.loc[cells.image_id == image]
        maxid = int(c.cell_id.max())
        labels = np.full(maxid + 1, 2, np.float32)
        labels[0] = 0
        known = c.ground_truth_binary.isin([0, 1])
        labels[c.loc[known, "cell_id"].to_numpy(int)] = c.loc[
            known, "ground_truth_binary"
        ].to_numpy(np.float32)
        np.save(u / "label_lookup.npy", labels)
        found = np.zeros(maxid + 1, bool)
        eligible = []
        for y in range(0, h, 256):
            for x in range(0, w, 256):
                m = ids[y : y + 256, x : x + 256]
                g = labels[m]
                selected = (m > 0) & (g < 2)
                if selected.any():
                    found[np.unique(m[selected])] = True
                    eligible.append(
                        dict(
                            image_id=image,
                            unit=unit_code(image),
                            split=split,
                            y=y,
                            x=x,
                            n_human_pixels=int(selected.sum()),
                        )
                    )
        source_n = len(eligible)
        if split == "internal_val_WSIs" and len(eligible) > 128:
            ix = np.sort(
                np.random.default_rng(0).choice(len(eligible), 128, replace=False)
            )
            eligible = [eligible[int(k)] for k in ix]
        for d in eligible:
            d["role"] = (
                "training" if split == "training_WSIs" else "checkpoint_validation"
            )
        rows.extend(eligible)
        absent = c.loc[known & ~c.cell_id.map(lambda k: bool(found[int(k)])), KEYS]
        if len(absent):
            missing.append(
                absent.assign(
                    reason="no labelled pixels remain after official nearest 0.5 downsampling"
                )
            )
        summary.append(
            dict(
                image_id=image,
                split=split,
                eligible_tiles=source_n,
                used_tiles=len(eligible),
                labelled_instances=int(known.sum()),
                instances_absent_from_downsampled_support=len(absent),
            )
        )
    pd.DataFrame(rows).to_csv(R / "provenance/finetuning_tile_index.csv", index=False)
    pd.DataFrame(summary).to_csv(
        R / "results/finetuning_data_coverage.csv", index=False
    )
    (
        pd.concat(missing, ignore_index=True)
        if missing
        else pd.DataFrame(columns=KEYS + ["reason"])
    ).to_csv(R / "results/finetuning_instances_without_scaled_support.csv", index=False)
    dump(
        R / "provenance/finetuning_tile_index.sha256.json",
        {
            "sha256": sha(R / "provenance/finetuning_tile_index.csv"),
            "selection": "all eligible training tiles; uniform seed0 up to128 validation tiles per WSI, before model scoring",
            "code_sha256": sha(__file__),
            "protocol_sha256": sha(R / "NIMBUS_PROTOCOL.json"),
            "shared_calibration": calibration_signature(),
        },
    )
    return pd.DataFrame(rows)


class PhenoTileDataset(torch.utils.data.Dataset):
    def __init__(self, index):
        self.index = index.to_dict("records")
        self.cache = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        d = self.index[k]
        i = d["image_id"]
        y = int(d["y"])
        x = int(d["x"])
        if i not in self.cache:
            self.cache[i] = (
                cached(i, "input"),
                cached(i, "ids_scaled"),
                cached(i, "label_lookup"),
            )
        inp, ids, lut = self.cache[i]
        h, w = ids.shape
        yy = reflect_indices(np.arange(y, y + 256), h)
        xx = reflect_indices(np.arange(x, x + 256), w)
        a = np.array(inp[0][:, yy[:, None], xx[None, :]], dtype=np.float32, copy=True)
        # numpy advanced indexing places spatial axes first when scalar axis is present.
        if a.shape != (2, 256, 256):
            a = np.moveaxis(a, -1, 0)
        m = np.array(ids[yy[:, None], xx[None, :]], dtype=np.float32)[None]
        g = lut[m.astype(np.int64)]
        return a, g, m, f"{i}_,_CD8_,_{y}_,_{x}"


class ChannelSafeAugmentation(AugmentationPipeline):
    def __call__(self, input_tensor, mask_tensor, label_tensor):
        a, m, g = self.geometric_transforms(input_tensor, mask_tensor, label_tensor)
        a[:, 0:1] = self.intensity_transforms(a[:, 0:1])
        return a, m, g


class LoggedTrainer(Trainer):
    def __init__(self, *a, **kw):
        self.validation_calls = 0
        self.validation_records = []
        super().__init__(*a, **kw)
        audit = json.loads((R / "tests/official_augmentation_audit.json").read_text())
        if audit["original_failed"]:
            self.augmenter = ChannelSafeAugmentation(
                p=protocol()["training"]["augmentation_p"]
            )

    def run_validation(self):
        start = time.perf_counter()
        r = super().run_validation()
        row = dict(
            validation_call=self.validation_calls,
            seconds=time.perf_counter() - start,
            learning_rate_applied=float(self.optimizer.param_groups[0]["lr"]),
            **{k: float(v) for k, v in r.items()},
        )
        self.validation_records.append(row)
        self.validation_calls += 1
        pd.DataFrame(self.validation_records).to_csv(
            R / "results/finetuning_validation_history.csv", index=False
        )
        return r

    def save_checkpoint(self):
        super().save_checkpoint()
        assert (
            Path(self.checkpoint_path)
            .resolve()
            .is_relative_to((R / "upstream/src").resolve())
        ), "Trainer attempted writing outside isolated source"
        shutil.copy2(self.checkpoint_path, R / "models/Nimbus_outerP1P99_finetuned.pt")
        dump(
            R / "models/finetuned_checkpoint.json",
            dict(
                validation_call=self.validation_calls - 1,
                selection_metric="official tile/interior mean_f1",
                best_f1=float(self.best_f1),
                weight_sha256=sha(self.checkpoint_path),
                pretrained_sha256=WEIGHT_SHA,
            ),
        )

    def _print_epoch_summary(self, *a, **kw):
        super()._print_epoch_summary(*a, **kw)
        dump(R / "results/finetuning_history.json", self.history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    seed_all()
    index_path = R / "provenance/finetuning_tile_index.csv"
    if index_path.exists():
        index_state = json.loads(
            (R / "provenance/finetuning_tile_index.sha256.json").read_text()
        )
        assert index_state["sha256"] == sha(index_path)
        assert index_state["code_sha256"] == sha(__file__)
        assert index_state["protocol_sha256"] == sha(R / "NIMBUS_PROTOCOL.json")
        assert index_state["shared_calibration"] == calibration_signature()
        index = pd.read_csv(index_path)
    else:
        index = build_indices()
    if a.prepare_only:
        return
    assert (R / "tests/PILOT_PASSED.json").exists()
    assert (
        R / "provenance/pretrained_inference_complete.json"
    ).exists(), "pretrained full inference must complete before finetuning"
    assert not (
        R / "models/finetuned_complete.json"
    ).exists(), "do not overwrite completed fit"
    tr = PhenoTileDataset(index.query("role=='training'"))
    va = PhenoTileDataset(index.query("role=='checkpoint_validation'"))
    print("TRAIN_TILES", len(tr), "VALIDATION_TILES", len(va), flush=True)
    previous = pd.read_csv(PREVIOUS_NIMBUS / "provenance/finetuning_tile_index.csv")
    compare_columns = ["image_id", "y", "x", "role"]
    # Without a calibration failure, exactly the original geometric train/validation tile set is retained.
    if not any(
        bounds_for(i)["calibration_invalid"]
        for i in units()
        if not i.startswith("external")
    ):
        pd.testing.assert_frame_equal(
            index[compare_columns].reset_index(drop=True),
            previous[compare_columns].reset_index(drop=True),
        )
    n = make_model()
    n.input_shape = [256, 256]
    cfg = protocol()["training"]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = LoggedTrainer(
        n,
        tr,
        va,
        checkpoint_name="Nimbus_outerP1P99_finetuned.pt",
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        initial_regularization=cfg["initial_regularization"],
        num_workers=cfg["num_workers"],
        label_smoothing=cfg["label_smoothing"],
        patience=cfg["patience"],
        augmentation_p=cfg["augmentation_p"],
        warmup_steps=cfg["warmup_steps"],
    )
    trainer.train(epochs=cfg["epochs"])
    assert (
        R / "models/Nimbus_outerP1P99_finetuned.pt"
    ).exists(), "official trainer saved no checkpoint"
    dump(
        R / "models/finetuned_complete.json",
        dict(
            seconds=time.perf_counter() - start,
            weight_sha256=sha(R / "models/Nimbus_outerP1P99_finetuned.pt"),
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            peak_rss_gib=__import__("resource")
            .getrusage(__import__("resource").RUSAGE_SELF)
            .ru_maxrss
            / 1024**2,
            training_tiles=len(tr),
            validation_tiles=len(va),
            validation_calls=trainer.validation_calls,
            code_sha256=sha(__file__),
            shared_calibration=calibration_signature(),
            initial_weight_sha256=WEIGHT_SHA,
            protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
        ),
    )
    print("FINETUNING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
