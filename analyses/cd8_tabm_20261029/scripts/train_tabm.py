"""Official TabM training, validation-only model selection, and frozen inference.

Each ensemble member receives its own two-class cross-entropy loss. Inference
averages member softmax probabilities, never logits. No GMM or identifiers enter X.
"""
import argparse
import copy
import gc
import importlib.metadata
import random
import resource
import time
from common_tabm import *
from resources_tabm import MemoryMonitor
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score


def runtime(root):
    sys.path.insert(0, str(root / "dependencies"))
    sys.path.insert(0, str(root / "upstream"))
    import torch
    import tabm
    import rtdl_num_embeddings

    torch.set_num_threads(4)
    return torch, tabm, rtdl_num_embeddings


def configuration():
    return dict(
        official_commit=COMMIT,
        variant="TabM with piecewise-linear embeddings (TabM dagger)",
        arch_type="tabm",
        k=32,
        n_blocks=2,
        d_block=512,
        dropout=0.1,
        embedding=dict(
            type="PiecewiseLinearEmbeddings",
            n_bins=48,
            d_embedding=16,
            activation=False,
            version="B",
            fitting="train-only quantiles",
        ),
        optimizer=dict(type="AdamW", weight_decay=0.0003),
        candidates=[
            dict(name="official_default", learning_rate=0.002),
            dict(name="lower_lr", learning_rate=0.0005),
        ],
        batch_size=256,
        eval_batch_size=1024,
        max_epochs=100,
        patience=16,
        clip_grad_norm=1.0,
        seeds=[0, 1, 2],
        share_training_batches=True,
        selection_metric="equal-WSI mean AP on original internal validation; max",
        precision="float32",
        training_loss="mean cross entropy over B*k separate member outputs",
        inference="mean(softmax(logits, class_dim)[positive], member_dim)",
        threshold_grid=[round(x / 100, 2) for x in range(1, 100)],
        threshold_comparison="strict >",
        threshold_selection="max validation equal-WSI mean F1; ties nearest .5 then higher",
        train_split=TRAIN,
        validation_split=VALID,
        test_split=TEST,
        features="A=8 raw intensity; B=A+19 morphology; C=B+63 saved spatial features; training-only drop constant/all-missing",
        GMM_input=False,
        PanM="sealed; no access",
        validation_status="development fit; used for early stopping, hyperparameters and thresholds",
        test_status="external PhenoBIC fixed until all models frozen",
        HGB=dict(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=2.0,
            early_stopping=False,
        ),
        preprocessing="train-labelled-only median then official jitter-fit normal quantile transform; no test fitting",
        geometry="existing expanded_nucleus features, unchanged segmentation",
    )


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def score_members(model, X, torch, batch_size=1024):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for a in range(0, len(X), batch_size):
            x = torch.as_tensor(
                np.asarray(X[a : a + batch_size]), device="cuda", dtype=torch.float32
            )
            logits = model(x, None)
            assert logits.ndim == 3 and logits.shape[-1] == 2
            probs = logits.softmax(dim=-1)[..., 1]
            predictions.append(probs.cpu().numpy())
    return np.concatenate(predictions)


def macro_ap(y, score, groups):
    return float(
        np.mean(
            [
                average_precision_score(y[groups == g], score[groups == g])
                for g in np.unique(groups)
            ]
        )
    )


def choose_threshold(y, score, groups, grid):
    values = []
    for t in grid:
        value = float(
            np.mean(
                [
                    f1_score(y[groups == g], score[groups == g] > t, zero_division=0)
                    for g in np.unique(groups)
                ]
            )
        )
        values.append((float(t), value))
    # Round distances only to make 0.49 and 0.51 equidistant numerically.
    winner = sorted(values, key=lambda v: (-v[1], round(abs(v[0] - 0.5), 12), -v[0]))[0]
    return winner, values


def build_model(root, Xtrain, config, torch, tabm, embeddings, bins=None):
    if bins is None:
        bins = embeddings.compute_bins(
            torch.as_tensor(Xtrain, dtype=torch.float32),
            n_bins=config["embedding"]["n_bins"],
        )
    model = tabm.TabM.make(
        n_num_features=Xtrain.shape[1],
        d_out=2,
        num_embeddings=embeddings.PiecewiseLinearEmbeddings(
            bins,
            d_embedding=config["embedding"]["d_embedding"],
            activation=False,
            version="B",
        ),
        arch_type=config["arch_type"],
        k=config["k"],
        n_blocks=config["n_blocks"],
        d_block=config["d_block"],
        dropout=config["dropout"],
    ).to("cuda")
    return model, bins


def train_one(
    root,
    group,
    seed,
    candidate,
    Xtrain,
    ytrain,
    Xval,
    yval,
    valgroups,
    torch,
    tabm,
    embeddings,
    signature,
    features,
):
    config = load_json(root / "TABM_PROTOCOL.json")
    run_id = f'TabM_{group}_s{seed}_{candidate["name"]}'
    out = root / "models" / run_id
    out.mkdir(exist_ok=True)
    if (out / "metadata.json").exists():
        meta = load_json(out / "metadata.json")
        assert meta["signature"] == signature and meta["candidate"] == candidate
        assert meta["weights_sha256"] == sha256(out / "weights.pt")
        print("REUSE", run_id, flush=True)
        return meta
    monitor = MemoryMonitor()
    seed_all(seed, torch)
    model, bins = build_model(root, Xtrain, config, torch, tabm, embeddings)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    tx = torch.as_tensor(Xtrain, device="cuda", dtype=torch.float32)
    ty = torch.as_tensor(ytrain, device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    best = -np.inf
    best_epoch = 0
    history = []
    last_improvement = 0
    training_seconds = 0.0
    validation_seconds = 0.0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        epstart = time.perf_counter()
        for idx in torch.randperm(len(tx), device="cuda").split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tx[idx], None)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), ty[idx].repeat_interleave(model.k)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss: " + run_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - epstart
        validation_start = time.perf_counter()
        members = score_members(model, Xval, torch, config["eval_batch_size"])
        score = members.mean(axis=1)
        value = macro_ap(yval, score, valgroups)
        validation_seconds += time.perf_counter() - validation_start
        record = dict(
            epoch=epoch,
            train_mean_batch_CE=float(np.mean(losses)),
            validation_macro_AP=value,
            seconds=time.perf_counter() - epstart,
        )
        history.append(record)
        print(run_id, json.dumps(record), flush=True)
        if value > best:
            best = value
            best_epoch = epoch
            last_improvement = epoch
            torch.save(
                dict(
                    state_dict={
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    bins=[b.cpu() for b in bins],
                    features=features,
                    config=config,
                    seed=seed,
                    signature=signature,
                ),
                out / "weights.pt",
            )
            np.save(out / "validation_members.npy", members)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if epoch - last_improvement >= config["patience"]:
            break
    elapsed = time.perf_counter() - start
    best_members = np.load(out / "validation_members.npy")
    (threshold, threshold_f1), grid = choose_threshold(
        yval, best_members.mean(1), valgroups, config["threshold_grid"]
    )
    pd.DataFrame(grid, columns=["threshold", "validation_macro_F1"]).to_csv(
        out / "threshold_grid.csv", index=False
    )
    meta = dict(
        model_family="TabM",
        feature_set=group,
        seed=seed,
        candidate=candidate,
        features=features,
        requested_features=load_json(root / "FEATURE_SETS.json")[group]["requested"],
        variant=config["variant"],
        k=config["k"],
        best_epoch=best_epoch,
        epochs_run=epoch,
        epoch_cap_reached=epoch == config["max_epochs"],
        validation_macro_AP=best,
        threshold_validation=threshold,
        validation_threshold_macro_F1=threshold_f1,
        threshold_status="validation development fit, not unbiased generalization",
        fit_seconds=elapsed,
        training_optimization_seconds=training_seconds,
        validation_monitoring_seconds=validation_seconds,
        fit_timing_definition="optimization plus validation and checkpoint IO",
        **monitor.finish(),
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        peak_process_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        signature=signature,
        weights_sha256=sha256(out / "weights.pt"),
        model_id=run_id,
    )
    save_json(out / "metadata.json", meta)
    del model, optimizer, tx, ty, members, best_members
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def load_inputs(root):
    info = load_json(root / "data/DATA_READY.json")
    assert sha256(root / "data/cells.parquet") == info["cells_sha256"]
    assert sha256(root / "data/X.npy") == info["matrix_sha256"]
    assert sha256(root / "models/preprocessor.joblib") == info["preprocessor_sha256"]
    assert sha256(root / "data/train_indices.npy") == info["train_indices_sha256"]
    assert (
        sha256(root / "data/validation_indices.npy")
        == info["validation_indices_sha256"]
    )
    cells = pd.read_parquet(
        root / "data/cells.parquet",
        columns=KEYS
        + [
            "split",
            "context",
            "source_image_id",
            "ground_truth_original",
            "ground_truth_binary",
            "qc_flags",
            "mask_type",
        ],
    )
    validate_keys(cells)
    X = np.load(root / "data/X.npy", mmap_mode="r")
    tr = np.load(root / "data/train_indices.npy")
    va = np.load(root / "data/validation_indices.npy")
    labelled = cells.ground_truth_binary.isin([0, 1])
    assert np.array_equal(
        tr, np.flatnonzero((labelled & cells.split.eq(TRAIN)).to_numpy())
    )
    assert np.array_equal(
        va, np.flatnonzero((labelled & cells.split.eq(VALID)).to_numpy())
    )
    return cells, X, tr, va, info


def current_signature(root, info):
    return dict(
        protocol=sha256(root / "TABM_PROTOCOL.json"),
        features=sha256(root / "FEATURE_SETS.json"),
        preprocessor=sha256(root / "models/preprocessor.joblib"),
        data=info["cells_sha256"],
        matrix=info["matrix_sha256"],
        train_indices=sha256(root / "data/train_indices.npy"),
        validation_indices=sha256(root / "data/validation_indices.npy"),
        core_code=core_hash(root),
        upstream=sha256(root / "upstream/tabm.py"),
    )


def train(root):
    torch, tabm, embeddings = runtime(root)
    config = configuration()
    if (root / "TABM_PROTOCOL.json").exists():
        assert load_json(root / "TABM_PROTOCOL.json") == config
    else:
        save_json(root / "TABM_PROTOCOL.json", config)
    cells, X, tr, va, info = load_inputs(root)
    groups = load_json(root / "FEATURE_SETS.json")
    signature = current_signature(root, info)
    if (root / "FROZEN_TEST_PROTOCOL.json").exists():
        assert load_json(root / "FROZEN_TEST_PROTOCOL.json")["signature"] == signature
        print("Compatible final protocol already frozen", flush=True)
        return
    versions = {}
    for package in [
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "joblib",
        "psutil",
    ]:
        versions[package] = importlib.metadata.version(package)
    save_json(
        root / "RUN_CONFIG.json",
        dict(
            signature=signature,
            hardware=torch.cuda.get_device_name(0),
            python=sys.version,
            versions=versions,
            tabm_version=tabm.__version__,
            rtdl_num_embeddings="0.0.12",
            runtime_executable=sys.executable,
            CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            fp32=True,
        ),
    )
    ytr = cells.iloc[tr].ground_truth_binary.to_numpy(int)
    yva = cells.iloc[va].ground_truth_binary.to_numpy(int)
    valgroups = cells.iloc[va].image_id.to_numpy()
    selected = []
    for group in "ABC":
        columns = groups[group]["matrix_indices"]
        xtr = np.asarray(X[tr][:, columns], dtype=np.float32)
        xva = np.asarray(X[va][:, columns], dtype=np.float32)
        candidates = [
            train_one(
                root,
                group,
                0,
                c,
                xtr,
                ytr,
                xva,
                yva,
                valgroups,
                torch,
                tabm,
                embeddings,
                signature,
                groups[group]["effective"],
            )
            for c in config["candidates"]
        ]
        winner = sorted(
            candidates,
            key=lambda m: (
                -m["validation_macro_AP"],
                m["candidate"]["name"] != "official_default",
            ),
        )[0]
        selected.append(
            dict(
                feature_set=group,
                selected_candidate=winner["candidate"],
                validation_macro_AP=winner["validation_macro_AP"],
            )
        )
        for seed in config["seeds"]:
            result = (
                winner
                if seed == 0
                else train_one(
                    root,
                    group,
                    seed,
                    winner["candidate"],
                    xtr,
                    ytr,
                    xva,
                    yva,
                    valgroups,
                    torch,
                    tabm,
                    embeddings,
                    signature,
                    groups[group]["effective"],
                )
            )
            canonical = root / "models" / f"TabM_{group}_s{seed}"
            canonical.mkdir(exist_ok=True)
            # Stable entry points; fitted files remain in the candidate directory.
            meta = dict(result)
            meta["candidate_model_id"] = result["model_id"]
            meta["model_id"] = canonical.name
            meta["weights_relative_path"] = str(
                (root / "models" / result["model_id"] / "weights.pt").relative_to(root)
            )
            save_json(canonical / "metadata.json", meta)
        for seed in config["seeds"]:
            model_id = f"HGB_{group}_s{seed}"
            out = root / "models" / model_id
            out.mkdir(exist_ok=True)
            if (out / "metadata.json").exists():
                assert load_json(out / "metadata.json")["signature"] == signature
                assert load_json(out / "metadata.json")["weights_sha256"] == sha256(
                    out / "model.joblib"
                )
                continue
            monitor = MemoryMonitor()
            start = time.perf_counter()
            model = HistGradientBoostingClassifier(**config["HGB"], random_state=seed)
            model.fit(xtr, ytr)
            elapsed = time.perf_counter() - start
            infer_start = time.perf_counter()
            score = model.predict_proba(xva)[:, 1]
            infer_seconds = time.perf_counter() - infer_start
            (threshold, f1), grid = choose_threshold(
                yva, score, valgroups, config["threshold_grid"]
            )
            pd.DataFrame(grid, columns=["threshold", "validation_macro_F1"]).to_csv(
                out / "threshold_grid.csv", index=False
            )
            joblib.dump(model, out / "model.joblib")
            save_json(
                out / "metadata.json",
                dict(
                    model_id=model_id,
                    model_family="HGB",
                    feature_set=group,
                    seed=seed,
                    features=groups[group]["effective"],
                    requested_features=groups[group]["requested"],
                    threshold_validation=threshold,
                    validation_macro_AP=macro_ap(yva, score, valgroups),
                    validation_threshold_macro_F1=f1,
                    fit_seconds=elapsed,
                    **monitor.finish(),
                    validation_inference_seconds=infer_seconds,
                    peak_process_rss_kib=int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    ),
                    weights_sha256=sha256(out / "model.joblib"),
                    signature=signature,
                    preprocessing="identical training-fitted median and QT as TabM",
                    parameters={**config["HGB"], "random_state": seed},
                ),
            )
            print("FITTED", model_id, "seconds", elapsed, flush=True)
        del xtr, xva
        gc.collect()
    save_json(root / "results/selected_configurations.json", selected)
    frozen = dict(
        signature=signature,
        models={
            p.parent.name: load_json(p)
            for p in sorted((root / "models").glob("*/metadata.json"))
            if p.parent.name
            in [
                f"{family}_{g}_s{s}"
                for family in ["TabM", "HGB"]
                for g in "ABC"
                for s in config["seeds"]
            ]
        },
        selected=selected,
        test_labels_used_in_selection=False,
        frozen_unix_time=time.time(),
    )
    assert len(frozen["models"]) == 18
    save_json(root / "FROZEN_TEST_PROTOCOL.json", frozen)
    print("ALL 18 FINAL MODELS AND THRESHOLDS FROZEN", flush=True)


def predict(root):
    torch, tabm, embeddings = runtime(root)
    frozen = load_json(root / "FROZEN_TEST_PROTOCOL.json")
    cells, X, tr, va, info = load_inputs(root)
    assert frozen["signature"] == current_signature(
        root, info
    ), "Frozen inputs or code changed"
    groups = load_json(root / "FEATURE_SETS.json")
    config = load_json(root / "TABM_PROTOCOL.json")
    import pyarrow as pa
    import pyarrow.parquet as pq

    timing = []
    for model_id, meta in frozen["models"].items():
        dest = root / "predictions" / f"{model_id}.parquet"
        done = dest.with_suffix(".json")
        if done.exists():
            record = load_json(done)
            assert record["frozen_sha256"] == sha256(
                root / "FROZEN_TEST_PROTOCOL.json"
            ) and record["sha256"] == sha256(dest)
            timing.append(record)
            continue
        monitor = MemoryMonitor()
        columns = groups[meta["feature_set"]]["matrix_indices"]
        if meta["model_family"] == "TabM":
            weight = root / meta["weights_relative_path"]
            assert sha256(weight) == meta["weights_sha256"]
            saved = torch.load(weight, map_location="cpu", weights_only=False)
            model, _ = build_model(
                root,
                np.asarray(X[tr[:1]][:, columns]),
                config,
                torch,
                tabm,
                embeddings,
                bins=saved["bins"],
            )
            model.load_state_dict(saved["state_dict"])
            torch.cuda.reset_peak_memory_stats()
        else:
            assert (
                sha256(root / "models" / model_id / "model.joblib")
                == meta["weights_sha256"]
            )
            model = joblib.load(root / "models" / model_id / "model.joblib")
        start = time.perf_counter()
        inference_time = 0.0
        writer = None
        temp = dest.with_suffix(".parquet.tmp")
        for a in range(0, len(cells), 65536):
            x = np.asarray(X[a : a + 65536][:, columns], dtype=np.float32)
            begin = time.perf_counter()
            if meta["model_family"] == "TabM":
                members = score_members(model, x, torch, config["eval_batch_size"])
            else:
                members = model.predict_proba(x)[:, 1, None]
            inference_time += time.perf_counter() - begin
            assert np.isfinite(members).all()
            out = cells.iloc[a : a + len(x)].copy()
            out["model_id"] = model_id
            out["model_version"] = (
                COMMIT if meta["model_family"] == "TabM" else "sklearn_HGB_fixed_150"
            )
            out["score"] = members.mean(1)
            out["score_std"] = members.std(1)
            out["score_min"] = members.min(1)
            out["score_max"] = members.max(1)
            out["prediction_fixed_0_5"] = out.score > 0.5
            out["threshold_validation"] = meta["threshold_validation"]
            out["prediction_validation_selected"] = (
                out.score > meta["threshold_validation"]
            )
            out["score_finite"] = np.isfinite(out.score)
            if meta["model_family"] == "TabM":
                out = pd.concat(
                    [
                        out,
                        pd.DataFrame(
                            members,
                            index=out.index,
                            columns=[
                                f"submodel_{i:02d}" for i in range(members.shape[1])
                            ],
                        ),
                    ],
                    axis=1,
                )
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
        writer.close()
        temp.replace(dest)
        record = dict(
            model_id=model_id,
            n=len(cells),
            **monitor.finish(),
            inference_seconds=inference_time,
            wall_seconds_including_io=time.perf_counter() - start,
            peak_process_rss_kib=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated())
            if meta["model_family"] == "TabM"
            else 0,
            sha256=sha256(dest),
            frozen_sha256=sha256(root / "FROZEN_TEST_PROTOCOL.json"),
        )
        save_json(done, record)
        timing.append(record)
        print("PREDICTED", json.dumps(record), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(timing).to_csv(root / "results/inference_resources.csv", index=False)
    save_json(
        root / "PREDICTIONS_COMPLETE.json",
        dict(
            models=len(timing),
            n_per_model=len(cells),
            frozen_sha256=sha256(root / "FROZEN_TEST_PROTOCOL.json"),
        ),
    )


def pilot(root):
    torch, tabm, embeddings = runtime(root)
    cells, X, tr, va, info = load_inputs(root)
    spec = load_json(root / "FEATURE_SETS.json")
    seed_all(0, torch)
    config = configuration()
    checks = []
    for group in "ABC":
        columns = spec[group]["matrix_indices"]
        x = np.asarray(X[tr][:, columns], dtype=np.float32)
        model, bins = build_model(root, x, config, torch, tabm, embeddings)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
        xx = torch.as_tensor(x[:256], device="cuda")
        yy = torch.as_tensor(
            cells.iloc[tr[:256]].ground_truth_binary.to_numpy(int), device="cuda"
        )
        logits = model(xx, None)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), yy.repeat_interleave(model.k)
        )
        separate = torch.stack(
            [
                torch.nn.functional.cross_entropy(logits[:, k], yy)
                for k in range(model.k)
            ]
        ).mean()
        assert torch.allclose(loss, separate, atol=1e-6)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred = score_members(model, np.asarray(X[va[:4096]][:, columns]), torch)
        assert pred.shape == (4096, 32) and np.isfinite(pred).all()
        checks.append(
            dict(
                group=group,
                n_features=len(columns),
                output_shape=list(pred.shape),
                loss=float(loss.detach()),
                separate_loss_equivalent=True,
                validation_4096_seconds=time.perf_counter() - start,
                score_min=float(pred.min()),
                score_max=float(pred.max()),
            )
        )
        del model, optimizer, x, xx, yy, logits
        torch.cuda.empty_cache()
    # A counterexample makes the required probability aggregation explicit.
    logits = torch.tensor([[[0.0, 4.0], [0.0, 0.0]]])
    p = logits.softmax(-1)[..., 1].mean(1)
    incorrect = logits.mean(1).softmax(-1)[:, 1]
    assert not torch.allclose(p, incorrect)
    save_json(
        root / "results/PILOT_VALIDATION.json",
        dict(
            checks=checks,
            probability_average=float(p[0]),
            incorrect_logit_average=float(incorrect[0]),
            pass_status=True,
        ),
    )
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("stage", choices=["pilot", "train", "predict"])
    args = parser.parse_args()
    globals()[args.stage](Path(args.run_dir))
