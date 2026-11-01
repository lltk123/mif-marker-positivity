"""Paired target-aware image and tabular ablations under one frozen budget."""
import os

os.environ.update(
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)
import json, time, random, resource, argparse, hashlib, collections, warnings
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from experiment_common import *
from execution_guards import (
    BudgetLedger,
    BudgetExhausted,
    atomic_npy,
    atomic_checkpoint,
)

_LEDGER = None
GROUPS = [
    "image",
    "image_target",
    "image_target_other",
    "tabular_only",
    "image_target_other_tabular",
]


class Network(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        layers = []
        last = 3
        for width in [32, 64, 96, 128]:
            layers += [
                nn.Conv2d(last, width, 3, stride=2, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            ]
            last = width
        self.encoder = nn.Sequential(*layers)
        self.tabular = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(160, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )

    def forward(self, image, features, group):
        if group == "tabular_only":
            a = torch.zeros(
                (len(features), 128), device=features.device, dtype=features.dtype
            )
        else:
            a = self.encoder(image).mean((-2, -1))
        if group in ["tabular_only", "image_target_other_tabular"]:
            b = self.tabular(features)
        else:
            b = torch.zeros(
                (len(features), 32), device=features.device, dtype=features.dtype
            )
        return self.head(torch.cat([a, b], dim=1))


class Crops(Dataset):
    def __init__(self, features, labels, group, augment=False, seed=0):
        self.features = features
        self.labels = labels
        self.group = group
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.chunk = np.load(RUN / "data/cache_chunk.npy", mmap_mode="r")
        self.offset = np.load(RUN / "data/cache_offset.npy", mmap_mode="r")
        self.cache = collections.OrderedDict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        i = int(i)
        a = np.zeros((3, 96, 96), np.float32)
        if self.group != "tabular_only":
            chunk = int(self.chunk[i])
            off = int(self.offset[i])
            if chunk not in self.cache:
                stem = "chunk_%05d" % chunk
                self.cache[chunk] = [
                    np.load(
                        RUN / "data/crops" / (stem + "_" + kind + ".npy"), mmap_mode="r"
                    )
                    for kind in ["image", "target", "other"]
                ]
                if len(self.cache) > 24:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(chunk)
            im, ta, ot = self.cache[chunk]
            a[0] = im[off]
            if self.group != "image":
                a[1] = np.unpackbits(ta[off]).reshape(96, 96)
            if self.group in ["image_target_other", "image_target_other_tabular"]:
                a[2] = np.unpackbits(ot[off]).reshape(96, 96)
            if self.augment:
                # Row/epoch/seed-derived geometry is identical between all paired groups.
                h = (
                    i * 2654435761 + self.epoch * 2246822519 + self.seed * 3266489917
                ) & 0xFFFFFFFF
                a = np.rot90(a, int(h % 4), axes=(1, 2))
                if (h >> 2) & 1:
                    a = a[:, :, ::-1]
                if (h >> 3) & 1:
                    a = a[:, ::-1, :]
            a = np.ascontiguousarray(a)
        tab = (
            self.features[i]
            if self.group in ["tabular_only", "image_target_other_tabular"]
            else np.zeros(self.features.shape[1], np.float32)
        )
        return (
            torch.from_numpy(a),
            torch.from_numpy(np.asarray(tab, np.float32)),
            int(self.labels[i]),
            i,
        )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tabular_matrix(d, cols, train):
    X = d[cols].to_numpy(float)
    missing = np.isnan(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(X[train], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    A = np.where(missing, median[None, :], X)
    mean = A[train].mean(0)
    sd = A[train].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = np.concatenate([(A - mean) / sd, missing.astype(float)], axis=1).astype(
        np.float32
    )
    if not np.isfinite(A).all():
        raise ValueError("Nonfinite tabular matrix")
    prep = {
        "columns": cols,
        "median": median.tolist(),
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "indicators": "one per requested measured feature; states not biological truth",
    }
    return A, prep


def subset(d):
    path = RUN / "data/neural_checkpoint_validation_indices.npy"
    if path.exists():
        return np.load(path)
    ids = []
    for uid, g in d[d.split == DEV].groupby("image_id", sort=True):
        keys = [
            int.from_bytes(
                hashlib.blake2b(
                    (str(uid) + "|" + str(int(c))).encode(), digest_size=8
                ).digest(),
                "little",
            )
            for c in g.cell_id
        ]
        ids.extend(g.index.to_numpy()[np.argsort(np.asarray(keys, np.uint64))[:10000]])
    ids = np.sort(np.asarray(ids, np.int64))
    np.save(path, ids)
    return ids


def loader(ds, indices, batch=512, workers=2):
    gen = torch.Generator()
    gen.manual_seed(771)
    return DataLoader(
        ds,
        batch_size=batch,
        sampler=indices.tolist(),
        num_workers=workers,
        pin_memory=True,
        generator=gen,
        persistent_workers=False,
    )


def predict(model, ds, indices, group, batch=512, ledger=None):
    model.eval()
    out = np.full(len(ds), np.nan, np.float32)
    started = time.time()
    with torch.no_grad():
        for image, features, y, rows in loader(ds, np.asarray(indices), batch):
            if ledger:
                ledger.checkpoint()
            score = torch.softmax(
                model(
                    image.cuda(non_blocking=True),
                    features.cuda(non_blocking=True),
                    group,
                ),
                1,
            )[:, 1]
            out[rows.numpy()] = score.cpu().numpy()
    torch.cuda.synchronize()
    return out, time.time() - started


def active_parameters(model, group):
    enc = sum(p.numel() for p in model.encoder.parameters())
    tab = sum(p.numel() for p in model.tabular.parameters())
    head = 64 * 160 + 64 + 2 * 64 + 2
    active = head
    if group == "tabular_only":
        active -= 64 * 128
        active += tab
    else:
        active -= 0 if group == "image_target_other_tabular" else 64 * 32
        active += enc
        missing = 2 if group == "image" else 1 if group == "image_target" else 0
        active -= missing * 32 * 3 * 3
        if group == "image_target_other_tabular":
            active += tab
    return int(active)


def main():
    global _LEDGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    d, manifest = load()
    train = (d.split == TRAIN).to_numpy()
    dev = (d.split == DEV).to_numpy()
    devidx = d.index[dev].to_numpy()
    development = d.index[(train | dev)].to_numpy()
    y = d.ground_truth_binary.to_numpy(np.int64)
    A, prep = tabular_matrix(d, manifest["feature_sets"]["H2"], train)
    savej(RUN / "data/NEURAL_TABULAR_PREPROCESSOR.json", prep)
    valid_idx = subset(d)
    if args.smoke_only:
        records = []
        for group in GROUPS:
            seed_all(0)
            m = Network(A.shape[1]).cuda()
            ds = Crops(A, y, group)
            image, features, z, idx = next(
                iter(loader(ds, d.index[train].to_numpy()[:8], batch=8, workers=0))
            )
            loss = nn.functional.cross_entropy(
                m(image.cuda(), features.cuda(), group), z.cuda()
            )
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in m.parameters()
                if p.grad is not None
            )
            records.append(
                {
                    "group": group,
                    "loss_finite": bool(torch.isfinite(loss)),
                    "nominal_parameters": sum(p.numel() for p in m.parameters()),
                    "active_parameters": active_parameters(m, group),
                    "target_masks_binary": bool(
                        np.isin(image[:, 1:].numpy(), [0, 1]).all()
                    ),
                    "optimizer_steps": 0,
                    "initial_state_sha256": hashlib.sha256(
                        b"".join(
                            v.detach().cpu().numpy().tobytes()
                            for v in m.state_dict().values()
                        )
                    ).hexdigest(),
                }
            )
        assert len(set(z["initial_state_sha256"] for z in records)) == 1
        savej(RUN / "results/NEURAL_SMOKE_COMPLETE.json", records)
        print(json.dumps(records), flush=True)
        return
    assert (RUN / "results/NEURAL_SMOKE_COMPLETE.json").exists()
    budget_path = RUN / "logs/GPU_BUDGET.json"
    ledger = BudgetLedger(budget_path)
    _LEDGER = ledger
    budget = ledger.data
    models = []
    for seed in [0, 1, 2]:
        for group in GROUPS:
            name = "N_" + group + "_s" + str(seed)
            done = RUN / "models" / (name + ".json")
            path = RUN / "models" / (name + ".pt")
            if done.exists():
                models.append(json.loads(done.read_text()))
                continue
            if budget["consumed_seconds"] >= budget["hard_cap_seconds"]:
                raise BudgetExhausted("No remaining budget")
            started = time.time()
            ledger.begin(name)
            seed_all(seed)
            model = Network(A.shape[1]).cuda()
            savej(
                RUN / "provenance" / (name + "_initialization.json"),
                {
                    "recorded_during_training": True,
                    "seed": seed,
                    "hash": hashlib.sha256(
                        b"".join(
                            v.detach().cpu().numpy().tobytes()
                            for v in model.state_dict().values()
                        )
                    ).hexdigest(),
                },
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0003, weight_decay=0.0001
            )
            ds = Crops(A, y, group, True, seed)
            ev = Crops(A, y, group)
            best = -np.inf
            wait = 0
            epochs = []
            train_groups = [
                g.index.to_numpy()
                for _, g in d.loc[train].groupby("image_id", sort=True)
            ]
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(10):
                ledger.checkpoint()
                ds.epoch = epoch
                rng = np.random.default_rng(seed * 10000 + epoch)
                indices = np.concatenate(
                    [
                        rng.choice(g, 16667 if k < 2 else 16666, replace=len(g) < 16667)
                        for k, g in enumerate(train_groups)
                    ]
                )
                rng.shuffle(indices)
                savej(
                    RUN
                    / "provenance"
                    / (name + "_epoch" + str(epoch) + "_indices.json"),
                    {
                        "recorded_during_training": True,
                        "seed": seed,
                        "epoch": epoch,
                        "rows": len(indices),
                        "sha256": hashlib.sha256(
                            np.asarray(indices, dtype="<i8").tobytes()
                        ).hexdigest(),
                    },
                )
                model.train()
                losses = []
                epoch_start = time.time()
                for image, features, z, rows in loader(ds, indices, batch=128):
                    ledger.checkpoint()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(
                        image.cuda(non_blocking=True),
                        features.cuda(non_blocking=True),
                        group,
                    )
                    loss = nn.functional.cross_entropy(
                        logits, z.cuda(non_blocking=True)
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                scores, eval_sec = predict(model, ev, valid_idx, group, ledger=ledger)
                apval = macro_ap(d.loc[valid_idx], scores)
                rec = {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "checkpoint_macro_AP": apval,
                    "seconds": time.time() - epoch_start,
                    "n_training_examples": len(indices),
                    "n_checkpoint_validation": len(valid_idx),
                }
                epochs.append(rec)
                print("EPOCH", name, json.dumps(rec), flush=True)
                if apval > best + 1e-5:
                    best = apval
                    wait = 0
                    atomic_checkpoint(
                        path,
                        {
                            "state_dict": model.state_dict(),
                            "epoch": epoch + 1,
                            "seed": seed,
                            "group": group,
                            "n_features": A.shape[1],
                            "best_checkpoint_macro_AP": best,
                            "feature_sha256": manifest["feature_sha256"],
                            "protocol_sha256": sha(RUN / "PROTOCOL.json"),
                        },
                    )
                else:
                    wait += 1
                if wait >= 3:
                    break
            if not path.exists():
                raise RuntimeError("No neural checkpoint before budget")
            b = torch.load(path, map_location="cpu")
            model.load_state_dict(b["state_dict"])
            model.cuda()
            scores, infer = predict(model, ev, development, group, ledger=ledger)
            atomic_npy(RUN / "predictions" / (name + "_development.npy"), scores)
            ts, grid = thresholds(d.loc[dev], scores)
            pd.DataFrame(grid).to_csv(
                RUN / "results" / (name + "_development_threshold_grid.csv"),
                index=False,
            )
            rec = {
                "model": name,
                "family": "neural",
                "group": group,
                "seed": seed,
                "epochs_run": len(epochs),
                "best_epoch": b["epoch"],
                "best_checkpoint_macro_AP": best,
                "development_macro_AP": macro_ap(d.loc[dev], scores),
                "thresholds": ts,
                "nominal_parameters": sum(p.numel() for p in model.parameters()),
                "active_parameters": active_parameters(model, group),
                "fit_and_development_seconds": time.time() - started,
                "development_inference_seconds": infer,
                "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
                "process_peak_RSS_KiB": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "model_sha256": sha(path),
                "feature_sha256": manifest["feature_sha256"],
                "deployment_candidate": True,
                "checkpoint_validation_population": "fixed label-blind <=10000 per internal WSI; full internal set used for final thresholds and candidate comparison",
                "no_external_selection": True,
            }
            savej(done, rec)
            pd.DataFrame(epochs).to_csv(
                RUN / "results" / (name + "_epochs.csv"), index=False
            )
            models.append(rec)
            ledger.finish("running")
            print("NEURAL_DONE", json.dumps(rec), flush=True)
            del model, optimizer, ds, ev
            torch.cuda.empty_cache()
    ledger.data["status"] = "completed"
    ledger.persist()
    pd.DataFrame(
        [
            {k: v for k, v in z.items() if not isinstance(v, (dict, list))}
            for z in models
        ]
    ).to_csv(RUN / "results/NEURAL_DEVELOPMENT.csv", index=False)
    savej(
        RUN / "models/NEURAL_COMPLETE.json",
        {
            "n_models": len(models),
            "models": [z["model"] for z in models],
            "seeds": [0, 1, 2],
            "budget": budget,
            "no_external_metrics_inspected": True,
            "code_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _LEDGER is not None:
            _LEDGER.abort(error)
            if isinstance(error, BudgetExhausted):
                savej(
                    RUN / "models/NEURAL_BUDGET_REACHED.json",
                    {"status": "partial_budget_exhausted", "budget": _LEDGER.data},
                )
        raise
