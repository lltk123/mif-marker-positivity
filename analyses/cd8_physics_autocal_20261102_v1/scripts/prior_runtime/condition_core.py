"""CD8 relative-signal/observable-condition ablation; no external selection."""
import os

os.environ.update(
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)
from pathlib import Path
import json, hashlib, random, collections, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from execution_guards import (
    atomic_json,
    atomic_npy,
    atomic_checkpoint,
    file_sha,
    canonical_keys_hash,
)

PROJECT = Path(
    os.environ.get("MIF_PROJECT_ROOT", str(Path(__file__).resolve().parents[4]))
)
RUN = Path(
    os.environ.get(
        "MIF_CONDITION_ROOT",
        str(PROJECT / "analyses/cd8_condition_targetaware_20261102_v1"),
    )
)
OLD = PROJECT / "analyses/cd8_representation_targetaware_20261031_v1"
SHARED = PROJECT / "analyses/cd8_robust_calibration_20261030"
TRAIN = "training_WSIs"
DEV = "internal_val_WSIs"
TEST = "external_val_TMA_cores"
GROUPS = ["A", "B", "C", "D"]
REPAIRED = (
    ["nonzero_rate_" + r for r in ["whole", "edge", "inner", "reference"]]
    + ["positive_mean_" + r for r in ["whole", "edge", "inner", "reference"]]
    + [
        "nonzero_rate_edge_minus_reference",
        "nonzero_rate_edge_minus_inner",
        "positive_mean_edge_minus_reference",
        "positive_mean_edge_minus_inner",
        "ordinal_edge_vs_reference",
    ]
)
LOCAL = (
    REPAIRED
    + [
        "repair_" + r + "_support_valid"
        for r in ["whole", "edge", "inner", "reference"]
    ]
    + [
        "repair_" + r + "_no_nonzero_signal"
        for r in ["whole", "edge", "inner", "reference"]
    ]
    + ["qc_reference_mad_zero", "crop_target_fraction", "crop_context_valid_fraction"]
)
CONDITION = [
    "log1p_U",
    "log1p_raw_p90_max",
    "zero_pixel_fraction",
    "above_U_pixel_fraction",
    "eligible_instance_fraction",
    "log1p_mean_finite_pixels",
]


def load_data():
    meta = json.loads((OLD / "data/FEATURES_COMPLETE.json").read_text())
    assert file_sha(OLD / "data/features.parquet") == meta["feature_sha256"]
    d = pd.read_parquet(OLD / "data/features.parquet")
    assert len(d) == 1077669 and np.array_equal(d.row_id, np.arange(len(d)))
    assert d.groupby("split").size().to_dict() == {
        TRAIN: 101057,
        DEV: 539132,
        TEST: 437480,
    }
    assert not d.duplicated(["dataset_id", "image_id", "cell_id", "marker"]).any()
    assert d.ground_truth_binary.isin([0, 1]).all()
    return d, meta


def condition_table():
    """All-instance unit summaries; no label columns or model scores are read."""
    b = pd.read_csv(
        SHARED / "calibration/unit_bounds.csv", float_precision="round_trip"
    )
    q = pd.read_csv(
        SHARED / "calibration/clipping_audit.csv", float_precision="round_trip"
    )
    assert not b.image_id.duplicated().any() and len(b) == 48
    # Exact column names are validated before any fit.
    q = q.set_index("image_id")
    b = b.set_index("image_id")
    out = pd.DataFrame(index=b.index)
    out["log1p_U"] = np.log1p(b.U)
    out["log1p_raw_p90_max"] = np.log1p(b.raw_cell_p90_max)
    out["zero_pixel_fraction"] = q.fraction_at_zero
    out["above_U_pixel_fraction"] = q.fraction_above_U
    out["eligible_instance_fraction"] = b.n_eligible / b.n_instances
    out["log1p_mean_finite_pixels"] = np.log1p(b.finite_instance_pixels / b.n_instances)
    assert np.isfinite(out.to_numpy()).all()
    return out.reset_index()


def preprocess(d, conditions, fit_indices):
    """Cell values fitted only on training rows; condition scaler uses training units equally."""
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    assert (d.loc[fit_indices, "split"] == TRAIN).all()
    x = d[LOCAL].to_numpy(float)
    missing = ~np.isfinite(x)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(
            np.where(missing[fit_indices], np.nan, x[fit_indices]), axis=0
        )
    med = np.where(np.isfinite(med), med, 0)
    filled = np.where(missing, med, x)
    mean = filled[fit_indices].mean(0)
    sd = filled[fit_indices].std(0)
    sd = np.where(sd > 0, sd, 1)
    local = np.concatenate(
        [np.clip((filled - mean) / sd, -5, 5), missing.astype(float)], axis=1
    )
    units = sorted(d.loc[fit_indices, "image_id"].unique())
    ct = conditions.set_index("image_id")
    cm = ct.loc[units, CONDITION].mean(0).to_numpy()
    cs = ct.loc[units, CONDITION].std(0, ddof=0).to_numpy()
    cs = np.where(cs > 0, cs, 1)
    raw = ct.loc[d.image_id, CONDITION].to_numpy()
    unclipped = (raw - cm) / cs
    cond = np.clip(unclipped, -5, 5)
    a = np.zeros((len(d), 56), np.float32)
    a[:, :48] = local
    a[:, 48:54] = cond
    assert len(LOCAL) == 24 and np.isfinite(a).all()
    state = {
        "local_columns": LOCAL,
        "median": med.tolist(),
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "condition_columns": CONDITION,
        "condition_mean": cm.tolist(),
        "condition_sd": cs.tolist(),
        "fit_units": units,
        "fit_row_hash": hashlib.sha256(fit_indices.astype("<i8").tobytes()).hexdigest(),
        "clipped_standardized_values_at": 5,
        "no_external_labels_used": True,
    }
    return a, state, np.max(abs(unclipped), axis=1) > 5


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Network(nn.Module):
    def __init__(self):
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
            nn.Linear(56, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(160, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )
        self.condition = nn.Linear(6, 256)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, image, features, group):
        a = self.encoder(image).mean((-2, -1))
        if group == "C":
            gamma, beta = (0.1 * torch.tanh(self.condition(features[:, 48:54]))).chunk(
                2, dim=1
            )
            a = a * (1 + gamma) + beta
        if group == "A":
            b = torch.zeros(
                (len(features), 32), device=features.device, dtype=features.dtype
            )
        else:
            local = features.clone()
            if group in ["B", "C"]:
                local[:, 48:] = 0
            b = self.tabular(local)
        return self.head(torch.cat([a, b], dim=1))


def initial_hash(model):
    return hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()


class Crops(Dataset):
    def __init__(self, a, y, augment=False, seed=0):
        self.a = a
        self.y = y
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.chunk = np.load(OLD / "data/cache_chunk.npy", mmap_mode="r")
        self.offset = np.load(OLD / "data/cache_offset.npy", mmap_mode="r")
        self.cache = collections.OrderedDict()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        i = int(i)
        chunk = int(self.chunk[i])
        off = int(self.offset[i])
        if chunk not in self.cache:
            stem = "chunk_%05d" % chunk
            self.cache[chunk] = [
                np.load(OLD / "data/crops" / (stem + "_" + k + ".npy"), mmap_mode="r")
                for k in ["image", "target", "other"]
            ]
            if len(self.cache) > 24:
                self.cache.popitem(last=False)
        self.cache.move_to_end(chunk)
        im, ta, ot = self.cache[chunk]
        x = np.stack(
            [
                im[off],
                np.unpackbits(ta[off]).reshape(96, 96),
                np.unpackbits(ot[off]).reshape(96, 96),
            ]
        ).astype(np.float32)
        if self.augment:
            h = (
                i * 2654435761 + self.epoch * 2246822519 + self.seed * 3266489917
            ) & 0xFFFFFFFF
            x = np.rot90(x, int(h % 4), axes=(1, 2))
            if (h >> 2) & 1:
                x = x[:, :, ::-1]
            if (h >> 3) & 1:
                x = x[:, ::-1, :]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(self.a[i]),
            int(self.y[i]),
            i,
        )


def loader(ds, indices, batch=512, workers=2):
    gen = torch.Generator()
    gen.manual_seed(771)
    return DataLoader(
        ds,
        batch_size=batch,
        sampler=np.asarray(indices).tolist(),
        num_workers=workers,
        pin_memory=True,
        generator=gen,
    )


def sample_epoch(d, fit_indices, n, seed, epoch):
    groups = [
        g.index.to_numpy() for _, g in d.loc[fit_indices].groupby("image_id", sort=True)
    ]
    rng = np.random.default_rng(seed * 10000 + epoch)
    parts = []
    for i, g in enumerate(groups):
        count = n // len(groups) + (i < n % len(groups))
        parts.append(rng.choice(g, count, replace=len(g) < count))
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def predict(model, ds, indices, group, ledger=None):
    model.eval()
    out = np.full(len(ds), np.nan, np.float32)
    with torch.no_grad():
        for x, a, y, idx in loader(ds, indices):
            if ledger:
                ledger.checkpoint()
            out[idx.numpy()] = (
                torch.softmax(
                    model(x.cuda(non_blocking=True), a.cuda(non_blocking=True), group),
                    1,
                )[:, 1]
                .cpu()
                .numpy()
            )
    return out


def metrics(y, s, t):
    y = np.asarray(y, int)
    s = np.asarray(s)
    assert np.isfinite(s).all()
    p = s > np.asarray(t, dtype=s.dtype)
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    tn = int(((y == 0) & ~p).sum())
    pos = tp + fn
    neg = fp + tn
    n = len(y)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "P": tp / (tp + fp) if tp + fp else 0.0,
        "R": tp / pos if pos else np.nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if pos else np.nan,
        "FPR": fp / neg if neg else np.nan,
        "AP": float(average_precision_score(y, s)) if pos else np.nan,
        "ROC_AUC": float(roc_auc_score(y, s)) if pos and neg else np.nan,
        "true_positive_fraction": float(y.mean()) if n else np.nan,
        "predicted_positive_fraction": float(p.mean()) if n else np.nan,
        "positive_fraction_absolute_error": abs(float(p.mean() - y.mean()))
        if n
        else np.nan,
        "Brier": float(np.mean((s - y) ** 2)) if n else np.nan,
    }


def threshold_fpr(d, s, idx, target=0.01):
    values = []
    assert (d.loc[idx, "split"] == DEV).all()
    for _, g in d.loc[idx].groupby("image_id", sort=True):
        neg = np.sort(np.asarray(s)[g.index[g.ground_truth_binary == 0]])
        k = int(np.floor(target * len(neg)))
        values.append(float(neg[-k - 1]))
    t = max(values)
    for _, g in d.loc[idx].groupby("image_id"):
        assert metrics(g.ground_truth_binary, s[g.index], t)["FPR"] <= target + 1e-12
    return t
