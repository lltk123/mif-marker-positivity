# Reuse immutable prior A weights/crops. New synthetic proxy transforms only.
import os, sys, json, time, hashlib, importlib.util
from pathlib import Path
import numpy as np

P = Path(os.environ.get("MIF_PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
BASE = P / "analyses/cd8_condition_targetaware_20261102_v1"
R = Path(os.environ.get("MIF_RUN_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(Path(__file__).resolve().parent / "prior_runtime"))
import condition_core as cc
from execution_guards import atomic_json, atomic_npy, file_sha, BudgetLedger
import torch, pandas as pd

SCENARIOS = (
    [("gain_" + str(v), {"gain": v}) for v in [0.25, 0.5, 0.75, 1.5, 2.0, 4.0]]
    + [("gamma_" + str(v), {"gamma": v}) for v in [0.7, 1.3]]
    + [("zero_" + str(v), {"zero": v}) for v in [0.7, 0.85, 0.92, 0.95, 0.98]]
    + [
        ("noise_low", {"eta": 0.002, "sigma": 0.005}),
        ("noise_high", {"eta": 0.01, "sigma": 0.02}),
        ("compound", {"gain": 0.4, "sigma": 0.01, "zero": 0.95}),
    ]
)
PARAMS = dict(SCENARIOS)
PARAMS["clean"] = {}


def transform(im, valid, row, scenario, threshold=0):
    p = PARAMS[scenario]
    x = np.asarray(im, np.float32).copy()
    if "gain" in p:
        x = np.clip(x * p["gain"], 0, 1)
    if "gamma" in p:
        x = x ** p["gamma"]
    if "sigma" in p:
        salt = int.from_bytes(
            hashlib.blake2b(scenario.encode(), digest_size=4).digest(), "little"
        )
        rng = np.random.default_rng((int(row) * 2654435761 + salt + 4401) & 0xFFFFFFFF)
        read = rng.standard_normal(x.shape, dtype=np.float32) * p["sigma"]
        if p.get("eta", 0):
            read += np.sqrt(p["eta"] * x) * rng.standard_normal(
                x.shape, dtype=np.float32
            )
        x = np.maximum(x + read, 0)
    if "zero" in p:
        x = np.maximum(x - threshold, 0)
    return np.where(valid, np.clip(x, 0, 1), 0).astype(np.float32)


def stable_indices(d, split, n_per_unit, salt):
    result = []
    for unit, g in d[d.split == split].groupby("image_id", sort=True):
        h = [
            int.from_bytes(
                hashlib.blake2b(
                    (str(salt) + "|" + str(unit) + "|" + str(int(c))).encode(),
                    digest_size=8,
                ).digest(),
                "little",
            )
            for c in g.cell_id
        ]
        result.extend(
            g.index.to_numpy()[np.argsort(np.asarray(h, np.uint64))[:n_per_unit]]
        )
    return np.sort(np.asarray(result, np.int64))


def cell_proxies(arr):
    im = arr[:, 0]
    ta = arr[:, 1] > 0
    ot = arr[:, 2] > 0
    tn = ta.sum((1, 2))
    on = ot.sum((1, 2))
    nz = ((im > 0) & ta).sum((1, 2))
    total = (im * ta).sum((1, 2))
    neighbor = (im * ot).sum((1, 2))
    return np.stack(
        [
            nz / np.maximum(tn, 1),
            total / np.maximum(nz, 1),
            neighbor / np.maximum(on, 1),
            on,
        ],
        axis=1,
    ).astype(np.float32)


class Views(cc.Crops):
    def __init__(self, d, scenario="clean", cutoffs=None):
        super().__init__(
            np.zeros((len(d), 56), np.float32), d.ground_truth_binary.to_numpy(int)
        )
        self.scenario = scenario
        self.units = d.image_id.to_numpy()
        self.cutoffs = cutoffs or {}
        self.valid = {}
        if (R / "data/partial_valid.npz").exists():
            with np.load(R / "data/partial_valid.npz") as z:
                self.valid = {int(k): z[k] for k in z.files}

    def __getitem__(self, i):
        x, a, y, idx = super().__getitem__(i)
        p = PARAMS[self.scenario]
        if self.scenario != "clean":
            threshold = self.cutoffs.get(str(self.units[int(i)]), {}).get(
                self.scenario, 0
            )
            xx = x.numpy()
            xx[0] = transform(
                xx[0],
                self.valid.get(int(i), np.ones((96, 96), bool)),
                int(i),
                self.scenario,
                threshold,
            )
            x = torch.from_numpy(xx)
        a = a.clone()
        a[:4] = torch.from_numpy(cell_proxies(x.numpy()[None])[0])
        return x, a, y, idx


def prepare(d):
    # Only rare partial train/internal crops need original-reader support reconstruction.
    partial = d[
        (d.split.isin([cc.TRAIN, cc.DEV])) & (d.crop_context_valid_fraction < 1)
    ]
    valid = {}
    if len(partial):
        sys.path.insert(0, str(cc.OLD / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "old_raw_reader", cc.OLD / "scripts/stage4_review.py"
        )
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        sys.path.pop(0)
        for row in partial.itertuples():
            valid[str(row.Index)] = reader.raw_crop(row)[2]
    np.savez_compressed(R / "data/partial_valid.npz", **valid)
    ds = Views(d)
    bounds = {}
    pixel_records = []
    proxies = np.full((len(d), 4), np.nan, np.float32)
    for unit, g in d[d.split == cc.DEV].groupby("image_id", sort=True):
        baseline = []
        compound = []
        for x, a, y, rows in cc.loader(ds, g.index.to_numpy(), batch=128):
            arr = x.numpy()
            proxies[rows.numpy()] = a.numpy()[:, :4]
            for j, i in enumerate(rows.numpy()):
                mask = arr[j, 1] > 0
                baseline.append(arr[j, 0][mask])
                v = transform(
                    arr[j, 0],
                    ds.valid.get(int(i), np.ones((96, 96), bool)),
                    int(i),
                    "compound",
                    0,
                )
                compound.append(v[mask])
        vals = np.concatenate(baseline)
        comp = np.concatenate(compound)
        assert np.isfinite(vals).all()
        bounds[str(unit)] = {}
        for name, param in SCENARIOS:
            if "zero" not in param:
                continue
            z = comp if name == "compound" else vals
            q = param["zero"]
            threshold = float(np.quantile(z, q, method="higher"))
            bounds[str(unit)][name] = threshold
            pixel_records.append(
                {
                    "image_id": unit,
                    "scenario": name,
                    "target_zero": q,
                    "threshold_fixed_normalized_units": threshold,
                    "finite_target_pixels": len(z),
                    "initial_zero_fraction": float((z == 0).mean()),
                    "achieved_zero_fraction_before_final_float_roundoff": float(
                        (z <= threshold).mean()
                    ),
                    "denominator": "labelled target instance pixels, valid finite support, not all tissue pixels",
                }
            )
        print("PIXEL_BOUNDS", unit, len(vals), flush=True)
    atomic_npy(R / "data/CLEAN_INTERNAL_CELL_PROXIES.npy", proxies)
    atomic_json(R / "data/ZERO_THRESHOLDS.json", bounds)
    pd.DataFrame(pixel_records).to_csv(
        R / "results/STAGE0_PIXEL_DENOMINATORS.csv", index=False
    )
    atomic_json(
        R / "provenance/INPUT_FROZEN.json",
        {
            "feature_sha256": file_sha(cc.OLD / "data/features.parquet"),
            "keys_sha256": cc.canonical_keys_hash(d),
            "base_protocol_sha256": file_sha(BASE / "PROTOCOL.json"),
            "new_protocol_sha256": file_sha(R / "PROTOCOL.json"),
            "partial_support_rows": len(valid),
            "stage0_units": sorted(bounds),
            "all_transform_parameters": SCENARIOS,
        },
    )
    return bounds


def model_score(d, indices, scenario, seed, ledger, tag, cutoffs):
    name = "A_s" + str(seed)
    stem = name + "__" + scenario + "__" + tag
    path = R / "predictions" / (stem + ".npy")
    meta = path.with_suffix(".json")
    idxhash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    weight = BASE / "models" / (name + ".pt")
    modelsha = file_sha(weight)
    expected = json.loads((BASE / "models" / (name + ".json")).read_text())["sha256"]
    assert modelsha == expected
    sig = {
        "input_frozen_sha256": file_sha(R / "provenance/INPUT_FROZEN.json"),
        "indices_sha256": idxhash,
        "model_sha256": modelsha,
        "protocol_sha256": file_sha(R / "PROTOCOL.json"),
        "transform_code_sha256": file_sha(__file__),
        "cutoffs_sha256": file_sha(R / "data/ZERO_THRESHOLDS.json"),
    }
    if meta.exists():
        m = json.loads(meta.read_text())
        a = np.load(path)
        assert (
            m["signature"] == sig
            and file_sha(path) == m["sha256"]
            and a.shape == (len(d),)
            and np.isfinite(a[indices]).all()
        )
        return a, m["seconds"]
    ledger.begin("inference:" + stem)
    started = time.time()
    cc.seed_all(seed)
    model = cc.Network()
    model.load_state_dict(torch.load(weight, map_location="cpu")["state_dict"])
    model.cuda().eval()
    scores = np.full(len(d), np.nan, np.float32)
    proxies = np.full((len(d), 4), np.nan, np.float32)
    ds = Views(d, scenario, cutoffs)
    with torch.no_grad():
        for x, a, y, rows in cc.loader(ds, indices, batch=512):
            ledger.checkpoint()
            proxies[rows.numpy()] = a.numpy()[:, :4]
            scores[rows.numpy()] = (
                torch.softmax(
                    model(x.cuda(non_blocking=True), a.cuda(non_blocking=True), "A"), 1
                )[:, 1]
                .cpu()
                .numpy()
            )
    assert np.isfinite(scores[indices]).all()
    atomic_npy(path, scores)
    if seed == 0:
        atomic_npy(
            R / "data" / ("CELL_PROXIES__" + scenario + "__" + tag + ".npy"), proxies
        )
    seconds = time.time() - started
    atomic_json(
        meta,
        {
            "signature": sig,
            "sha256": file_sha(path),
            "scored_rows": len(indices),
            "full_array_rows": len(d),
            "dtype": str(scores.dtype),
            "seconds": seconds,
            "activity": "inference",
            "scenario": scenario,
            "seed": seed,
        },
    )
    ledger.finish("running")
    del model, ds
    torch.cuda.empty_cache()
    return scores, seconds


def stage0_metrics(d, idx, scenario, scores):
    y = d.ground_truth_binary.to_numpy(int)
    risk = json.loads((BASE / "provenance/STRATA_FROZEN.json").read_text())
    signal = d.nonzero_rate_whole.to_numpy() * d.positive_mean_whole.to_numpy()
    signal = np.where(
        d.repair_whole_support_valid & d.repair_whole_no_nonzero_signal, 0, signal
    )
    weak = (y == 1) & (
        signal <= risk["normalized_observed_target_mean_q25_train_positive"]
    )
    thresholds = {
        x["model"]: x
        for x in json.loads((BASE / "models/THRESHOLDS_FROZEN.json").read_text())[
            "records"
        ]
    }
    out = []
    for model, a in scores.items():
        clean = np.load(BASE / "predictions" / (model + "_all.npy"))
        for unit, g in [("pooled", d.loc[idx])] + list(
            d.loc[idx].groupby("image_id", sort=True)
        ):
            ii = g.index.to_numpy()
            ww = ii[weak[ii]]
            eps = np.finfo(np.float32).eps
            z = np.log(
                np.clip(a[ii], eps, 1 - eps) / (1 - np.clip(a[ii], eps, 1 - eps))
            )
            cz = np.log(
                np.clip(clean[ii], eps, 1 - eps)
                / (1 - np.clip(clean[ii], eps, 1 - eps))
            )
            for mode in ["fixed_0.5", "dev_FPR01_each_WSI"]:
                t = thresholds[model][mode]
                base = cc.metrics(y[ii], clean[ii], t)
                m = cc.metrics(y[ii], a[ii], t)
                out.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "unit": str(unit),
                        "threshold_mode": mode,
                        "threshold": t,
                        "weak_n": len(ww),
                        "weak_recall": float((a[ww] > np.asarray(t, a.dtype)).mean())
                        if len(ww)
                        else np.nan,
                        "flip_rate": float(
                            np.mean(
                                (a[ii] > np.asarray(t, a.dtype))
                                != (clean[ii] > np.asarray(t, clean.dtype))
                            )
                        ),
                        "delta_F1": m["F1"] - base["F1"],
                        "delta_AP": m["AP"] - base["AP"],
                        "AP_retention": m["AP"] / base["AP"],
                        "delta_mean_logit": float(np.mean(z - cz)),
                        **m,
                    }
                )
    return out
