"""Bounded, paired continuation experiments; immutable old checkpoints reused."""
from physics_common import *
from execution_guards import atomic_checkpoint
from scipy.ndimage import distance_transform_edt
from torch import nn

VARIANTS = ["control", "physics", "sdf", "ranking"]


def geometry(x):
    ta = x[1] > 0
    nb = x[2] > 0
    inside = distance_transform_edt(ta)
    outside = distance_transform_edt(~ta)
    sdf = np.where(ta, inside - 1, -outside)
    nd = distance_transform_edt(~nb) if nb.any() else np.full(ta.shape, np.inf)
    ring = (inside > 0) & (inside <= 3) & (nd > 3)
    return np.stack([x[0], np.clip(sdf / 12, -1, 1), x[2], ring]).astype(np.float32)


def oriented(x, row, seed, epoch):
    h = (
        int(row) * 2654435761 + int(epoch) * 2246822519 + int(seed) * 3266489917
    ) & 0xFFFFFFFF
    x = np.rot90(x, int(h % 4), axes=(1, 2))
    if (h >> 2) & 1:
        x = x[:, :, ::-1]
    if (h >> 3) & 1:
        x = x[:, ::-1, :]
    return np.ascontiguousarray(x)


def eligible_physics(original, transformed, target, choice):
    if not target.any():
        return False
    a = original[target]
    b = transformed[target]
    nz = int((a > 0).sum())
    if (
        nz == 0
        or int((b > 0).sum()) < 0.5 * nz
        or float(b.sum()) < 0.5 * float(a.sum())
        or float((b >= 1).mean()) > 0.02
    ):
        return False
    if choice == "noise_low":
        signal_power = float(np.mean(a.astype(float) ** 2))
        noise_power = float(np.mean(0.002 * a.astype(float) + 0.005**2))
        if signal_power < 9 * noise_power:
            return False
    return True


class ContinuationDataset(cc.Crops):
    def __init__(
        self,
        d,
        variant,
        training=False,
        seed=0,
        cutoffs=None,
        scenario="clean",
        stage0_cutoffs=None,
    ):
        super().__init__(
            np.zeros((len(d), 56), np.float32),
            d.ground_truth_binary.to_numpy(int),
            False,
            seed,
        )
        self.d = d
        self.variant = variant
        self.training = training
        self.cutoffs = cutoffs or {}
        self.scenario = scenario
        self.stage0_cutoffs = stage0_cutoffs or {}
        self.valid = {}
        with np.load(R / "data/partial_valid.npz") as z:
            self.valid = {int(k): z[k] for k in z.files}

    def __getitem__(self, i):
        xx, a, y, idx = super().__getitem__(i)
        x = xx.numpy()
        aug = x.copy()
        valid = self.valid.get(int(i), np.ones((96, 96), bool))
        eligible = False
        if self.scenario != "clean":
            x[0] = transform(
                x[0],
                valid,
                i,
                self.scenario,
                self.stage0_cutoffs.get(str(self.d.image_id.iloc[i]), {}).get(
                    self.scenario, 0
                ),
            )
            aug = x.copy()
        if self.training and self.variant == "physics":
            h = (
                int(i) * 2654435761 + self.epoch * 2246822519 + self.seed * 3266489917
            ) & 0xFFFFFFFF
            choice = ["gain_0.75", "gain_1.5", "noise_low", "zero_0.7"][h % 4]
            threshold = self.cutoffs.get(str(self.d.image_id.iloc[i]), 0)
            aug[0] = transform(x[0], valid, i, choice, threshold)
            target = x[1] > 0
            oldnz = int(((x[0] > 0) & target).sum())
            newnz = int(((aug[0] > 0) & target).sum())
            sat = float((aug[0][target] >= 1).mean()) if target.any() else 1.0
            eligible = eligible_physics(x[0], aug[0], target, choice)
            if not eligible:
                aug = x.copy()
        if self.variant == "sdf":
            x = geometry(x)
            aug = x.copy()
        if self.training:
            x = oriented(x, i, self.seed, self.epoch)
            aug = oriented(aug, i, self.seed, self.epoch)
        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(aug.copy()),
            a,
            y,
            idx,
            eligible,
        )


def network(variant, state):
    m = cc.Network()
    if variant == "sdf":
        rng_state = torch.get_rng_state()
        conv = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        torch.set_rng_state(rng_state)
        m.encoder[0] = conv
        state = {k: v.clone() for k, v in state.items()}
        w = state["encoder.0.weight"]
        state["encoder.0.weight"] = torch.cat([w, torch.zeros_like(w[:, :1])], dim=1)
    m.load_state_dict(state)
    return m


def train_cutoffs(d):
    path = R / "data/TRAIN_FLOOR70.json"
    if path.exists():
        return json.loads(path.read_text())
    ds = Views(d)
    bounds = {}
    for uid, g in d[d.split == cc.TRAIN].groupby("image_id", sort=True):
        vals = []
        for x, a, y, ix in cc.loader(ds, g.index.to_numpy(), batch=128):
            arr = x.numpy()
            vals.append(arr[:, 0][arr[:, 1] > 0])
        z = np.concatenate(vals)
        bounds[str(uid)] = float(np.quantile(z, 0.7, method="higher"))
    atomic_json(path, bounds)
    return bounds


def signature(parent, variant, fitidx):
    return {
        "protocol_sha256": file_sha(R / "PROTOCOL.json"),
        "continuation_plan_sha256": file_sha(R / "TRAINING_CONDITIONAL_PLAN.json"),
        "eligibility_amendment_sha256": file_sha(
            R / "TRAINING_ELIGIBILITY_AMENDMENT.json"
        ),
        "training_code_sha256": file_sha(__file__),
        "parent_checkpoint_sha256": file_sha(BASE / "models" / (parent + ".pt")),
        "fit_indices_sha256": hashlib.sha256(
            np.asarray(fitidx, dtype="<i8").tobytes()
        ).hexdigest(),
        "variant": variant,
    }


def fit(
    d, variant, parent, name, seed, fitidx, n, epochs, start_epoch, ledger, cutoffs
):
    path = R / "models" / (name + ".pt")
    meta = path.with_suffix(".json")
    sig = signature(parent, variant, fitidx)
    if meta.exists():
        q = json.loads(meta.read_text())
        assert q["signature"] == sig and q["sha256"] == file_sha(path)
        return q
    assert not path.exists(), "orphan checkpoint requires audit, not overwrite"
    expected = json.loads((BASE / "models" / (parent + ".json")).read_text())["sha256"]
    assert expected == sig["parent_checkpoint_sha256"]
    assert (d.loc[fitidx, "split"] == cc.TRAIN).all()
    ledger.begin("training:" + name)
    started = time.time()
    cc.seed_all(seed)
    state = torch.load(BASE / "models" / (parent + ".pt"), map_location="cpu")[
        "state_dict"
    ]
    m = network(variant, state).cuda()
    initial = cc.initial_hash(m)
    ds = ContinuationDataset(d, variant, True, seed, cutoffs)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    yall = d.ground_truth_binary.to_numpy(int)
    signal = d.nonzero_rate_whole.to_numpy() * d.positive_mean_whole.to_numpy()
    signal = np.where(
        d.repair_whole_support_valid & d.repair_whole_no_nonzero_signal, 0, signal
    )
    pfit = fitidx[yall[fitidx] == 1]
    weak_cut = float(np.nanquantile(signal[pfit], 0.25))
    area = d.audit_raw_neighbor_edge_area_fraction.to_numpy()
    difference = d.audit_raw_neighbor_edge_mean_diff.to_numpy()
    nfit = fitidx[
        (yall[fitidx] == 0) & (area[fitidx] > 0) & np.isfinite(difference[fitidx])
    ]
    hard_cut = float(np.quantile(difference[nfit], 0.75))
    weak = (yall == 1) & (signal <= weak_cut)
    hard = (yall == 0) & (area > 0) & (difference > hard_cut)
    log = []
    atomic_json(
        R / "provenance" / (name + "_initialization.json"),
        {
            "seed": seed,
            "variant": variant,
            "parent": parent,
            "initial_hash": initial,
            "parent_sha256": expected,
            "recorded_before_optimizer_steps": True,
            "fit_units": sorted(d.loc[fitidx, "image_id"].unique()),
            "weak_cut_fit_only": weak_cut,
            "neighbor_negative_cut_fit_only": hard_cut,
            "nominal_parameters": sum(p.numel() for p in m.parameters()),
        },
    )
    for epoch in range(start_epoch, start_epoch + epochs):
        ds.epoch = epoch
        sample = cc.sample_epoch(d, fitidx, n, seed, epoch)
        atomic_json(
            R / "provenance" / (name + "_epoch" + str(epoch) + ".json"),
            {
                "rows": len(sample),
                "indices_sha256": hashlib.sha256(
                    sample.astype("<i8").tobytes()
                ).hexdigest(),
                "fit_units": sorted(d.loc[fitidx, "image_id"].unique()),
                "recorded_before_epoch": True,
            },
        )
        m.train()
        losses = []
        pair_count = 0
        eligible_count = 0
        epstart = time.time()
        for x, aug, a, y, rows, eligible in cc.loader(ds, sample, batch=128):
            ledger.checkpoint()
            opt.zero_grad(set_to_none=True)
            x = x.cuda(non_blocking=True)
            a = a.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            out = m(x, a, "A")
            ce = nn.functional.cross_entropy(out, y, reduction="none")
            loss = ce.mean()
            if variant == "physics":
                other = m(aug.cuda(non_blocking=True), a, "A")
                ok = eligible.cuda(non_blocking=True)
                z = out[:, 1] - out[:, 0]
                za = other[:, 1] - other[:, 0]
                lc = (
                    0.5 * ce
                    + 0.5 * nn.functional.cross_entropy(other, y, reduction="none")
                    + 0.1 * nn.functional.smooth_l1_loss(z, za, reduction="none")
                )
                loss = torch.where(ok, lc, ce).mean()
                eligible_count += int(eligible.sum())
            elif variant == "ranking":
                ix = rows.numpy()
                wi = np.flatnonzero(weak[ix])
                hi = np.flatnonzero(hard[ix])
                if len(wi) and len(hi):
                    z = out[:, 1] - out[:, 0]
                    pair = torch.relu(1 - (z[wi, None] - z[hi][None, :])) ** 2
                    loss = loss + 0.1 * pair.mean()
                    pair_count += len(wi) * len(hi)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        item = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "seconds": time.time() - epstart,
            "physics_eligible_rows": eligible_count,
            "ranking_pairs": pair_count,
        }
        log.append(item)
        print("CONTINUATION_EPOCH", name, json.dumps(item), flush=True)
    atomic_checkpoint(
        path,
        {
            "state_dict": m.cpu().state_dict(),
            "variant": variant,
            "seed": seed,
            "parent": parent,
            "signature": sig,
        },
    )
    rec = {
        "signature": sig,
        "sha256": file_sha(path),
        "seconds": time.time() - started,
        "initial_hash": initial,
        "epochs": epochs,
        "rows_per_epoch": n,
        "nominal_parameters": sum(p.numel() for p in m.parameters()),
    }
    atomic_json(meta, rec)
    pd.DataFrame(log).to_csv(R / "results" / (name + "_epochs.csv"), index=False)
    ledger.finish("running")
    del m, opt, ds
    torch.cuda.empty_cache()
    return rec


def predict_new(d, name, variant, idx, ledger, tag, scenario="clean"):
    path = R / "predictions" / (name + "__" + tag + ".npy")
    meta = path.with_suffix(".json")
    modelsha = file_sha(R / "models" / (name + ".pt"))
    idxhash = hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()
    sig = {
        "model_sha256": modelsha,
        "indices_sha256": idxhash,
        "scenario": scenario,
        "code_sha256": file_sha(__file__),
    }
    if meta.exists():
        old = json.loads(meta.read_text())
        a = np.load(path)
        assert (
            old["signature"] == sig
            and file_sha(path) == old["sha256"]
            and a.shape == (len(d),)
            and np.isfinite(a[idx]).all()
        )
        return a
    ledger.begin("inference:" + name + "__" + tag)
    started = time.time()
    cc.seed_all(0)
    ck = torch.load(R / "models" / (name + ".pt"), map_location="cpu")
    m = cc.Network()
    if variant == "sdf":
        m.encoder[0] = nn.Conv2d(4, 32, 3, stride=2, padding=1)
    m.load_state_dict(ck["state_dict"])
    m.cuda().eval()
    ds = ContinuationDataset(
        d,
        variant,
        scenario=scenario,
        stage0_cutoffs=json.loads((R / "data/ZERO_THRESHOLDS.json").read_text()),
    )
    aout = np.full(len(d), np.nan, np.float32)
    with torch.no_grad():
        for x, aug, a, y, ii, eligible in cc.loader(ds, idx, batch=512):
            ledger.checkpoint()
            aout[ii.numpy()] = (
                torch.softmax(
                    m(x.cuda(non_blocking=True), a.cuda(non_blocking=True), "A"), 1
                )[:, 1]
                .cpu()
                .numpy()
            )
    assert np.isfinite(aout[idx]).all()
    atomic_npy(path, aout)
    atomic_json(
        meta,
        {
            "signature": sig,
            "sha256": file_sha(path),
            "scored_n": len(idx),
            "seconds": time.time() - started,
        },
    )
    ledger.finish("running")
    del m, ds
    torch.cuda.empty_cache()
    return aout
