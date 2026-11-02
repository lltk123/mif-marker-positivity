"""Frozen calibration utilities; cohort IDs never form model features."""
from physics_common import *
from scipy.special import logit, expit
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from condition_evaluate import strata


def logits(s):
    return logit(
        np.clip(np.asarray(s, float), np.finfo(float).eps, 1 - np.finfo(float).eps)
    )


def intercept(z, y):
    if len(np.unique(y)) < 2:
        return 0.0, True
    return float(brentq(lambda b: float(np.sum(expit(z + b) - y)) + b, -64, 64)), False


def shifted(s, b):
    return np.asarray(s, float).copy() if b == 0 else expit(logits(s) + b)


def verify_scores(d, name):
    p = BASE / "predictions" / (name + "_all.npy")
    m = json.loads(p.with_suffix(".json").read_text())
    a = np.load(p)
    assert file_sha(p) == m["sha256"] and a.shape == (len(d),) and np.isfinite(a).all()
    assert m["signatures"]["four_keys_sha256"] == cc.canonical_keys_hash(d)
    if name.startswith("ENS_"):
        assert np.array_equal(
            a,
            np.mean(
                [verify_scores(d, name[4:] + "_s" + str(k)) for k in range(3)],
                axis=0,
                dtype=np.float32,
            ),
        )
    return a


def reserved(d):
    table = pd.read_csv(BASE / "results/CALIBRATION_RESERVED_INSTANCES.csv")
    test = d[d.split == cc.TEST]
    out = {}
    for rep in [101, 102, 103]:
        byunit = {}
        hold = np.zeros(len(d), bool)
        hold[test.index] = True
        for uid, g in test.groupby("image_id", sort=True):
            prior = table[
                (table.replicate == rep) & (table.image_id == uid)
            ].sort_values("sampling_rank")
            ix = prior.row_id.to_numpy(int)
            assert len(ix) == 100 and len(np.unique(ix)) == 100
            assert np.array_equal(
                d.loc[ix, "cell_id"].to_numpy(), prior.cell_id.to_numpy()
            )
            assert (d.loc[ix, "image_id"] == uid).all()
            byunit[uid] = ix
            hold[ix] = False
        assert hold.sum() == 433280 and sum(map(len, byunit.values())) == 4200
        out[rep] = (byunit, hold)
    return out


def weak_mask(d):
    return strata(d)["weak_observed_positive"]


def aggregate(d, idx, s, t, extra):
    y = d.ground_truth_binary.to_numpy(int)
    weak = weak_mask(d)
    rows = []
    groups = (
        [("pooled", "all", idx)]
        + [
            ("source", str(k), g.index.to_numpy())
            for k, g in d.loc[idx].groupby("context", sort=True)
        ]
        + [
            ("core", str(k), g.index.to_numpy())
            for k, g in d.loc[idx].groupby("image_id", sort=True)
        ]
    )
    for level, unit, ii in groups:
        jj = ii[weak[ii]]
        rows.append(
            {
                **extra,
                "level": level,
                "unit": unit,
                "threshold": float(t),
                "weak_n": len(jj),
                "weak_recall": float((s[jj] > t).mean()) if len(jj) else np.nan,
                **cc.metrics(y[ii], s[ii], t),
            }
        )
    cores = [x for x in rows if x["level"] == "core"]
    pooled = rows[0]
    f = np.asarray([x["F1"] for x in cores])
    pooled["core_F1_median"] = float(np.nanmedian(f))
    pooled["core_F1_IQR"] = float(np.nanquantile(f, 0.75) - np.nanquantile(f, 0.25))
    pooled["core_prevalence_MAE"] = float(
        np.mean([x["positive_fraction_absolute_error"] for x in cores])
    )
    pooled["n_cores"] = len(cores)
    return rows


def anchor(z, features):
    nz, mu, neighbor, area = features.T
    mask = (
        (nz <= np.quantile(nz, 0.2))
        & (neighbor <= np.quantile(neighbor, 0.5))
        & (area > 0)
    )
    n = int(mask.sum())
    return (float(np.quantile(z[mask], 0.95)) if n >= 200 else None), n


def mixture(z, features):
    eligible = np.flatnonzero(features[:, 0] > 0)
    if len(eligible) < 200:
        return None, {"accepted": False, "reason": "nonzero_rows<200"}
    if len(eligible) > 20000:
        eligible = np.sort(
            np.random.default_rng(4401).choice(eligible, 20000, replace=False)
        )
    gm = GaussianMixture(
        n_components=2, reg_covar=0.25, random_state=4401, n_init=1, max_iter=200
    ).fit(z[eligible, None])
    means = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    order = np.argsort(means)
    sep = float(abs(means[1] - means[0]) / np.sqrt(np.mean(sd**2)))
    ok = bool(gm.converged_ and gm.weights_.min() >= 0.05 and sep >= 1)
    k = order[0]
    return (float(means[k] + 2.3263478740408408 * sd[k]) if ok else None), {
        "accepted": ok,
        "weights": gm.weights_.tolist(),
        "means": means.tolist(),
        "sd": sd.tolist(),
        "separation_pooledSD": sep,
        "fit_n": len(eligible),
        "reason": "accepted" if ok else "convergence_weight_or_separation_gate",
    }


def summary(z, features):
    nz, mu, neighbor, area = features.T
    v = mu[nz > 0]
    ratio = (
        float(np.median(v) / np.quantile(v, 0.95))
        if len(v) and np.quantile(v, 0.95) > 0
        else 0.0
    )
    return np.array(
        [float(np.mean(1 - nz)), ratio, float(np.mean((z >= -4) & (z <= 4)))]
    )


def unit_threshold(y, s):
    neg = np.sort(s[np.asarray(y) == 0])
    k = int(np.floor(0.01 * len(neg)))
    return float(neg[-k - 1])


def baseline_thresholds():
    return {
        r["model"]: r
        for r in json.loads((BASE / "models/THRESHOLDS_FROZEN.json").read_text())[
            "records"
        ]
    }
