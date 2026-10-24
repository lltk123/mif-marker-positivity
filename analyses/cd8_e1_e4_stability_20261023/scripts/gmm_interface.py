"""Frozen SCIMAP-compatible single-marker GMM fitting and piecewise scoring."""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


def mapping_from_gate(means, gate, cell_ids=None, gmm_info=None):
    x = np.asarray(means, np.float64)
    assert np.isfinite(x).all() and (x >= 0).all()
    z = np.log1p(x)
    closest = float(z[np.abs(z - gate).argmin()])
    lo = z[z <= closest]
    hi = z[z >= closest]
    low = MinMaxScaler(feature_range=(0, 0.5)).fit(lo[:, None])
    high = MinMaxScaler(feature_range=(0.5, 1)).fit(hi[:, None])
    # Frozen MinMaxScaler transforms extrapolate outside training extrema (clip=False).
    state = dict(
        gate_log1p=float(gate),
        nearest_observed_log1p=closest,
        n_fit=len(x),
        log=True,
        clipping_percentiles=[0.01, 99.99],
        clip_low=float(np.percentile(x, 0.01)),
        clip_high=float(np.percentile(x, 99.99)),
        raw_min=float(x.min()),
        raw_max=float(x.max()),
        low=dict(
            scale=float(low.scale_[0]),
            offset=float(low.min_[0]),
            minimum=float(low.data_min_[0]),
            maximum=float(low.data_max_[0]),
        ),
        high=dict(
            scale=float(high.scale_[0]),
            offset=float(high.min_[0]),
            minimum=float(high.data_min_[0]),
            maximum=float(high.data_max_[0]),
        ),
        fixed_mapping_extrapolation="Actual sklearn MinMaxScaler clip=False; no extra clamp",
        gmm_info=gmm_info or {},
    )
    if closest == z.min():
        raise ValueError(
            "Degenerate low mapping needs original tie/index handling; do not silently approximate"
        )
    return state


def score_fixed(means, state):
    """Apply the saved GMM transform and mapping to new cell intensities without refitting."""
    x = np.asarray(means, np.float64)
    out = np.full(x.shape, np.nan)
    v = np.isfinite(x) & (x >= 0)
    z = np.log1p(x[v])
    islo = z <= state["nearest_observed_log1p"]
    lo = state["low"]
    hi = state["high"]
    out[v] = np.where(
        islo, z * lo["scale"] + lo["offset"], z * hi["scale"] + hi["offset"]
    )
    return out


def fit_score(means):
    """Fit the original unsupervised GMM procedure and retain its actual scoring state."""
    x = np.asarray(means, np.float64)
    assert np.isfinite(x).all() and (x >= 0).all()
    bounds = np.percentile(x, [0.01, 99.99])
    data = np.log1p(np.clip(x, *bounds))[:, None]
    gm = GaussianMixture(n_components=3, random_state=0).fit(data)
    assert gm.converged_
    gate = np.sort(gm.means_.ravel())[-2:].mean()
    info = dict(
        means=gm.means_.ravel().tolist(),
        weights=gm.weights_.tolist(),
        covariances=gm.covariances_.ravel().tolist(),
        precisions_cholesky=gm.precisions_cholesky_.ravel().tolist(),
        converged=bool(gm.converged_),
        iterations=int(gm.n_iter_),
        lower_bound=float(gm.lower_bound_),
        configuration=gm.get_params(),
    )
    state = mapping_from_gate(x, gate, gmm_info=info)
    return score_fixed(x, state), state
