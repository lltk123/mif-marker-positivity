from physics_common import *
from sklearn.metrics import average_precision_score


def ap_independent(y, s):
    order = np.argsort(-s, kind="stable")
    yy = np.asarray(y)[order]
    ss = s[order]
    end = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    tp = np.cumsum(yy)[end]
    prev = np.r_[0, tp[:-1]]
    return float(np.sum((tp - prev) * tp / (end + 1)) / yy.sum())


def main():
    started = time.time()
    d, _ = cc.load_data()
    idx = d.index[d.split == cc.DEV].to_numpy()
    y = d.ground_truth_binary.to_numpy(int)
    checks = []
    tab = pd.read_csv(
        R / "results/STAGE0_FULL_METRICS.csv", float_precision="round_trip"
    )
    manifest = {}
    for scenario in ["clean"] + [s for s, _ in SCENARIOS]:
        arrays = []
        for n in ["A_s0", "A_s1", "A_s2", "ENS_A"]:
            p = (
                BASE / "predictions" / (n + "_all.npy")
                if scenario == "clean"
                else R / "predictions" / (n + "__" + scenario + "__full_internal.npy")
            )
            m = json.loads(p.with_suffix(".json").read_text())
            a = np.load(p)
            assert (
                a.shape == (len(d),)
                and np.isfinite(a[idx]).all()
                and file_sha(p) == m["sha256"]
            )
            manifest[str(p)] = file_sha(p)
            if n == "ENS_A":
                assert np.array_equal(a[idx], np.mean(arrays, axis=0, dtype=np.float32))
            else:
                arrays.append(a[idx])
            ap = ap_independent(y[idx], a[idx])
            rows = tab[
                (tab.model == n) & (tab.scenario == scenario) & (tab.unit == "pooled")
            ]
            for row in rows.itertuples():
                pred = a[idx] > np.asarray(row.threshold, a.dtype)
                tp = int(np.sum(pred & (y[idx] == 1)))
                fp = int(np.sum(pred & (y[idx] == 0)))
                fn = int(np.sum(~pred & (y[idx] == 1)))
                f1 = 2 * tp / (2 * tp + fp + fn)
                assert abs(f1 - row.F1) <= 1e-12 and abs(ap - row.AP) <= 1e-12
                checks.append(
                    {
                        "scenario": scenario,
                        "model": n,
                        "threshold_mode": row.threshold_mode,
                        "independent_AP_abs_error": abs(ap - row.AP),
                        "independent_F1_abs_error": abs(f1 - row.F1),
                        "probability_ensemble_verified": n == "ENS_A",
                    }
                )
        fp = (
            R
            / "data"
            / (
                "CLEAN_INTERNAL_CELL_PROXIES.npy"
                if scenario == "clean"
                else "CELL_PROXIES__" + scenario + "__full_internal.npy"
            )
        )
        ff = np.load(fp)
        assert ff.shape == (len(d), 4) and np.isfinite(ff[idx]).all()
        assert ((ff[idx, 0] >= 0) & (ff[idx, 0] <= 1)).all()
        assert (ff[idx, 1:3] >= 0).all() and (ff[idx, 1:3] <= 1.000001).all()
        manifest[str(fp)] = file_sha(fp)
    pd.DataFrame(checks).to_csv(R / "results/STAGE0_INDEPENDENT_AUDIT.csv", index=False)
    atomic_json(R / "provenance/STAGE0_VALIDATED_ARTIFACT_HASHES.json", manifest)
    atomic_json(
        R / "results/STAGE0_AUDIT_COMPLETE.json",
        {
            "checks": len(checks),
            "max_AP_error": max(x["independent_AP_abs_error"] for x in checks),
            "max_F1_error": max(x["independent_F1_abs_error"] for x in checks),
            "seconds_cpu_wall": time.time() - started,
            "full_internal_n": len(idx),
        },
    )
    print("STAGE0_AUDIT_PASS", len(checks), flush=True)


if __name__ == "__main__":
    main()
