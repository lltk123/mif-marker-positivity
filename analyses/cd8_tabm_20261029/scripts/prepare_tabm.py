"""Audit existing features and fit all preprocessing using original training labels only."""
import argparse
import importlib.util
import time
import warnings
from common_tabm import *
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer


def main(root):
    root = Path(root)
    project = root.parents[1]
    old = project / 'analyses/cd8_morphospatial_20261023'
    source = old / 'PhenoBIC_CD8_features.parquet'
    labels = project / 'analyses/cd8_e1_e4_stability_20261023/audit/cells.parquet'
    dictionary = old / 'FEATURE_DICTIONARY.csv'
    spec = importlib.util.spec_from_file_location('source_mif_features', old / 'scripts/mif_features.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    requested = {'A': mod.WHOLE, 'B': mod.WHOLE + mod.MORPH, 'C': mod.WHOLE + mod.MORPH + [x for x in mod.FEATURES if x not in mod.WHOLE + mod.MORPH]}
    assert [len(requested[x]) for x in 'ABC'] == [8, 27, 90]
    manifest = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p)) for p in [source, labels, dictionary, old / 'scripts/mif_features.py']]
    saved = root / 'data/DATA_READY.json'
    if saved.exists():
        prior = load_json(saved)
        assert prior['inputs'] == manifest and prior['preparation_sha256'] == sha256(__file__)
        print('Compatible prepared data already exists', flush=True)
        return
    data = pd.read_parquet(source)
    assert not data.duplicated(KEYS[:3]).any()
    ann = pd.read_parquet(labels, columns=KEYS[:3] + ['ground_truth_original', 'ground_truth_binary', 'GMM_score', 'GMM_prediction', 'pixel_count', 'mean_CD8'])
    data = data.merge(ann, on=KEYS[:3], how='outer', validate='one_to_one', indicator=True)
    assert data['_merge'].eq('both').all()
    data = data.drop(columns='_merge')
    data['marker'] = 'CD8'
    data = data.sort_values(KEYS).reset_index(drop=True)
    validate_keys(data)
    assert len(data) == 1817783
    assert np.array_equal(data.observed_pixel_count, data.pixel_count)
    assert np.allclose(data.observed_mean_CD8, data.mean_CD8, rtol=1e-10, atol=1e-10)
    assert data.mask_type.eq('expanded_nucleus').all()
    labeled = data.ground_truth_binary.isin([0, 1])
    train = labeled & data.split.eq(TRAIN)
    valid = labeled & data.split.eq(VALID)
    expected = {TRAIN:101057, VALID:539132, TEST:437480}
    counts = data.loc[labeled].groupby('split').size().to_dict()
    assert counts == expected, counts
    assert data.loc[train, 'image_id'].nunique() == data.loc[valid, 'image_id'].nunique() == 3
    hashes = []
    for split, part in data.groupby('split', sort=True):
        for selection, frame in [('all', part), ('labelled', part.loc[part.ground_truth_binary.isin([0, 1])])]:
            dest = root / 'data' / f'keys_{split}_{selection}.csv.gz'
            frame[KEYS].to_csv(dest, index=False, compression='gzip')
            hashes.append(dict(split=split, selection=selection, n=len(frame), key_sha256=key_hash(frame), file_sha256=sha256(dest)))
    pd.DataFrame(hashes).to_csv(root / 'results/key_hashes.csv', index=False)
    # Raw inputs remain unchanged; the imputation and transform are explicit fitted model components.
    raw = data[requested['C']].to_numpy(dtype=np.float64)
    assert not np.isinf(raw).any(), 'Unexpected infinity in saved features'
    tr = raw[train.to_numpy()]
    reasons = {}
    for j, name in enumerate(requested['C']):
        finite = tr[np.isfinite(tr[:, j]), j]
        if len(finite) == 0:
            reasons[name] = 'all_missing_in_training'
        elif np.min(finite) == np.max(finite):
            reasons[name] = 'constant_observed_training_values'
    active = [x for x in requested['C'] if x not in reasons]
    active_idx = [requested['C'].index(x) for x in active]
    imputer = SimpleImputer(strategy='median')
    xtrain = imputer.fit_transform(tr[:, active_idx])
    # Official example: noisy training-only observations fit the quantile map; transform clean values.
    noise_rng = np.random.default_rng(0)
    noisy_fit = xtrain + noise_rng.normal(0, 1e-5, xtrain.shape).astype(xtrain.dtype)
    transformer = QuantileTransformer(n_quantiles=max(min(len(xtrain)//30, 1000), 10), output_distribution='normal', subsample=10**9, random_state=0)
    transformer.fit(noisy_fit)
    joblib.dump(dict(imputer=imputer, transformer=transformer, features=active, fit_split=TRAIN), root / 'models/preprocessor.joblib')
    X = np.lib.format.open_memmap(root / 'data/X.npy', mode='w+', dtype=np.float32, shape=(len(data), len(active)))
    for start in range(0, len(data), 65536):
        X[start:start+65536] = transformer.transform(imputer.transform(raw[start:start+65536, active_idx])).astype(np.float32)
    assert np.isfinite(X).all()
    X.flush()
    effective = {g:[f for f in requested[g] if f in active] for g in requested}
    groups = {g:dict(requested=requested[g], effective=effective[g], matrix_indices=[active.index(f) for f in effective[g]], excluded={f:reasons[f] for f in requested[g] if f in reasons}) for g in requested}
    save_json(root / 'FEATURE_SETS.json', groups)
    pd.DataFrame([dict(feature_set=g, position=i, feature=f, included=f in active, exclusion_reason=reasons.get(f,'')) for g in requested for i,f in enumerate(requested[g])]).to_csv(root / 'results/feature_manifest.csv', index=False)
    if dictionary.exists():
        (root / 'FEATURE_DICTIONARY.csv').write_bytes(dictionary.read_bytes())
    rows = []
    for split, part in data.groupby('split', sort=True):
        for name in requested['C']:
            a = part[name].to_numpy(float)
            rows.append(dict(split=split, feature=name, n=len(a), n_missing=int(np.isnan(a).sum()), missing_rate=float(np.isnan(a).mean()), n_zero=int((a==0).sum()), excluded_reason=reasons.get(name,'')))
    pd.DataFrame(rows).to_csv(root / 'results/feature_availability.csv', index=False)
    training = data.loc[train]
    pos = training.loc[training.ground_truth_binary.eq(1)]
    contact = training.loc[training.neighbor_edge_area_fraction.gt(0)]
    risk = dict(weak_positive_whole_mean_q25=float(pos.whole_mean.quantile(.25)), contact_bright_diff_q75=float(contact.neighbor_edge_mean_diff.quantile(.75)), local_reference_q75=float(training.local_reference_median.quantile(.75)), sector_cv_q75=float(training.sector_signal_cv.quantile(.75)), fit_split=TRAIN, interpretation='Image-property proxies, not evidence of causal contamination', contact_comparison='strict >', weak_comparison='<=')
    save_json(root / 'data/risk_strata.json', risk)
    data.to_parquet(root / 'data/cells.parquet', index=False, compression='zstd')
    np.save(root / 'data/train_indices.npy', np.flatnonzero(train.to_numpy()))
    np.save(root / 'data/validation_indices.npy', np.flatnonzero(valid.to_numpy()))
    info = dict(inputs=manifest, preparation_sha256=sha256(__file__), cells=len(data), labeled_counts=counts, mask_type='expanded_nucleus', patient_ids_available=sorted(str(x) for x in data.patient_id.dropna().unique()), grouping='Original WSI; no random cell split; cores are not independent patients', preprocessing_fit_count=int(train.sum()), requested_features=len(requested['C']), effective_features=len(active), excluded=reasons, key_hash=key_hash(data), cells_sha256=sha256(root / 'data/cells.parquet'), matrix_sha256=sha256(root / 'data/X.npy'), preprocessor_sha256=sha256(root / 'models/preprocessor.joblib'), train_indices_sha256=sha256(root / 'data/train_indices.npy'), validation_indices_sha256=sha256(root / 'data/validation_indices.npy'))
    save_json(saved, info)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    main(p.parse_args().run_dir)

