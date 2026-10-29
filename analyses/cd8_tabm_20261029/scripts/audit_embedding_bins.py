"""Audit actual numerical embedding bins in the frozen official TabM weights."""
import argparse
from pathlib import Path
from common_tabm import *
from train_tabm import runtime
import numpy as np
import pandas as pd


def main(root):
    root=Path(root)
    torch,_,_=runtime(root)
    frozen=load_json(root/'FROZEN_TEST_PROTOCOL.json')
    rows=[]
    shared={}
    for model_id,meta in frozen['models'].items():
        if meta['model_family']!='TabM':
            continue
        path=root/meta['weights_relative_path']
        assert sha256(path)==meta['weights_sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['features']==meta['features']
        for feature,bins in zip(meta['features'],checkpoint['bins']):
            values=bins.numpy()
            assert np.isfinite(values).all() and np.all(np.diff(values)>0)
            if feature in shared:
                assert np.array_equal(np.asarray(shared[feature],dtype=values.dtype),values), 'Same feature has different bins'
            else:
                shared[feature]=values.tolist()
            rows.append(dict(model_id=model_id,feature_set=meta['feature_set'],seed=meta['seed'],feature=feature,requested_intervals=48,actual_intervals=len(values)-1,minimum_edge=float(values[0]),maximum_edge=float(values[-1]),fitting_split=TRAIN,training_cells=101057))
    pd.DataFrame(rows).to_csv(root/'results/embedding_bins.csv',index=False)
    save_json(root/'models/embedding_bin_edges.json',dict(fit_split=TRAIN,feature_edges=shared,all_shared_features_have_identical_bins=True))
    print('Embedding bins verified for',len(rows),'model-feature combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    main(p.parse_args().run_dir)
