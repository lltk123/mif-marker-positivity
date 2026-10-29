"""Reusable table inference for a frozen TabM/HGB CD8 model.

Input rows must contain explicit cell keys and every effective raw feature used
by the selected model. Extra metadata is retained, never used as predictors.
"""
import argparse
from common_tabm import *
import joblib
import numpy as np
import pandas as pd
from train_tabm import runtime, build_model, score_members


def predict_table(frame, run_dir, model_id):
    root=Path(run_dir)
    frozen=load_json(root/'FROZEN_TEST_PROTOCOL.json')
    meta=frozen['models'][model_id]
    validate_keys(frame)
    missing=set(meta['features'])-set(frame.columns)
    if missing:
        raise ValueError('Required raw feature columns absent: '+', '.join(sorted(missing)))
    assert sha256(root/'models/preprocessor.joblib')==frozen['signature']['preprocessor']
    prep=joblib.load(root/'models/preprocessor.joblib')
    # Unused columns are placeholders only; independent per-column transforms mean
    # they never affect the model's selected feature columns.
    raw=frame.reindex(columns=prep['features']).to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError('Infinite raw feature values must be audited, not silently replaced')
    transformed=prep['transformer'].transform(prep['imputer'].transform(raw)).astype(np.float32)
    idx=[prep['features'].index(f) for f in meta['features']]
    X=transformed[:,idx]
    if meta['model_family']=='TabM':
        torch,tabm,embeddings=runtime(root)
        path=root/meta['weights_relative_path']
        assert sha256(path)==meta['weights_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model,_=build_model(root,X[:1],saved['config'],torch,tabm,embeddings,bins=saved['bins'])
        model.load_state_dict(saved['state_dict'])
        members=score_members(model,X,torch)
    else:
        path=root/'models'/model_id/'model.joblib'
        assert sha256(path)==meta['weights_sha256']
        members=joblib.load(path).predict_proba(X)[:,1,None]
    out=frame.copy()
    out['model_id']=model_id
    out['score']=members.mean(1)
    out['score_std']=members.std(1)
    out['prediction_fixed_0_5']=out.score>.5
    out['threshold_validation']=meta['threshold_validation']
    out['prediction_validation_selected']=out.score>meta['threshold_validation']
    for i in range(members.shape[1]):
        out[f'submodel_{i:02d}']=members[:,i]
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--model-id',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    frame=pd.read_parquet(a.input) if a.input.endswith('.parquet') else pd.read_csv(a.input,float_precision='round_trip')
    result=predict_table(frame,a.run_dir,a.model_id)
    if a.output.endswith('.parquet'):
        result.to_parquet(a.output,index=False)
    else:
        result.to_csv(a.output,index=False)
