#!/usr/bin/env python
from pathlib import Path
import argparse, copy
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from medvoi.utils import load_config,set_seed,get_device,ensure_dir
from medvoi.models import build_task_model
from medvoi.engine import train_task_model,evaluate_all_states
from medvoi.gains import gains_from_utilities

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    cfg=load_config(args.config); set_seed(cfg['training'].get('seed',2026)); device=get_device(cfg)
    out=Path(cfg['output_dir']); cf=out/'crossfit'; ensure_dir(cf)
    df=pd.read_csv(cfg['dataset']['manifest']); train=df[df.split.astype(str)=='train'].copy().reset_index(drop=True)
    n=cfg.get('crossfit',{}).get('folds',5); kf=KFold(n_splits=n,shuffle=True,random_state=cfg['training'].get('seed',2026))
    all_u=[]
    for fold,(tr,ho) in enumerate(kf.split(train)):
        tmp=train.copy(); tmp['split']='train_cf'; tmp.loc[ho,'split']='test_cf'
        # Add original validation rows only if requested; cross-fit training uses no test patients.
        path=cf/f'fold_{fold}_manifest.csv'; tmp.to_csv(path,index=False)
        c=copy.deepcopy(cfg); c['dataset']['manifest']=str(path); c['training']['epochs']=c.get('crossfit',{}).get('epochs',c['training'].get('epochs',50))
        model=build_task_model(c); model,_=train_task_model(model,c,device,'train_cf',None)
        u,_=evaluate_all_states(model,c,'test_cf',device,include_state=False); u['fold']=fold; all_u.append(u)
        print(f'fold {fold+1}/{n}: {len(u)} state utilities')
    util=pd.concat(all_u,ignore_index=True); util['split']='train'; util.to_csv(out/'crossfit_utilities.csv',index=False)
    gains_from_utilities(out/'crossfit_utilities.csv',out/'crossfit_gains.csv')
    print(f'Saved {out/"crossfit_gains.csv"}')
if __name__=='__main__': main()
