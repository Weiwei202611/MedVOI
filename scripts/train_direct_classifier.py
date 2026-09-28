#!/usr/bin/env python
from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from medvoi.utils import load_config,get_device
from medvoi.baselines.direct_classifier import train_classifier

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_config(args.config); out=Path(cfg['output_dir']); device=get_device(cfg)
    g=pd.read_csv(out/'crossfit_gains.csv'); z=np.load(out/'states_train.npz',allow_pickle=True); lookup={(str(p),str(s)):v for p,s,v in zip(z['patient_id'],z['subset'],z['state'])}
    xs=[]; ys=[]; n=len(cfg['dataset']['sources']); stop=n
    for sid,grp in g.groupby('state_id'):
        p,s=sid.split(':',1); best=grp.loc[grp.gain.idxmax()]
        label=int(best.candidate) if float(best.gain)>cfg['policy'].get('tau',0.0) else stop
        if (p,s) in lookup: xs.append(lookup[(p,s)]); ys.append(label)
    model=train_classifier(np.stack(xs),ys,n+1,cfg['baselines'].get('direct_epochs',100),cfg['baselines'].get('direct_lr',1e-3),128,device)
    torch.save({'model':model.state_dict(),'state_dim':len(xs[0]),'n_actions':n+1},out/'direct_classifier.pt'); print('saved direct_classifier.pt')
if __name__=='__main__': main()
