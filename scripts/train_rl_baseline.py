#!/usr/bin/env python
from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from medvoi.utils import load_config,get_device,parse_subset_key,subset_key
from medvoi.data.subsets import add_candidate,candidate_indices
from medvoi.baselines.rl import OfflineDQN

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_config(args.config); out=Path(cfg['output_dir']); device=get_device(cfg)
    g=pd.read_csv(out/'crossfit_gains.csv'); z=np.load(out/'states_train.npz',allow_pickle=True); lookup={(str(p),str(s)):v.astype(np.float32) for p,s,v in zip(z['patient_id'],z['subset'],z['state'])}
    n=len(cfg['dataset']['sources']); stop=n; transitions=[]
    for _,r in g.iterrows():
        pid,s=str(r.patient_id),str(r.subset); k=(pid,s)
        if k not in lookup: continue
        mask=parse_subset_key(s); ns=add_candidate(mask,int(r.candidate)); nk=(pid,subset_key(ns)); next_state=lookup.get(nk,lookup[k]); valid=np.zeros(n+1,dtype=bool); valid[candidate_indices(ns)]=True; valid[stop]=True
        transitions.append((lookup[k],int(r.candidate),float(r.gain),next_state,float(ns.sum()==n),valid))
    # Explicit STOP transitions with zero reward.
    for (pid,s),st in lookup.items():
        mask=parse_subset_key(s); valid=np.zeros(n+1,dtype=bool); valid[stop]=True
        transitions.append((st,stop,0.0,st,1.0,valid))
    agent=OfflineDQN(len(transitions[0][0]),n+1,device,gamma=cfg['baselines'].get('rl_gamma',.95),lr=cfg['baselines'].get('rl_lr',1e-3)).fit(transitions,cfg['baselines'].get('rl_epochs',30))
    torch.save({'model':agent.q.state_dict(),'state_dim':len(transitions[0][0]),'n_actions':n+1},out/'rl_qnet.pt'); print('saved rl_qnet.pt')
if __name__=='__main__': main()
