#!/usr/bin/env python
from pathlib import Path
import argparse, json, numpy as np, pandas as pd, torch
from medvoi.utils import load_config,get_device,load_checkpoint,parse_subset_key,save_json
from medvoi.models.value_model import ConditionalValueNet
from medvoi.metrics import gain_metrics, decision_metrics
from medvoi.calibration import gain_calibration
from medvoi.gains import gains_from_utilities


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    cfg=load_config(args.config); out=Path(cfg['output_dir']); device=get_device(cfg)
    util=out/'utilities_test.csv'; gains=out/'gains_test.csv'
    if not gains.exists(): gains_from_utilities(util,gains)
    g=pd.read_csv(gains)
    z=np.load(out/'states_test.npz',allow_pickle=True)
    state_lookup={(str(p),str(s)):v.astype(np.float32) for p,s,v in zip(z['patient_id'],z['subset'],z['state'])}
    ck=torch.load(out/'value_model.pt',map_location='cpu'); state_dim=int(ck['extra']['state_dim']); vc=cfg['value_model']
    model=ConditionalValueNet(state_dim,len(cfg['dataset']['sources']),vc.get('candidate_dim',32),tuple(vc.get('hidden',[256,128])),vc.get('dropout',.1))
    model.load_state_dict(ck['model']); model.to(device).eval()
    mus=[]; sig=[]
    for _,r in g.iterrows():
        st=torch.tensor(state_lookup[(str(r.patient_id),str(r.subset))],dtype=torch.float32,device=device)[None]
        c=torch.tensor([int(r.candidate)],device=device)
        with torch.no_grad(): mu,lv=model(st,c)
        mus.append(float(mu)); sig.append(float(torch.exp(.5*lv)))
    g['pred_gain']=mus; g['pred_sigma']=sig
    beta=float(cfg['policy'].get('beta',0.0)); tau=float(cfg['policy'].get('tau',0.0))
    g['score']=g.pred_gain-beta*g.pred_sigma
    chosen=[]
    for sid,grp in g.groupby('state_id'):
        j=grp.score.idxmax(); a=grp.loc[j,'candidate'] if grp.loc[j,'score']>tau else 'STOP'; chosen.extend([(idx,a) for idx in grp.index])
    m=dict(chosen); g['selected_candidate']=[m[i] for i in g.index]
    gm=gain_metrics(g.pred_gain,g.gain)
    cal,caldev=gain_calibration(g.pred_gain,g.gain,cfg['evaluation'].get('gain_bins',8)); cal.to_csv(out/'gain_calibration_test.csv',index=False)
    dm=decision_metrics(g)
    metrics={**gm,'calibration_mad':caldev,**dm}
    g.to_csv(out/'one_step_predictions.csv',index=False); save_json(metrics,out/'one_step_metrics.json')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
