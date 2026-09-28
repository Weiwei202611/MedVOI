#!/usr/bin/env python
from pathlib import Path
import argparse, json, numpy as np, pandas as pd, torch
from medvoi.utils import load_config,get_device,parse_subset_key,subset_key,save_json
from medvoi.models.value_model import ConditionalValueNet
from medvoi.acquisition import select_action
from medvoi.data.subsets import add_candidate


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_config(args.config); out=Path(cfg['output_dir']); device=get_device(cfg)
    util=pd.read_csv(out/'utilities_test.csv'); z=np.load(out/'states_test.npz',allow_pickle=True)
    state_lookup={(str(p),str(s)):v.astype(np.float32) for p,s,v in zip(z['patient_id'],z['subset'],z['state'])}; util_lookup={(str(r.patient_id),str(r.subset)):float(r.utility) for _,r in util.iterrows()}
    ck=torch.load(out/'value_model.pt',map_location='cpu'); vc=cfg['value_model']; model=ConditionalValueNet(int(ck['extra']['state_dim']),len(cfg['dataset']['sources']),vc.get('candidate_dim',32),tuple(vc.get('hidden',[256,128])),vc.get('dropout',.1)); model.load_state_dict(ck['model']); model.to(device).eval()
    init=np.asarray(cfg['policy'].get('initial_subset',[1]+[0]*(len(cfg['dataset']['sources'])-1)),dtype=int)
    tau=float(cfg['policy'].get('tau',0.0)); beta=float(cfg['policy'].get('beta',0.0)); costs=cfg['policy'].get('costs'); lam=float(cfg['policy'].get('lambda_cost',0.0))
    rows=[]; finals=[]
    for pid in sorted({str(x) for x in util.patient_id.unique()}):
        mask=init.copy(); step=0
        while True:
            k=subset_key(mask); st=state_lookup[(pid,k)]; q=util_lookup[(pid,k)]
            dec=select_action(model,torch.tensor(st),mask,tau,beta,costs,lam,device)
            rows.append({'patient_id':pid,'step':step,'subset':k,'task_utility':q,'action':'STOP' if dec.stop else cfg['dataset']['sources'][dec.action],'pred_gain':dec.predicted_gain,'pred_sigma':dec.predicted_sigma,'decision_utility':dec.utility})
            if dec.stop or mask.sum()==len(mask): break
            mask=add_candidate(mask,dec.action); step+=1
            if mask.sum()==len(mask):
                k=subset_key(mask); q=util_lookup[(pid,k)]; rows.append({'patient_id':pid,'step':step,'subset':k,'task_utility':q,'action':'FULL','pred_gain':np.nan,'pred_sigma':np.nan,'decision_utility':np.nan}); break
        finals.append({'patient_id':pid,'final_utility':q,'n_sources':int(mask.sum()),'stopped_early':int(mask.sum()<len(mask))})
    pd.DataFrame(rows).to_csv(out/'sequential_trajectories.csv',index=False); f=pd.DataFrame(finals); f.to_csv(out/'sequential_patient_summary.csv',index=False)
    fullkey='1'*len(cfg['dataset']['sources']); full=np.mean([util_lookup[(pid,fullkey)] for pid in f.patient_id])
    summ={'mean_final_utility':float(f.final_utility.mean()),'mean_sources':float(f.n_sources.mean()),'early_stop_rate':float(f.stopped_early.mean()),'full_information_utility':float(full),'retained_fraction':float(f.final_utility.mean()/full) if full!=0 else None}
    save_json(summ,out/'sequential_summary.json'); print(json.dumps(summ,indent=2))
if __name__=='__main__': main()
