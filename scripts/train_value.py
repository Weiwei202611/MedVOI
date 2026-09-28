#!/usr/bin/env python
from pathlib import Path
import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from medvoi.utils import load_config,get_device,set_seed,save_checkpoint
from medvoi.value_data import GainDataset,collate_gain
from medvoi.models.value_model import ConditionalValueNet
from medvoi.losses import value_loss

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_config(args.config); set_seed(cfg['training'].get('seed',2026)); device=get_device(cfg); out=Path(cfg['output_dir'])
    ds=GainDataset(out/'crossfit_gains.csv',out/'states_train.npz')
    vc=cfg['value_model']; model=ConditionalValueNet(ds[0]['state'].numel(),len(cfg['dataset']['sources']),vc.get('candidate_dim',32),tuple(vc.get('hidden',[256,128])),vc.get('dropout',.1)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=vc.get('lr',1e-3),weight_decay=vc.get('weight_decay',1e-5))
    dl=DataLoader(ds,batch_size=vc.get('batch_size',256),shuffle=True,collate_fn=collate_gain)
    best=1e9; best_state=None
    for ep in range(vc.get('epochs',100)):
        model.train(); ls=[]
        for b in dl:
            st,c,g,grp=b['state'].to(device),b['candidate'].to(device),b['gain'].to(device),b['group'].to(device)
            mu,lv=model(st,c); loss,_=value_loss(mu,lv,g,c,grp,vc.get('ranking_weight',.5),vc.get('ranking_tolerance',.01))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); ls.append(float(loss.detach()))
        cur=float(np.mean(ls))
        if cur<best: best=cur; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%10==0: print(ep,cur)
    model.load_state_dict(best_state); save_checkpoint(out/'value_model.pt',model,extra={'state_dim':ds[0]['state'].numel(),'best_train_loss':best,'config':cfg})
    print('saved',out/'value_model.pt')
if __name__=='__main__': main()
