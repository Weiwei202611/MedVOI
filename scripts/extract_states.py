#!/usr/bin/env python
from pathlib import Path
import argparse
from medvoi.utils import load_config,get_device,load_checkpoint
from medvoi.models import build_task_model
from medvoi.engine import evaluate_all_states,save_states_npz

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--splits',default='train,val,test')
    args=ap.parse_args(); cfg=load_config(args.config); device=get_device(cfg); out=Path(cfg['output_dir'])
    model=build_task_model(cfg); load_checkpoint(args.checkpoint,model,map_location=device); model.to(device)
    for split in args.splits.split(','):
        util,states=evaluate_all_states(model,cfg,split.strip(),device,True)
        util.to_csv(out/f'utilities_{split.strip()}.csv',index=False); save_states_npz(states,out/f'states_{split.strip()}.npz')
        print(split,len(states))
if __name__=='__main__': main()
