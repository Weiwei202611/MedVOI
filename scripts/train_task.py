#!/usr/bin/env python
from pathlib import Path
import argparse, torch
from medvoi.utils import load_config,set_seed,get_device,ensure_dir,save_checkpoint
from medvoi.models import build_task_model
from medvoi.engine import train_task_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); seed=cfg['training'].get('seed',2026); set_seed(seed); device=get_device(cfg)
    out=Path(args.output or cfg['output_dir']); ensure_dir(out)
    model=build_task_model(cfg); model,meta=train_task_model(model,cfg,device)
    save_checkpoint(out/'task_final.pt',model,extra={'config':cfg,'meta':meta})
    print(f'Saved {out/"task_final.pt"}')
if __name__=='__main__': main()
