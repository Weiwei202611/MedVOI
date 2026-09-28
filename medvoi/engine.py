from __future__ import annotations
import os, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data.manifest import ManifestDataset, collate_manifest
from .data.subsets import all_nonempty_masks, sample_mask
from .losses import segmentation_loss, classification_loss
from .metrics import patient_mean_dice, negative_brier
from .state import segmentation_uncertainty, classification_uncertainty, build_state_vector
from .utils import ensure_dir, subset_key


def make_dataset(cfg, split):
    d=cfg['dataset']
    return ManifestDataset(d['manifest'],split,d['sources'],d['task'],d.get('target_shape'),d.get('normalize',True),d.get('label_col','label'))


def move_sources(sources,device):
    return {k:v.to(device,non_blocking=True) for k,v in sources.items()}


def make_loader(cfg, split, shuffle=False):
    ds=make_dataset(cfg,split)
    tr=cfg['training']
    return DataLoader(ds,batch_size=tr.get('batch_size',1),shuffle=shuffle,num_workers=tr.get('num_workers',0),collate_fn=collate_manifest)


def train_task_model(model,cfg,device,train_split='train',val_split='val'):
    model.to(device)
    tr=cfg['training']
    opt=torch.optim.AdamW(model.parameters(),lr=tr.get('lr',2e-4),weight_decay=tr.get('weight_decay',1e-5))
    train_loader=make_loader(cfg,train_split,True)
    val_loader=make_loader(cfg,val_split,False) if val_split else None
    best=float('inf'); best_state=None; patience=0
    rng=np.random.default_rng(tr.get('seed',2026))
    for epoch in range(tr.get('epochs',50)):
        model.train(); losses=[]
        for batch in train_loader:
            xs=move_sources(batch['sources'],device); y=batch['target'].to(device)
            mask_np=sample_mask(len(y),len(cfg['dataset']['sources']),tr['modality_dropout'],rng)
            mask=torch.tensor(mask_np,device=device)
            out=model(xs,mask)
            loss=segmentation_loss(out,y,tr.get('dice_weight',0.5)) if cfg['dataset']['task']=='segmentation' else classification_loss(out,y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            losses.append(float(loss.detach()))
        val=float(np.mean(losses)) if val_loader is None else evaluate_loss(model,val_loader,cfg,device)
        if val<best-1e-6:
            best=val; patience=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
        if patience>=tr.get('early_stopping_patience',15): break
    if best_state is not None: model.load_state_dict(best_state)
    return model, {'best_val_loss':best}


def evaluate_loss(model,loader,cfg,device):
    model.eval(); ls=[]
    full=torch.ones((1,len(cfg['dataset']['sources'])),device=device)
    with torch.no_grad():
        for batch in loader:
            xs=move_sources(batch['sources'],device); y=batch['target'].to(device)
            mask=full.repeat(len(y),1)
            out=model(xs,mask)
            loss=segmentation_loss(out,y) if cfg['dataset']['task']=='segmentation' else classification_loss(out,y)
            ls.append(float(loss))
    return float(np.mean(ls))


def _seg_utility_and_state(model,xs,y,mask,device):
    logits,z=model(xs,mask,return_repr=True)
    utility=patient_mean_dice(logits,y)
    u=segmentation_uncertainty(logits)
    state=build_state_vector(z,u,mask)
    return utility,state.detach().cpu().numpy(),logits


def _class_utility_and_state(model,xs,y,mask,device):
    logits,z=model(xs,mask,return_repr=True)
    p=torch.sigmoid(logits)
    utility=negative_brier(p.detach().cpu().numpy(),y.detach().cpu().numpy())
    u=classification_uncertainty(logits)
    state=build_state_vector(z,u,mask)
    return utility,state.detach().cpu().numpy(),logits


def evaluate_all_states(model,cfg,split,device,include_state=True):
    """Evaluate every non-empty source subset per patient.

    Uses batch_size=1 to preserve per-patient state rows and avoid ambiguous
    variable state enumeration.
    """
    ds=make_dataset(cfg,split)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_manifest)
    masks=all_nonempty_masks(len(cfg['dataset']['sources']))
    rows=[]; state_rows=[]
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            pid=batch['patient_id'][0]; xs=move_sources(batch['sources'],device); y=batch['target'].to(device)
            for m in masks:
                mt=torch.tensor(m[None],dtype=torch.float32,device=device)
                if cfg['dataset']['task']=='segmentation':
                    utility,state,_=_seg_utility_and_state(model,xs,y,mt,device)
                else:
                    utility,state,_=_class_utility_and_state(model,xs,y,mt,device)
                k=subset_key(m)
                rows.append({'patient_id':pid,'subset':k,'utility':float(utility[0]),'split':split})
                if include_state:
                    state_rows.append((pid,k,state[0]))
    return pd.DataFrame(rows), state_rows


def save_states_npz(state_rows,path):
    ids=np.array([r[0] for r in state_rows],dtype=object)
    subsets=np.array([r[1] for r in state_rows],dtype=object)
    states=np.stack([r[2] for r in state_rows]).astype(np.float32)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,patient_id=ids,subset=subsets,state=states)


def load_states_npz(path):
    z=np.load(path,allow_pickle=True)
    return z['patient_id'],z['subset'],z['state']
