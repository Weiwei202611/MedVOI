from __future__ import annotations
import numpy as np
import torch


def segmentation_uncertainty(logits):
    p=torch.sigmoid(logits)
    ent=-(p.clamp(1e-6,1-1e-6)*torch.log(p.clamp(1e-6,1-1e-6)) +
          (1-p).clamp(1e-6,1-1e-6)*torch.log((1-p).clamp(1e-6,1-1e-6)))
    mean_ent=ent.flatten(1).mean(1)
    fg_conf=torch.maximum(p,1-p).flatten(1).mean(1)
    boundary=((p>0.3)&(p<0.7)).float()
    b_unc=(ent*boundary).flatten(1).sum(1)/boundary.flatten(1).sum(1).clamp_min(1)
    volume=(p>0.5).float().flatten(1).mean(1)
    return torch.stack([mean_ent,fg_conf,b_unc,volume],1)


def classification_uncertainty(logits, ensemble_dispersion=None):
    p=torch.sigmoid(logits)
    ent=-(p.clamp(1e-6,1-1e-6)*torch.log(p.clamp(1e-6,1-1e-6)) +
          (1-p).clamp(1e-6,1-1e-6)*torch.log((1-p).clamp(1e-6,1-1e-6)))
    disp=torch.zeros_like(p) if ensemble_dispersion is None else ensemble_dispersion
    return torch.stack([p,ent,disp],1)


def build_state_vector(deep_repr, uncertainty, availability):
    return torch.cat([deep_repr,uncertainty,availability.float()],1)
