from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class SharedEncoder3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.b1 = ConvBlock3D(1, base)
        self.b2 = ConvBlock3D(base, base*2)
        self.b3 = ConvBlock3D(base*2, base*4)
        self.pool = nn.MaxPool3d(2)
        self.channels = [base, base*2, base*4]
    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(self.pool(f1))
        f3 = self.b3(self.pool(f2))
        return [f1, f2, f3]


class SourceEmbedding(nn.Module):
    def __init__(self, n_sources, channels):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(n_sources, c) for c in channels])
    def forward(self, source_idx, level, ndim=3):
        e = self.tables[level](source_idx)
        shape = [e.shape[0], e.shape[1]] + [1]*ndim
        return e.view(*shape)


def _masked_mean_stack(feats, mask):
    # feats: list M of tensors B,C,...; mask B,M
    st = torch.stack(feats, 1)
    view = [mask.shape[0], mask.shape[1]] + [1]*(st.ndim-2)
    w = mask.view(*view).to(st.dtype)
    denom = w.sum(1).clamp_min(1.0)
    return (st*w).sum(1) / denom


class SubsetSeg3D(nn.Module):
    """Shared 3D encoder + source embeddings + masked multiscale fusion."""
    def __init__(self, sources: List[str], out_channels=3, base=24):
        super().__init__()
        self.sources = list(sources)
        self.encoder = SharedEncoder3D(base)
        self.source_emb = SourceEmbedding(len(sources), self.encoder.channels)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.dec2 = ConvBlock3D(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.dec1 = ConvBlock3D(base*2, base)
        self.head = nn.Conv3d(base, out_channels, 1)
        self.repr_dim = base*4

    def forward(self, inputs: Dict[str, torch.Tensor], availability: torch.Tensor, return_repr=False):
        per_level = [[] for _ in range(3)]
        for j, s in enumerate(self.sources):
            fs = self.encoder(inputs[s])
            idx = torch.full((inputs[s].shape[0],), j, device=inputs[s].device, dtype=torch.long)
            for l, f in enumerate(fs):
                per_level[l].append(f + self.source_emb(idx, l, ndim=3))
        fused = [_masked_mean_stack(per_level[l], availability) for l in range(3)]
        f1, f2, f3 = fused
        x = self.up2(f3)
        x = self.dec2(torch.cat([x, f2], 1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, f1], 1))
        logits = self.head(x)
        pooled = F.adaptive_avg_pool3d(f3, 1).flatten(1)
        return (logits, pooled) if return_repr else logits


class SubsetClassifier3D(nn.Module):
    def __init__(self, sources: List[str], base=24, hidden=128):
        super().__init__()
        self.sources = list(sources)
        self.encoder = SharedEncoder3D(base)
        self.source_emb = SourceEmbedding(len(sources), self.encoder.channels)
        self.head = nn.Sequential(nn.Linear(base*4, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, 1))
        self.repr_dim = base*4
    def forward(self, inputs, availability, return_repr=False):
        reps = []
        for j,s in enumerate(self.sources):
            f3 = self.encoder(inputs[s])[-1]
            idx = torch.full((inputs[s].shape[0],), j, device=f3.device, dtype=torch.long)
            f3 = f3 + self.source_emb(idx, 2, ndim=3)
            reps.append(F.adaptive_avg_pool3d(f3, 1).flatten(1))
        st = torch.stack(reps,1)
        w = availability.unsqueeze(-1)
        z = (st*w).sum(1) / w.sum(1).clamp_min(1.0)
        logits = self.head(z).squeeze(-1)
        return (logits, z) if return_repr else logits


class CXR2DEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,5,2,2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(128,out_dim)
    def forward(self,x): return self.proj(self.net(x).flatten(1))


class ECG1DEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12,64,9,2,4), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64,128,7,2,3), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,128,5,2,2), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.proj=nn.Linear(128,out_dim)
    def forward(self,x):
        if x.ndim==4 and x.shape[1]==1: x=x[:,0]
        return self.proj(self.net(x).flatten(1))


class LabsEncoder(nn.Module):
    def __init__(self, n_features=50, out_dim=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,out_dim))
    def forward(self,x): return self.net(x.flatten(1))


class HeteroSubsetClassifier(nn.Module):
    """Symile-MIMIC style classifier with source-compatible encoders."""
    def __init__(self, sources: List[str], source_types: Dict[str,str], latent=128, labs_dim=50):
        super().__init__()
        self.sources=list(sources)
        enc={}
        for s in self.sources:
            t=source_types[s]
            if t=='image2d': enc[s]=CXR2DEncoder(latent)
            elif t=='ecg': enc[s]=ECG1DEncoder(latent)
            elif t=='labs': enc[s]=LabsEncoder(labs_dim, latent)
            else: raise ValueError(f'Unknown source type {t}')
        self.encoders=nn.ModuleDict(enc)
        self.source_emb=nn.Embedding(len(self.sources), latent)
        self.head=nn.Sequential(nn.Linear(latent,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
        self.repr_dim=latent
    def forward(self, inputs, availability, return_repr=False):
        reps=[]
        for j,s in enumerate(self.sources):
            z=self.encoders[s](inputs[s])
            z=z+self.source_emb.weight[j].view(1,-1)
            reps.append(z)
        st=torch.stack(reps,1)
        w=availability.unsqueeze(-1)
        z=(st*w).sum(1)/w.sum(1).clamp_min(1.0)
        logits=self.head(z).squeeze(-1)
        return (logits,z) if return_repr else logits


def build_task_model(cfg):
    sources=cfg['dataset']['sources']
    kind=cfg['model']['task_model']
    if kind=='seg3d':
        return SubsetSeg3D(sources, cfg['model'].get('out_channels',3), cfg['model'].get('base_channels',24))
    if kind=='class3d':
        return SubsetClassifier3D(sources, cfg['model'].get('base_channels',24), cfg['model'].get('hidden',128))
    if kind=='hetero_class':
        return HeteroSubsetClassifier(sources, cfg['dataset']['source_types'], cfg['model'].get('latent',128), cfg['model'].get('labs_dim',50))
    raise ValueError(f'Unknown task model {kind}')
