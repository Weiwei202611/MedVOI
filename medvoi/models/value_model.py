from __future__ import annotations
import torch
import torch.nn as nn


class ConditionalValueNet(nn.Module):
    def __init__(self, state_dim: int, n_sources: int, candidate_dim=32, hidden=(256,128), dropout=0.1):
        super().__init__()
        self.candidate = nn.Embedding(n_sources, candidate_dim)
        dims=[state_dim+candidate_dim,*hidden]
        layers=[]
        for a,b in zip(dims[:-1],dims[1:]):
            layers += [nn.Linear(a,b), nn.GELU(), nn.Dropout(dropout)]
        self.backbone=nn.Sequential(*layers)
        self.mean=nn.Linear(dims[-1],1)
        self.logvar=nn.Linear(dims[-1],1)
    def forward(self, state, candidate_idx):
        c=self.candidate(candidate_idx)
        h=self.backbone(torch.cat([state,c],-1))
        mu=self.mean(h).squeeze(-1)
        logvar=self.logvar(h).squeeze(-1).clamp(-8,5)
        return mu, logvar


class BestActionClassifier(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=(256,128)):
        super().__init__()
        dims=[state_dim,*hidden,n_actions]
        layers=[]
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(dims)-2: layers.append(nn.GELU())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
