from __future__ import annotations
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self,state_dim,n_actions):
        super().__init__(); self.net=nn.Sequential(nn.Linear(state_dim,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self,x): return self.net(x)

class OfflineDQN:
    """Lightweight modality-level fitted DQN baseline.

    Transitions are constructed retrospectively from state vectors and realized
    task-utility increments. This is a reference RL baseline, not a reimplementation
    of any specific published AFA algorithm.
    """
    def __init__(self,state_dim,n_actions,device='cpu',gamma=.95,lr=1e-3):
        self.device=device; self.n_actions=n_actions; self.gamma=gamma
        self.q=QNet(state_dim,n_actions).to(device); self.target=QNet(state_dim,n_actions).to(device); self.target.load_state_dict(self.q.state_dict())
        self.opt=torch.optim.AdamW(self.q.parameters(),lr=lr)
    def fit(self,transitions,epochs=30,batch_size=256):
        # transition = (state, action, reward, next_state, done, next_valid_mask)
        rng=np.random.default_rng(2026)
        for ep in range(epochs):
            idx=rng.permutation(len(transitions))
            for st in range(0,len(idx),batch_size):
                batch=[transitions[i] for i in idx[st:st+batch_size]]
                s=torch.tensor(np.stack([b[0] for b in batch]),dtype=torch.float32,device=self.device)
                a=torch.tensor([b[1] for b in batch],dtype=torch.long,device=self.device)
                r=torch.tensor([b[2] for b in batch],dtype=torch.float32,device=self.device)
                ns=torch.tensor(np.stack([b[3] for b in batch]),dtype=torch.float32,device=self.device)
                done=torch.tensor([b[4] for b in batch],dtype=torch.float32,device=self.device)
                masks=torch.tensor(np.stack([b[5] for b in batch]),dtype=torch.bool,device=self.device)
                q=self.q(s).gather(1,a[:,None]).squeeze(1)
                with torch.no_grad():
                    nq=self.target(ns); nq[~masks]=-1e9; best=nq.max(1).values; y=r+self.gamma*(1-done)*best
                loss=F.smooth_l1_loss(q,y); self.opt.zero_grad(); loss.backward(); self.opt.step()
            if (ep+1)%5==0: self.target.load_state_dict(self.q.state_dict())
        return self
    def scores(self,state):
        with torch.no_grad(): return self.q(torch.tensor(state,dtype=torch.float32,device=self.device)).cpu().numpy()
