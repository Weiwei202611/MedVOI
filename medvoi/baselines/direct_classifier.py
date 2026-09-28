from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from ..models.value_model import BestActionClassifier

class StateActionDataset(Dataset):
    def __init__(self,states,labels): self.states=torch.tensor(states,dtype=torch.float32); self.labels=torch.tensor(labels,dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return self.states[i],self.labels[i]

def train_classifier(states,labels,n_actions,epochs=100,lr=1e-3,batch_size=128,device='cpu'):
    model=BestActionClassifier(states.shape[1],n_actions).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    dl=DataLoader(StateActionDataset(states,labels),batch_size=batch_size,shuffle=True)
    for _ in range(epochs):
        model.train()
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            loss=F.cross_entropy(model(x),y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model
