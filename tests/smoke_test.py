#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from medvoi.data.subsets import all_nonempty_masks, candidate_indices, add_candidate
from medvoi.models.task_models import SubsetSeg3D, SubsetClassifier3D
from medvoi.models.value_model import ConditionalValueNet
from medvoi.losses import value_loss
from medvoi.acquisition import select_action
from medvoi.metrics import gain_metrics


def main():
    torch.manual_seed(1)
    sources=['t1','t1ce','t2','flair']
    model=SubsetSeg3D(sources,out_channels=3,base=4)
    x={s:torch.randn(2,1,16,16,16) for s in sources}
    mask=torch.tensor([[1,0,0,0],[1,1,0,0]],dtype=torch.float32)
    logits,z=model(x,mask,return_repr=True)
    assert logits.shape==(2,3,16,16,16)
    assert z.shape[0]==2

    state_dim=z.shape[1]+4+len(sources)
    vm=ConditionalValueNet(state_dim,len(sources),candidate_dim=8,hidden=(32,16))
    states=torch.randn(6,state_dim)
    cand=torch.tensor([1,2,3,1,2,3])
    target=torch.tensor([.08,.02,-.01,.04,.07,.01])
    group=torch.tensor([0,0,0,1,1,1])
    mu,lv=vm(states,cand)
    loss,parts=value_loss(mu,lv,target,cand,group,alpha=.5,tolerance=.005)
    loss.backward()
    assert np.isfinite(float(loss.detach()))

    dec=select_action(vm,states[0],np.array([1,0,0,0]),tau=-10,beta=.5,device='cpu')
    assert dec.action in (1,2,3)
    assert not dec.stop

    gm=gain_metrics([0.1,0.2,-0.1],[0.09,0.18,-0.08])
    assert gm['mae']<0.03
    print('MedVOI smoke test passed.')

if __name__=='__main__': main()
