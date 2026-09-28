from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .data.subsets import candidate_indices, add_candidate


@dataclass
class Decision:
    action: int | None
    utility: float
    predicted_gain: float
    predicted_sigma: float
    stop: bool


def select_action(value_model, state_tensor, mask, tau=0.0, beta=0.0, costs=None, lambda_cost=0.0, device='cpu'):
    candidates=candidate_indices(mask)
    if not candidates:
        return Decision(None,0.0,0.0,0.0,True)
    st=state_tensor.to(device).view(1,-1).repeat(len(candidates),1)
    c=torch.tensor(candidates,dtype=torch.long,device=device)
    with torch.no_grad():
        mu,lv=value_model(st,c)
        sigma=torch.exp(0.5*lv)
    util=mu-beta*sigma
    if costs is not None:
        cc=torch.tensor([costs[j] for j in candidates],device=device,dtype=util.dtype)
        util=util-lambda_cost*cc
    j=int(torch.argmax(util).item())
    if float(util[j]) <= tau:
        return Decision(None,float(util[j]),float(mu[j]),float(sigma[j]),True)
    return Decision(candidates[j],float(util[j]),float(mu[j]),float(sigma[j]),False)


def retrospective_trajectory(initial_mask, state_provider, value_model, tau=0.0, beta=0.0, costs=None, lambda_cost=0.0, device='cpu'):
    """state_provider(mask) -> torch state vector. Candidate data stay hidden until mask update."""
    mask=np.asarray(initial_mask,dtype=np.int64).copy()
    traj=[]
    while mask.sum()<len(mask):
        state=state_provider(mask)
        dec=select_action(value_model,state,mask,tau,beta,costs,lambda_cost,device)
        traj.append({'subset': ''.join(map(str,mask.tolist())), 'action': dec.action, 'stop': dec.stop,
                     'utility':dec.utility,'pred_gain':dec.predicted_gain,'pred_sigma':dec.predicted_sigma})
        if dec.stop: break
        mask=add_candidate(mask,dec.action)
    return traj
