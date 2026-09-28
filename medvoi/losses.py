from __future__ import annotations
import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits, target, eps=1e-6):
    if target.ndim == logits.ndim - 1:
        if logits.shape[1] == 1:
            target = target.unsqueeze(1).float()
        else:
            # Integer label map -> one-hot foreground channels 1..C
            chans=[]
            for c in range(1, logits.shape[1]+1):
                chans.append((target==c).float())
            target=torch.stack(chans,1)
    target=target.float()
    probs=torch.sigmoid(logits)
    dims=tuple(range(2,probs.ndim))
    inter=(probs*target).sum(dims)
    den=probs.sum(dims)+target.sum(dims)
    dice=(2*inter+eps)/(den+eps)
    return 1-dice.mean()


def segmentation_loss(logits, target, dice_weight=0.5):
    if target.ndim == logits.ndim - 1 and logits.shape[1] > 1:
        # Convert integer labels to foreground binary channels for BCE.
        target=torch.stack([(target==c).float() for c in range(1,logits.shape[1]+1)],1)
    elif target.ndim == logits.ndim - 1:
        target=target.unsqueeze(1).float()
    target=target.float()
    bce=F.binary_cross_entropy_with_logits(logits,target)
    dl=dice_loss_from_logits(logits,target)
    return dice_weight*dl+(1-dice_weight)*bce


def classification_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target.float())


def heteroscedastic_gain_nll(mu, logvar, target):
    inv_var=torch.exp(-logvar)
    return 0.5*(logvar + (target-mu)**2*inv_var).mean()


def pairwise_ranking_loss(mu_a, mu_b, gain_a, gain_b, tolerance=0.0):
    diff=gain_a-gain_b
    keep=diff.abs()>tolerance
    if not keep.any():
        return (mu_a.sum()+mu_b.sum())*0.0
    y=diff[keep].sign()
    score=(mu_a-mu_b)[keep]
    return F.softplus(-y*score).mean()


def value_loss(mu, logvar, target, candidate, state_key, alpha=0.5, tolerance=0.01):
    nll=heteroscedastic_gain_nll(mu,logvar,target)
    rank_terms=[]
    # Pair candidates within identical patient-state groups.
    unique=torch.unique(state_key)
    for g in unique:
        idx=torch.where(state_key==g)[0]
        if idx.numel()<2: continue
        for a in range(idx.numel()):
            for b in range(a+1,idx.numel()):
                ia,ib=idx[a],idx[b]
                d=(target[ia]-target[ib]).abs()
                if d>tolerance:
                    y=torch.sign(target[ia]-target[ib])
                    rank_terms.append(F.softplus(-y*(mu[ia]-mu[ib])))
    rank=torch.stack(rank_terms).mean() if rank_terms else nll*0.0
    return nll+alpha*rank, {'nll':float(nll.detach()),'rank':float(rank.detach())}
