from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def binary_dice(pred, target, eps=1e-6):
    pred=np.asarray(pred).astype(bool); target=np.asarray(target).astype(bool)
    den=pred.sum()+target.sum()
    if den==0: return 1.0
    return float((2*(pred & target).sum()+eps)/(den+eps))


def patient_mean_dice(logits, target, threshold=0.5):
    import torch
    probs=torch.sigmoid(logits).detach().cpu().numpy()
    t=target.detach().cpu().numpy()
    if t.ndim==probs.ndim-1:
        if probs.shape[1]==1:
            t=t[:,None]
        else:
            t=np.stack([(t==c).astype(np.float32) for c in range(1,probs.shape[1]+1)],1)
    out=[]
    for i in range(len(probs)):
        ds=[binary_dice(probs[i,c]>=threshold,t[i,c]>0.5) for c in range(probs.shape[1])]
        out.append(float(np.mean(ds)))
    return np.asarray(out)


def negative_brier(prob, target):
    p=np.asarray(prob,float); y=np.asarray(target,float)
    return -(p-y)**2


def gain_metrics(pred, true):
    p=np.asarray(pred,float); y=np.asarray(true,float)
    mae=float(np.mean(np.abs(p-y)))
    pear=float(pearsonr(p,y).statistic) if len(np.unique(p))>1 and len(np.unique(y))>1 else np.nan
    spear=float(spearmanr(p,y).statistic) if len(np.unique(p))>1 and len(np.unique(y))>1 else np.nan
    return {'mae':mae,'pearson':pear,'spearman':spear}


def decision_metrics(df, positive_oracle_only=False):
    """Expected columns: state_id, candidate, gain, selected(bool) or selected_candidate."""
    import pandas as pd
    rows=[]
    for state_id,g in df.groupby('state_id'):
        oracle_idx=g['gain'].astype(float).idxmax()
        oracle=g.loc[oracle_idx]
        if 'selected' in g:
            sel=g[g.selected.astype(bool)]
            if len(sel)!=1: continue
            sel=sel.iloc[0]
        else:
            chosen=g['selected_candidate'].iloc[0]
            sel=g[g.candidate==chosen].iloc[0]
        og=float(oracle.gain); sg=float(sel.gain)
        rows.append((state_id,int(sel.candidate==oracle.candidate),og,sg,og-sg,sg<0))
    z=pd.DataFrame(rows,columns=['state_id','correct','oracle_gain','selected_gain','regret','negative'])
    denom=z.loc[z.oracle_gain>0,'oracle_gain'].mean() if (z.oracle_gain>0).any() else np.nan
    recovery=z.loc[z.oracle_gain>0,'selected_gain'].mean()/denom if np.isfinite(denom) else np.nan
    return {'best_action_accuracy':float(z.correct.mean()),'realized_gain':float(z.selected_gain.mean()),
            'regret':float(z.regret.mean()),'negative_acquisition_rate':float(z.negative.mean()),
            'oracle_recovery':float(recovery), 'n_states':int(len(z))}


def classification_metrics(prob, target):
    p=np.asarray(prob); y=np.asarray(target)
    return {'auroc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p))}
