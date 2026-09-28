from __future__ import annotations
import numpy as np
import pandas as pd


def global_mean_scores(train_gains: pd.DataFrame, eval_gains: pd.DataFrame):
    means=train_gains.groupby('candidate').gain.mean().to_dict()
    out=eval_gains.copy(); out['score']=out.candidate.map(means).fillna(0.0)
    return out


def state_mean_scores(train_gains: pd.DataFrame, eval_gains: pd.DataFrame):
    means=train_gains.groupby(['subset','candidate']).gain.mean().to_dict()
    out=eval_gains.copy(); out['score']=[means.get((s,c),0.0) for s,c in zip(out.subset,out.candidate)]
    return out


def select_max_score(df, score_col='score', tau=-np.inf):
    out=[]
    for sid,g in df.groupby('state_id'):
        j=g[score_col].astype(float).idxmax()
        row=g.loc[j].copy()
        if float(row[score_col])<=tau:
            chosen='STOP'
        else:
            chosen=row.candidate
        gg=g.copy(); gg['selected_candidate']=chosen
        out.append(gg)
    return pd.concat(out,ignore_index=True)
