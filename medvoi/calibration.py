from __future__ import annotations
import numpy as np
import pandas as pd


def gain_calibration(pred, true, n_bins=8):
    pred=np.asarray(pred,float); true=np.asarray(true,float)
    q=np.quantile(pred,np.linspace(0,1,n_bins+1))
    q[0]-=1e-12; q[-1]+=1e-12
    bins=np.digitize(pred,q[1:-1],right=True)
    rows=[]
    for b in range(n_bins):
        m=bins==b
        if not m.any(): continue
        rows.append({'bin':b,'n':int(m.sum()),'predicted_gain':float(pred[m].mean()),'realized_gain':float(true[m].mean())})
    df=pd.DataFrame(rows)
    mad=float(np.mean(np.abs(df.predicted_gain-df.realized_gain))) if len(df) else np.nan
    return df,mad
