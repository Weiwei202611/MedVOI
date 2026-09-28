from __future__ import annotations
import numpy as np


def patient_bootstrap_difference(patient_ids, a, b, n_boot=10000, seed=2026, fn=np.mean):
    ids=np.asarray(patient_ids); a=np.asarray(a,float); b=np.asarray(b,float)
    uniq=np.unique(ids); rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(n_boot):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([np.where(ids==u)[0] for u in sampled])
        vals.append(fn(a[idx])-fn(b[idx]))
    vals=np.asarray(vals)
    return {'estimate':float(fn(a)-fn(b)), 'ci_low':float(np.quantile(vals,.025)), 'ci_high':float(np.quantile(vals,.975))}
