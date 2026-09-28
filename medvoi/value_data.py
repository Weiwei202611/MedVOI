from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GainDataset(Dataset):
    def __init__(self, gain_csv, state_npz):
        g=pd.read_csv(gain_csv)
        z=np.load(state_npz,allow_pickle=True)
        lookup={(str(p),str(s)):v.astype(np.float32) for p,s,v in zip(z['patient_id'],z['subset'],z['state'])}
        rows=[]
        for _,r in g.iterrows():
            key=(str(r.patient_id),str(r.subset))
            if key in lookup:
                rows.append((str(r.patient_id),str(r.subset),int(r.candidate),float(r.gain),lookup[key]))
        if not rows: raise ValueError('No gain rows matched state features.')
        self.rows=rows
        # Stable integer key used for within-state ranking in minibatches.
        keys={k:i for i,k in enumerate(sorted({(r[0],r[1]) for r in rows}))}
        self.group=[keys[(r[0],r[1])] for r in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        pid,subset,c,g,s=self.rows[i]
        return {'state':torch.tensor(s,dtype=torch.float32),'candidate':torch.tensor(c),'gain':torch.tensor(g,dtype=torch.float32),'group':torch.tensor(self.group[i]),'patient_id':pid,'subset':subset}


def collate_gain(batch):
    return {'state':torch.stack([x['state'] for x in batch]),
            'candidate':torch.stack([x['candidate'] for x in batch]).long(),
            'gain':torch.stack([x['gain'] for x in batch]),
            'group':torch.stack([x['group'] for x in batch]).long(),
            'patient_id':[x['patient_id'] for x in batch], 'subset':[x['subset'] for x in batch]}
