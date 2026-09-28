from __future__ import annotations
import pandas as pd
from .data.subsets import candidate_indices, add_candidate
from .utils import subset_key


def gains_from_utilities(utility_csv: str, output_csv: str):
    """Create realized one-step gain rows from utility table.

    utility_csv columns: patient_id, subset, utility[, split]
    subset is a bit-string like 1000.
    """
    from .utils import parse_subset_key
    df=pd.read_csv(utility_csv)
    lookup={(str(r.patient_id),str(r.subset)):float(r.utility) for _,r in df.iterrows()}
    rows=[]
    for _,r in df.iterrows():
        pid=str(r.patient_id); key=str(r.subset); mask=parse_subset_key(key)
        if mask.sum()==len(mask): continue
        q0=float(r.utility)
        for c in candidate_indices(mask):
            k2=subset_key(add_candidate(mask,c))
            if (pid,k2) not in lookup: continue
            q1=lookup[(pid,k2)]
            rows.append({'patient_id':pid,'subset':key,'candidate':c,'utility_before':q0,
                         'utility_after':q1,'gain':q1-q0,'state_id':f'{pid}:{key}',
                         'split':getattr(r,'split','train')})
    pd.DataFrame(rows).to_csv(output_csv,index=False)
    return output_csv
