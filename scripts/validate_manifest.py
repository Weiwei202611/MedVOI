#!/usr/bin/env python
import argparse, pandas as pd
from medvoi.utils import load_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); cfg=load_config(a.config); d=cfg['dataset']; df=pd.read_csv(d['manifest'])
    req=['patient_id','split',*d['sources'],d.get('label_col','label')]; missing=[x for x in req if x not in df.columns]
    if missing: raise SystemExit(f'Missing columns: {missing}')
    overlap=False
    by=df.groupby('patient_id').split.nunique()
    if (by>1).any(): overlap=True; print('ERROR: patient IDs appear in multiple splits:',by[by>1].index[:10].tolist())
    print('Rows:',len(df)); print('Patients:',df.patient_id.nunique()); print(df.split.value_counts());
    if overlap: raise SystemExit(2)
    print('Manifest validation passed.')
if __name__=='__main__': main()
