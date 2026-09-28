#!/usr/bin/env python
"""Generate simple publication-ready analysis plots from MedVOI outputs.

This script deliberately reads exported numerical result files rather than
hard-coding manuscript numbers. Styling can be adjusted before journal upload.
"""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from medvoi.utils import load_config


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_config(args.config); out=Path(cfg['output_dir']); fig=out/'figures'; fig.mkdir(parents=True,exist_ok=True)
    one=out/'one_step_predictions.csv'
    if one.exists():
        d=pd.read_csv(one)
        plt.figure(figsize=(4.2,4.0)); plt.scatter(d.gain,d.pred_gain,s=9,alpha=.35); lo=min(d.gain.min(),d.pred_gain.min()); hi=max(d.gain.max(),d.pred_gain.max()); plt.plot([lo,hi],[lo,hi],'--',linewidth=1); plt.xlabel('Realized gain'); plt.ylabel('Predicted gain'); plt.tight_layout(); plt.savefig(fig/'predicted_vs_realized_gain.pdf'); plt.close()
        plt.figure(figsize=(4.2,4.0)); plt.scatter(d.pred_sigma,np.abs(d.pred_gain-d.gain),s=9,alpha=.35); plt.xlabel('Predicted gain uncertainty'); plt.ylabel('Absolute gain error'); plt.tight_layout(); plt.savefig(fig/'gain_uncertainty_vs_error.pdf'); plt.close()
    seq=out/'sequential_patient_summary.csv'
    if seq.exists():
        d=pd.read_csv(seq); counts=d.n_sources.value_counts().sort_index(); plt.figure(figsize=(4.2,3.6)); plt.bar(counts.index,counts.values); plt.xlabel('Sources revealed'); plt.ylabel('Patients'); plt.tight_layout(); plt.savefig(fig/'sources_used_distribution.pdf'); plt.close()
    print('Figures written to',fig)
if __name__=='__main__': main()
