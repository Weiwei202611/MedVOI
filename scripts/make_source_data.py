#!/usr/bin/env python
from pathlib import Path
import argparse, shutil, pandas as pd
from medvoi.utils import load_config,ensure_dir

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_config(args.config); out=Path(cfg['output_dir']); sd=Path(ensure_dir(out/'source_data'))
    mapping={
      'one_step_predictions.csv':'Fig3_Fig4_one_step_source_data.csv',
      'gain_calibration_test.csv':'Fig3d_calibration_source_data.csv',
      'sequential_trajectories.csv':'Fig5_Fig6_trajectories_source_data.csv',
      'sequential_patient_summary.csv':'Fig5_patient_summary_source_data.csv',
      'utilities_test.csv':'task_utility_by_patient_state.csv',
      'gains_test.csv':'realized_gain_by_patient_state_candidate.csv'}
    for src,dst in mapping.items():
        if (out/src).exists(): shutil.copy2(out/src,sd/dst)
    print('Source data written to',sd)
if __name__=='__main__': main()
