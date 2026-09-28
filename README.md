# MedVOI: Patient-Specific Conditional Information Value

Reference implementation accompanying the manuscript:

**Predicting patient-specific information value enables adaptive multimodal medical AI**

Author: Hongwei Li, Southeast University

## What this package implements

This repository follows the manuscript's computational design:

1. A **subset-aware task model** makes a prediction from any non-empty subset of available sources.
2. Patient-level **task utility** is computed for every observed information state.
3. **Cross-fitted task models** generate out-of-fold realized acquisition-gain targets:

   `G_i(m; S) = Q_i(S ∪ {m}) - Q_i(S)`

4. A fixed task encoder extracts a common **patient-state representation** containing deep features, prediction-derived uncertainty descriptors, and an availability mask.
5. A candidate-specific **value network** predicts both mean gain and gain uncertainty without seeing the candidate source itself.
6. The value model is trained with **heteroscedastic gain regression + pairwise ranking**.
7. A sequential policy repeatedly re-estimates candidate value and chooses either a source or **STOP**.
8. One-step and sequential baselines, ablations, bootstrap statistics, and source-data export utilities are included.

The code is manifest-driven and supports the four study settings described in the manuscript:

- BraTS 2021: T1 / T1ce / T2 / FLAIR, 3D segmentation
- PI-CAI: T2W / DWI / ADC, patient-level classification
- Symile-MIMIC: CXR / ECG / laboratory measurements, classification
- autoPET: PET / CT, 3D lesion segmentation

## Important reproducibility note

The manuscript specifies the experimental logic but does not contain every optimizer and architecture hyperparameter. The YAML files in `configs/` therefore provide **reference implementation defaults** for fields that were not explicitly fixed in the manuscript. If you are reproducing an already executed experiment, replace those values with the exact run configuration before archival or submission.

No medical source data are redistributed by this package.

## Repository layout

```text
MedVOI_code_package/
├── configs/                  # Dataset-specific experiment configuration
├── docs/                     # Data-manifest and reproducibility guidance
├── medvoi/
│   ├── data/                 # Manifest datasets, subset utilities
│   ├── models/               # Subset-aware task models, value model
│   ├── baselines/            # Fixed/state mean, uncertainty, classifier, RL
│   ├── acquisition.py        # STOP, risk-aware and sequential acquisition
│   ├── calibration.py        # Gain calibration summaries
│   ├── gains.py              # Utility and realized gain construction
│   ├── losses.py             # Dice/CE and heteroscedastic ranking losses
│   ├── metrics.py            # Gain and decision metrics
│   ├── stats.py              # Patient-cluster bootstrap
│   └── utils.py              # Config, seed, checkpoint utilities
├── scripts/
│   ├── train_task.py
│   ├── crossfit_gain_targets.py
│   ├── extract_states.py
│   ├── train_value.py
│   ├── evaluate_one_step.py
│   ├── evaluate_sequential.py
│   ├── train_direct_classifier.py
│   ├── train_rl_baseline.py
│   └── make_source_data.py
├── tests/
│   └── smoke_test.py
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
conda env create -f environment.yml
conda activate medvoi
pip install -e .
```

or

```bash
pip install -r requirements.txt
pip install -e .
```

## Data manifests

Prepare one CSV per dataset. Exact formats are documented in `docs/DATASETS.md`.

Example BraTS row:

```csv
patient_id,split,t1,t1ce,t2,flair,label
BraTS_00001,train,/data/.../t1.nii.gz,/data/.../t1ce.nii.gz,/data/.../t2.nii.gz,/data/.../flair.nii.gz,/data/.../seg.nii.gz
```

The package never infers patient identity from filenames once a manifest has been supplied. Patient-level separation is enforced by the manifest.

## Primary BraTS pipeline

### 1. Train the final subset-aware task model

```bash
python scripts/train_task.py --config configs/brats.yaml
```

### 2. Generate cross-fitted out-of-fold gain targets

```bash
python scripts/crossfit_gain_targets.py --config configs/brats.yaml
```

This step trains the prespecified number of cross-fitting task models and writes out-of-fold patient-state utilities and realized one-step gains.

### 3. Extract common-coordinate patient-state features

```bash
python scripts/extract_states.py --config configs/brats.yaml \
  --checkpoint outputs/brats/task_final.pt
```

Cross-fitting defines gain **targets**. A single fixed task model then extracts state features so that the deep state vectors used by the value estimator share one latent coordinate system.

### 4. Train MedVOI

```bash
python scripts/train_value.py --config configs/brats.yaml
```

### 5. Evaluate one-step selection

```bash
python scripts/evaluate_one_step.py --config configs/brats.yaml
```

### 6. Evaluate retrospective sequential information revelation

```bash
python scripts/evaluate_sequential.py --config configs/brats.yaml
```

### 7. Export numerical source data underlying figures and tables

```bash
python scripts/make_source_data.py --config configs/brats.yaml
```

## Other datasets

Use the same commands with:

```bash
configs/picai.yaml
configs/symile_mimic.yaml
configs/autopet.yaml
```

The conditional-value objective is unchanged. Dataset adapters differ only in source encoders, endpoint and utility definition.

## Utility definitions

### Segmentation

Primary utility is patient-level mean Dice across configured foreground channels.

### Classification

Patient-level utility is negative Brier score:

`Q_i(S) = -(p_i(S) - y_i)^2`

so larger values are better.

### STOP

STOP is **not** passed through the candidate-value network. It is an explicit decision option whose gain is zero. A source is revealed only when the largest candidate utility exceeds the validation-selected stopping threshold.

## Risk-aware utility

The package implements:

```text
U(m; S) = mu_hat(m; S) - beta * sigma_hat(m; S) - lambda_cost * cost_m
```

with `beta=0` and `lambda_cost=0` for the primary expected-gain policy. Risk-aware and normalized-burden analyses are configuration options.

## Baselines

Implemented reference baselines:

- global mean-gain ranking
- state-conditioned mean gain
- uncertainty-based selection
- direct best-action classifier
- lightweight offline DQN-style modality-level RL baseline
- retrospective oracle upper bound

The manuscript discusses modern nongreedy active-feature-acquisition methods separately because a fair numerical comparison requires adapting them to the same modality-level action space, patient split and utility definition.

## Testing

Run the synthetic smoke test:

```bash
python tests/smoke_test.py
```

The test validates subset masking, gain construction, MedVOI forward/loss, STOP logic, sequential policy, and decision metrics without requiring medical data.

## Output files

Typical output folder:

```text
outputs/brats/
├── task_final.pt
├── crossfit_utilities.csv
├── crossfit_gains.csv
├── states_train.npz
├── states_val.npz
├── states_test.npz
├── value_model.pt
├── one_step_predictions.csv
├── one_step_metrics.json
├── sequential_trajectories.csv
├── sequential_summary.json
└── source_data/
```

## Data and code archival

For a submission/reproducibility archive, include:

- exact manifests with de-identified IDs
- immutable split files
- YAML configurations used for each run
- checkpoints where redistribution is permitted
- source-data CSV files for every main figure/table
- software environment
- random seeds and hardware information

Do not redistribute BraTS, PI-CAI, Symile-MIMIC, autoPET or third-party model weights unless their licenses permit it.
