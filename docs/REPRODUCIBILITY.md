# Reproducibility checklist

Before archiving the code used for a manuscript submission, replace reference defaults with the exact executed configuration and preserve:

- Python, PyTorch, CUDA and driver versions
- GPU type and count
- optimizer and learning-rate schedule
- weight decay
- batch size
- epochs / iteration budget
- early-stopping criterion
- all random seeds
- patient-level split files
- cross-fitting fold assignment
- modality-dropout schedule
- task-model architecture and dimensions
- value-model candidate embedding dimension
- value-model hidden dimensions
- heteroscedastic regression setup
- pairwise-ranking weight `alpha`
- pairwise-ranking tolerance `epsilon`
- stopping threshold `tau`
- risk penalty `beta`
- normalized source costs and cost weight if used
- calibration procedure
- checkpoint-selection rule
- source-data CSVs underlying all main figures and tables

## Recommended execution logic

The reference package uses a two-stage strategy that resolves a subtle representation issue in cross-fitting:

1. Cross-fitted task models are used only to construct out-of-fold **gain targets**.
2. A final task model trained on the full development-training partition is frozen and used to extract deep state vectors for value learning.

This keeps gain supervision out-of-fold while ensuring all deep patient-state vectors presented to the value estimator lie in one common latent coordinate system.
