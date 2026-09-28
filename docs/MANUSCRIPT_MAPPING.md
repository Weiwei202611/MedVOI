# Manuscript-to-code mapping

| Manuscript component | Code |
|---|---|
| Non-empty information subsets and availability mask | `medvoi/data/subsets.py` |
| Subset-aware task model | `medvoi/models/task_models.py` |
| Modality dropout | `medvoi/engine.py::train_task_model` + YAML `training.modality_dropout` |
| Patient-level task utility Q_i(S) | `medvoi/metrics.py`, `medvoi/engine.py::evaluate_all_states` |
| Realized gain G_i(m;S) | `medvoi/gains.py` |
| Five-fold patient-level cross-fitting | `scripts/crossfit_gain_targets.py` |
| Patient-state vector r_i^S | `medvoi/state.py` |
| Candidate identity embedding | `medvoi/models/value_model.py::ConditionalValueNet` |
| Predicted gain mean and log variance | `ConditionalValueNet.mean`, `ConditionalValueNet.logvar` |
| Heteroscedastic regression | `medvoi/losses.py::heteroscedastic_gain_nll` |
| Pairwise ranking supervision | `medvoi/losses.py::value_loss` |
| Expected-gain selection and STOP | `medvoi/acquisition.py::select_action` |
| Risk-aware utility mu - beta*sigma | `medvoi/acquisition.py` |
| Normalized-burden sensitivity | `lambda_cost` and `costs` in YAML |
| Direct action classifier baseline | `medvoi/baselines/direct_classifier.py` |
| RL baseline | `medvoi/baselines/rl.py` |
| Gain MAE/Pearson/Spearman | `medvoi/metrics.py::gain_metrics` |
| Calibration across gain bins | `medvoi/calibration.py` |
| Regret / negative acquisition / oracle recovery | `medvoi/metrics.py::decision_metrics` |
| Patient-level bootstrap | `medvoi/stats.py` |
| Sequential retrospective revelation | `scripts/evaluate_sequential.py` |
| Source-data export | `scripts/make_source_data.py` |
