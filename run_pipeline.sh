#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/brats.yaml}
python scripts/validate_manifest.py --config "$CONFIG"
python scripts/train_task.py --config "$CONFIG"
OUT=$(python - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['output_dir'])
PY
)
python scripts/crossfit_gain_targets.py --config "$CONFIG"
python scripts/extract_states.py --config "$CONFIG" --checkpoint "$OUT/task_final.pt"
python scripts/train_value.py --config "$CONFIG"
python scripts/evaluate_one_step.py --config "$CONFIG"
python scripts/evaluate_sequential.py --config "$CONFIG"
python scripts/make_source_data.py --config "$CONFIG"
python scripts/plot_results.py --config "$CONFIG"
