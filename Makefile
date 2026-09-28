CONFIG ?= configs/brats.yaml

smoke:
	python tests/smoke_test.py

validate:
	python scripts/validate_manifest.py --config $(CONFIG)

task:
	python scripts/train_task.py --config $(CONFIG)

crossfit:
	python scripts/crossfit_gain_targets.py --config $(CONFIG)

states:
	python scripts/extract_states.py --config $(CONFIG) --checkpoint $$(python -c "import yaml; c=yaml.safe_load(open('$(CONFIG)')); print(c['output_dir']+'/task_final.pt')")

value:
	python scripts/train_value.py --config $(CONFIG)

one-step:
	python scripts/evaluate_one_step.py --config $(CONFIG)

sequential:
	python scripts/evaluate_sequential.py --config $(CONFIG)

source-data:
	python scripts/make_source_data.py --config $(CONFIG)
