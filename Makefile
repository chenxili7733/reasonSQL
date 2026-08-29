PYTHON ?= python3

.PHONY: install test check smoke prepare download sft grpo opd eval eval-series compare clean-python

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m compileall -q reasonsql common scripts eval patches tests
	@for script in scripts/*.sh; do bash -n "$$script"; done

smoke:
	$(PYTHON) scripts/smoke_test.py

download:
	bash scripts/00_download_bird.sh

prepare:
	$(PYTHON) scripts/01_prepare_data.py --rl-size 2000

sft:
	bash scripts/02_train_sft.sh

grpo:
	bash scripts/10_run_grpo.sh

opd:
	bash scripts/11_run_grpo_opd.sh

eval:
	$(PYTHON) eval/eval_bird.py --help

eval-series:
	bash eval/eval_series.sh

compare:
	$(PYTHON) eval/compare_results.py --help

clean-python:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
