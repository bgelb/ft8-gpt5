PYTHON ?= /workspace/.venv/bin/python
SRC := /workspace/src

.PHONY: venv test char

venv:
	python3 -m venv /workspace/.venv || true
	/Workspace/.venv/bin/python -m pip install -U pip setuptools wheel || true
	/Workspace/.venv/bin/python -m pip install -r /workspace/requirements.txt || true

char:
	PYTHONPATH=$(SRC) $(PYTHON) $(SRC)/ft8gpt/char/run_char.py awgn_snr_sweep --outdir reports
	PYTHONPATH=$(SRC) $(PYTHON) $(SRC)/ft8gpt/char/run_char.py cfo_sweep --outdir reports
	PYTHONPATH=$(SRC) $(PYTHON) $(SRC)/ft8gpt/char/run_char.py occupancy --outdir reports
	PYTHONPATH=$(SRC) $(PYTHON) $(SRC)/ft8gpt/char/run_char.py end_to_end --outdir reports

# Default test runner

test:
	PYTHONPATH=$(SRC) /workspace/.venv/bin/pytest -q