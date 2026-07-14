.PHONY: bootstrap bootstrap-cpu bootstrap-gpu install-local lint format-check typecheck test check

VENV ?= .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest

bootstrap:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

bootstrap-cpu: bootstrap
	$(PYTHON) -m pip install -r requirements/torch-cpu.txt

bootstrap-gpu: bootstrap
	$(PYTHON) -m pip install -r requirements/torch-cu118.txt

install-local:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

typecheck:
	$(MYPY)

test:
	$(PYTEST) -m "not gpu"

check: lint format-check typecheck test
