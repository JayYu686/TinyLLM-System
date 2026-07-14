.PHONY: audit bootstrap bootstrap-cpu bootstrap-gpu check coverage format-check install-local lint links public-check schema-check test typecheck

VENV ?= .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PIP_AUDIT := $(VENV)/bin/pip-audit

bootstrap:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -c requirements/constraints/dev.txt -e ".[dev]"

bootstrap-cpu: bootstrap
	$(PYTHON) -m pip install -r requirements/torch-cpu.txt

bootstrap-gpu: bootstrap
	$(PYTHON) -m pip install -r requirements/torch-cu118.txt

install-local:
	$(PYTHON) -m pip install -c requirements/constraints/dev.txt -e ".[dev]"

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

typecheck:
	$(MYPY)

test:
	$(PYTEST) -m "not gpu"

coverage:
	$(PYTEST) -m "not gpu" --cov=tinyllm --cov-branch --cov-report=term-missing

schema-check:
	$(PYTHON) scripts/export_schemas.py --check

links:
	$(PYTHON) scripts/check_markdown_links.py

public-check:
	$(PYTHON) scripts/check_public_artifacts.py

audit:
	$(PIP_AUDIT) --skip-editable

check: lint format-check typecheck coverage schema-check links public-check
