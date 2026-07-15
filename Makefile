.PHONY: audit audit-baseline bootstrap bootstrap-baseline bootstrap-cpu bootstrap-gpu check coverage format-check install-local lint links public-check schema-check test typecheck

VENV ?= .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PIP_AUDIT := $(VENV)/bin/pip-audit
BASELINE_VENV ?= .venv-baseline
BASELINE_PYTHON := $(BASELINE_VENV)/bin/python
BASELINE_PIP_AUDIT := $(BASELINE_VENV)/bin/pip-audit

bootstrap:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -c requirements/constraints/dev.txt -e ".[dev]"

bootstrap-cpu: bootstrap
	$(PYTHON) -m pip install -r requirements/torch-cpu.txt

bootstrap-gpu: bootstrap
	$(PYTHON) -m pip install -r requirements/torch-cu118.txt

bootstrap-baseline:
	python3 -m venv $(BASELINE_VENV)
	$(BASELINE_PYTHON) -m pip install --upgrade pip
	$(BASELINE_PYTHON) -m pip install -r requirements/torch-cu118.txt
	$(BASELINE_PYTHON) -m pip install -c requirements/constraints/baseline.txt -e ".[baseline]" pip-audit setuptools

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

audit-baseline:
	$(BASELINE_PIP_AUDIT) --skip-editable \
		--ignore-vuln PYSEC-2025-217 \
		--ignore-vuln PYSEC-2026-1939 \
		--ignore-vuln PYSEC-2026-2288 \
		--ignore-vuln PYSEC-2026-2289 \
		--ignore-vuln PYSEC-2026-2290

check: lint format-check typecheck coverage schema-check links public-check
