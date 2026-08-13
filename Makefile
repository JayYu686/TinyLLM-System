.PHONY: audit audit-baseline audit-m4 audit-serving bootstrap bootstrap-baseline bootstrap-cpu bootstrap-gpu bootstrap-m4 bootstrap-m5 bootstrap-serving bootstrap-serving-vllm check coverage format-check install-local lint links m4-dependency-smoke public-check schema-check test typecheck

VENV ?= .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PIP_AUDIT := $(VENV)/bin/pip-audit
BASELINE_VENV ?= .venv-baseline
BASELINE_PYTHON := $(BASELINE_VENV)/bin/python
BASELINE_PIP_AUDIT := $(BASELINE_VENV)/bin/pip-audit
M4_VENV ?= .venv-m4
M4_PYTHON := $(M4_VENV)/bin/python
M4_PIP_AUDIT := $(M4_VENV)/bin/pip-audit
M5_VENV ?= .venv-m5
M5_PYTHON := $(M5_VENV)/bin/python
SERVING_VENV ?= .venv-serving
SERVING_PYTHON := $(SERVING_VENV)/bin/python
SERVING_PIP_AUDIT := $(SERVING_VENV)/bin/pip-audit

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

bootstrap-m4:
	python3 -m venv $(M4_VENV)
	$(M4_PYTHON) -m pip install --upgrade pip
	$(M4_PYTHON) -m pip install -r requirements/torch-cu118.txt
	$(M4_PYTHON) -m pip install -c requirements/constraints/m4.txt -e ".[m4]" pip-audit setuptools

bootstrap-m5:
	python3 -m venv $(M5_VENV)
	$(M5_PYTHON) -m pip install --upgrade pip
	$(M5_PYTHON) -m pip install -r requirements/torch-cu118.txt
	$(M5_PYTHON) -m pip install -c requirements/constraints/m5.txt -e ".[m5]"

bootstrap-serving:
	python3 -m venv $(SERVING_VENV)
	$(SERVING_PYTHON) -m pip install --upgrade pip
	$(SERVING_PYTHON) -m pip install -c requirements/constraints/serving.txt -e ".[serving]" pip-audit setuptools

bootstrap-serving-vllm: bootstrap-serving
	$(SERVING_PYTHON) -m pip install -r requirements/serving-cu118.txt \
		--extra-index-url https://download.pytorch.org/whl/cu118
	$(SERVING_PYTHON) -m pip install --no-deps \
		https://github.com/vllm-project/vllm/releases/download/v0.8.5.post1/vllm-0.8.5.post1+cu118-cp38-abi3-manylinux1_x86_64.whl
	$(SERVING_PYTHON) -m pip check

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

m4-dependency-smoke:
	$(M4_PYTHON) scripts/check_m4_dependencies.py

audit-m4:
	$(M4_PIP_AUDIT) --skip-editable \
		--ignore-vuln PYSEC-2025-217 \
		--ignore-vuln PYSEC-2026-2288 \
		--ignore-vuln PYSEC-2026-2289 \
		--ignore-vuln PYSEC-2026-2290

audit-serving:
	$(SERVING_PIP_AUDIT) --skip-editable

check: lint format-check typecheck coverage schema-check links public-check
