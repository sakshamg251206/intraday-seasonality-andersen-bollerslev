# Full reproduction:  make all   (needs uv: https://docs.astral.sh/uv/)
PY = uv run python -W ignore

all: data validate replicate extensions robustness oos paper

env:
	uv sync --frozen

data: env
	$(PY) scripts/01_download.py

validate:
	$(PY) scripts/02_validate.py

replicate:
	$(PY) scripts/03_replicate.py

extensions:
	$(PY) scripts/04_extensions.py

robustness:
	$(PY) scripts/05_robustness.py

oos:
	$(PY) scripts/06_oos.py

test:
	uv run pytest -q

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex

.PHONY: all env data validate replicate extensions robustness oos test paper
