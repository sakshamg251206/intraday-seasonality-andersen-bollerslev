#!/usr/bin/env bash
# One-command reproduction (same as `make all`). Python 3.12, deps pinned in uv.lock.
set -euo pipefail
uv sync --frozen
uv run pytest -q
for s in 01_download 02_validate 03_replicate 04_extensions 05_robustness 06_oos; do
  echo "== $s"; uv run python -W ignore "scripts/$s.py"
done
(cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex)
