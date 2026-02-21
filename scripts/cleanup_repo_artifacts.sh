#!/usr/bin/env bash
set -euo pipefail

echo "[cleanup] removing OS/editor artifacts and Python caches"
find . -type f \( -name '.DS_Store' -o -name '._*' -o -name '*.swp' -o -name '*~' \) -delete
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type d \( -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' -o -name '.snakemake' \) -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

echo "[cleanup] removing generated outputs"
rm -rf figures tests/fixtures/_tmp_out tests/fixtures/.cache_tmp tests/fixtures/Library/Caches/snakemake

echo "[cleanup] if files were tracked, untrack them once manually:"
echo "  git rm -r --cached --ignore-unmatch figures tests/fixtures/_tmp_out tests/fixtures/.cache_tmp tests/fixtures/Library/Caches/snakemake .snakemake"

echo "[cleanup] done"
