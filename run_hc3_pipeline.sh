#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p src/data/raw src/data/processed

"$PYTHON_BIN" src/scripts/load_hc3.py
"$PYTHON_BIN" src/scripts/clean_hc3_data.py
"$PYTHON_BIN" src/scripts/split_hc3_data.py

echo "HC3 data pipeline completed successfully."
