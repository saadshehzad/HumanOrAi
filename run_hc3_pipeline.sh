#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/raw data/processed

python src/scripts/load_hc3.py
python src/scripts/clean_hc3_data.py
python src/scripts/split_hc3_data.py

echo "HC3 data pipeline completed successfully."
