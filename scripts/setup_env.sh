#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV_DIR="$ROOT_DIR/.venv"

if [[ "${1:-}" == "--clean" ]]; then
  echo "Cleaning environment..."
  rm -rf "$VENV_DIR"
  find "$ROOT_DIR" -name "__pycache__" -type d -exec rm -rf {} + || true
  find "$ROOT_DIR" -name "*.pyc" -delete || true
  echo "Done."
  exit 0
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

echo "Running tests..."
pytest -q
echo "Environment ready."


