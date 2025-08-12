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

# Require Python >=3.11; prefer 3.13 then 3.12 then 3.11
select_python() {
  for bin in python3.13 python3.12 python3.11; do
    if command -v "$bin" >/dev/null 2>&1; then
      echo "$bin"; return 0
    fi
  done
  # Fallback to python3 if it is sufficiently new
  if command -v python3 >/dev/null 2>&1; then
    if python3 - <<'PY' >/dev/null 2>&1; then exit 0; else exit 1; fi <<'PY'
import sys; raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor) >= (3,11) else 1)
PY
    then
      echo python3; return 0
    fi
  fi
  return 1
}

PYBIN=$(select_python || true)
if [[ -z "${PYBIN:-}" ]]; then
  echo "Error: Python >=3.11 is required. Please install Python 3.11+ (3.13 recommended) and re-run." >&2
  exit 1
fi

echo "Using interpreter: $($PYBIN -V)"

# Recreate venv if it exists with an older Python
if [[ -x "$VENV_DIR/bin/python" ]]; then
  if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1; then
    echo "Existing venv uses Python <3.11; recreating..."
    rm -rf "$VENV_DIR"
  fi <<'PY'
import sys; raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor) >= (3,11) else 1)
PY
fi

"$PYBIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo "Installing dependencies..."
pip install -r "$ROOT_DIR/requirements.txt"

echo "Installing project (editable)..."
pip install -e "$ROOT_DIR" >/dev/null

echo "Running tests..."
pytest -q
echo "Environment ready."


