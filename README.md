FT8 Decoder (Educational)

This project implements an educational FT8 decoder library in Python with a focus on clear structure, tests, and reproducible benchmarks.

- Library: `ft8dec`
- Tests: `pytest` with regression/perf gates
- Dataset: fixed commit from `kgoba/ft8_lib` test WAVs

Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run tests

```bash
pytest
```

