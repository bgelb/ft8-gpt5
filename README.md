# FT8-GPT5

A from-scratch FT8 decoder library with tests and benchmarks.

## Original goals and approach
- Build a Python library that decodes FT8 signals from 15 s WAV inputs and outputs structured decodes (message text, timing/frequency estimates, SNR, integrity flags).
- Implement from first principles based on the FT8 literature (QEX paper) and compare against open-source references for understanding (not verbatim copying).
- Provide regression and performance tests that run in CI; prevent regressions.
- Target O(1 s) per-sample runtime long-term.

## Architecture summary
- Input/Waterfall: compute tone-aligned magnitudes with Hann windowing and oversampling.
- Sync (coarse): vectorized 7x7 Costas detection across time/frequency to propose top-N candidates from the waterfall.
- Sync (fine): quasi-coherent cross-correlation against a complex Costas reference, scanning Δt in ±40 ms (5 ms steps) and Δf in ±2.5 Hz (0.5 Hz steps), performed at 200 Hz after downmix+decimate.
- Demod: coherent per-symbol energies via complex mixing/integration at refined CFO and timing.
- LLR + LDPC: Gray-map LLRs formed via Gray-group log-sum ratios over linear energies; min-sum LDPC (174,91) with tuned iterations and early-stop.
- CRC + Message: CRC-14 check; message unpack to human-readable text (standard and selected non-standard forms).
- API: `ft8gpt.api.decode_wav(path_or_file)` returns `DecodeResult` records with integrity flags.

## Requirements
- Python 3.11+ (3.13 recommended on macOS).
- Pinned libraries (via `requirements.txt`):
  - numpy 2.3.2
  - scipy 1.16.1
  - soundfile 0.12.1
  - numba 0.58.1
  - pytest 7.4.4
  - pytest-benchmark 4.0.0
  - requests 2.31.0

## Environment setup
Use the script to create a venv with Python 3.13/3.12/3.11, install dependencies, and run tests.

```bash
./scripts/setup_env.sh           # creates .venv, installs deps, runs tests
./scripts/setup_env.sh --clean   # remove venv and caches
```

Notes:
- The script requires a system Python ≥3.11 and prefers 3.13 → 3.12 → 3.11. Install/upgrade manually if needed.

## Running tests
- Quick suite: `pytest -q`
- Full dataset regression (long): `pytest -q -m regression -m slow`

## Dataset
- Uses `kgoba/ft8_lib` test WAVs as a submodule under `external/ft8_lib` for regression comparison to WSJT-X outputs.

## CI
- GitHub Actions runs the test suite on push/PR.
- Regression and performance guardrails ensure no decode-rate or runtime regressions.

## Status and next steps
- Strong-signal decode validated (coherent path only) with zero LDPC syndrome and valid CRC.
- Costas gate threshold is currently relaxed to 3 matches (out of 21) to allow weak-but-correct candidates through; LDPC+CRC provide strong validation. We should revisit this threshold after broader dataset evaluation and potentially raise it (e.g., 7–10) once decode rate on real signals is healthy.
- Improve candidate breadth, LLR quality, and message normalization to raise aggregate decode rate; optimize runtime without sacrificing accuracy.
