# FT8-GPT5

A from-scratch FT8 decoder library with tests and benchmarks.

## Original goals and approach
- Build a Python library that decodes FT8 signals from 15 s WAV inputs and outputs structured decodes (message text, timing/frequency estimates, SNR, integrity flags).
- Implement from first principles based on the FT8 literature (QEX paper) and compare against open-source references for understanding (not verbatim copying).
- Provide regression and performance tests that run in CI; prevent regressions.
- Target O(1 s) per-sample runtime long-term.

## Architecture summary
- Input/Waterfall: compute tone-aligned magnitudes with Hann windowing and oversampling.
- Sync (coarse): STFT-based Costas matched-filter over time/frequency (+0.0/+0.5 frac bins) to propose top-N candidates.
- Sync (fine): quasi-coherent cross-correlation against a Costas reference at 200 Hz (downmix+decimate). Then a micro-refinement uses a Costas SNR objective to search small alignment tweaks: integer symbol shifts (±3), sample shifts (±8 at 200 Hz), and CFO (±0.8 Hz).
- Demod: coherent per-symbol energies via complex mixing/integration at refined CFO and timing.
- LLR + LDPC: Gray-group log-sum LLRs; variance-normalized to match reference; min-sum LDPC (174,91) with tuned iterations and early stop.
- CRC + Message: CRC-14 check; FT8 standard-message unpack with corrected basecall character mapping.
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

## Performance

- Baseline silent 15 s decode runtime: ~7.4–8.9 s on Apple Silicon (before).
- Current runtime: ~0.9–1.0 s for a 15 s silent sample on the same machine.
- Changes that enabled the speedup:
  - Peak-only candidate selection along base-bin axis (+ fallback to best scores).
  - Vectorized STFT via strided framing + batched rFFT with per-frame median normalization.
  - Vectorized fine sync refinement on a pre-decimated buffer (time/CFO grid in NumPy).
  - Vectorized coherent symbol-energy computation (einsum over tone bank).
  - Cached IIR decimator (Chebyshev-I SOS) with zero-phase `sosfiltfilt` and stride decimation.
  - Reduced downstream candidate count (`top_k=40`).

Numbers will vary by platform and input contents; the above measurements were produced with Python 3.12 and pinned NumPy/SciPy wheels using `./scripts/setup_env.sh`.

## Status and next steps
- Strong-signal decode validated end-to-end (coherent demod → LDPC → CRC) on synthetic and real samples. A deterministic dataset test (`20m_busy/test_04.wav`) passes with an exact text match.
- Bit mapping aligned to the FT8 spec (a91 = first 91 bits post-LDPC); we also try the systematic mapping for internal synthetic compatibility.
- Message unpacking fixed for basecall character mapping (digits/letters/space), eliminating call corruption (e.g., 9A9A).
- Micro-refinement uses Costas SNR only (no bit-match feedback), ensuring spec-aligned sync logic.

Planned improvements:
- Tighten Costas gating thresholds after broader evaluation, and consider adaptive candidate limits.
- Add lightweight heuristics for weak-signal handling (noise floor estimation, soft-windowing tweaks) without deviating from spec.
- Runtime: profile and accelerate hot loops (vectorization, optional Numba), keeping correctness paramount.
