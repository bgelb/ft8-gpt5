Coding conventions (ft8dec)

- Names: descriptive and include units where applicable (e.g., `sample_rate_hz`).
- Functions: small, composable, single-responsibility.
- Types: use dataclasses for structured data; favor immutability.
- Errors: raise specific exceptions; never swallow errors silently.
- Tests: write unit tests first; add integration and benchmark tests.
- Style: no inline comments; docstrings explain why; line length ~100.
- Performance: prefer vectorized NumPy; isolate hotspots for Numba/C.
- API: pure functions over globals; pass configuration via dataclasses.

