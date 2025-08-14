# Agent Operating Rules

- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
- Only terminate your turn when you are sure that the problem is solved.
- Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
- Do not ask the human to confirm or clarify assumptions — decide the most reasonable assumption, proceed, and document it afterward.
- Batch repeated actions via scripts where possible (e.g., run tests → fix → re-run loop).
- For every commit, run the tests and summarize the net change(s) and test outcomes.
  - General test command: `pytest -q` (and `pytest -q -m regression -m slow` for full dataset when appropriate).
- Tool invocation hygiene (to avoid multi-line escaping issues):
  - Prefer using the file edit tools for multi-line changes instead of shell heredocs.
  - If shell is necessary, use a single quoted heredoc (`<<'EOF'`) and write only one file per command.
  - Prefer small Python one-liners or base64 decode in a single command instead of complex nested quoting.
  - Re-read files before editing and avoid overwriting unknown contents.
  - Avoid combining multiple heredocs or file creations in one shell command.
- Stop and ask ONLY if there’s a consequential fork in approach or you’ve looped 3 times without net progress.

# Style/Coding Guidelines

- Use Python 3.11+.
- Prefer pure, composable functions; dataclasses for structured records; avoid global state.
- Include units in variable names (e.g., `sample_rate_hz`, `symbol_period_s`).
- Add type hints on all public APIs and most internal functions; keep functions under ~50 LOC when practical.
- Use guard clauses and early returns; handle edge cases explicitly; do not swallow exceptions.
- Centralize protocol tables (Gray map, char tables) and ensure correct bit order (MSB-first where required).
- Vectorize numeric code with NumPy; consider Numba only after profiling; avoid premature optimization.
- Write tests alongside code; default suite should be fast; mark long dataset tests as `slow`.
- Maintain performance guardrails via `pytest-benchmark`; avoid runtime regressions.
- Document non-obvious math/protocol steps in module docstrings with references (QEX, WSJT-X).

# Cursor Agents Guidelines for This Repo

- Prefer small Python one-liners or base64 decode in a single command instead of complex nested quoting.
- Do not introduce external service calls without explicit approval.
- Keep CI fast; mark long-running tests with `@pytest.mark.slow` and exclude by default.
- Use only the `external/ft8_lib/test/wav` dataset files (.wav and .txt) for tests. Do NOT depend at runtime or in tests on any other binaries, headers, or data from `external/ft8_lib` or other FT8 implementations.
- With the sole exception of using the sample `.wav` and `.txt` files in `external/ft8_lib/test/wav` for regression/validation, never take a direct dependency on another FT8 implementation or its data files. Embed any required small tables directly in this repository.

# Background Agents

This repository uses background agents to automate parts of the development and testing workflow.

## Responsibilities

- Maintain code quality and consistency
- Keep the test suite green
- Respect performance and reproducibility constraints

## Rules

- Do not make breaking API changes without bumping versions and updating downstream usages
- Keep edits localized and minimal; avoid broad refactors unless explicitly requested
- Prefer deterministic algorithms and reproducible pipelines
- When running commands, always use non-interactive flags
- Avoid long-running background jobs unless they are necessary for the task
- Keep artifacts and caches out of the repository
- Prefer reading context with search tools over guessing
- After code edits, run tests/build and fix failures before marking complete
- Do not add commentary inside code blocks just to explain actions
- Use `src/` for implementation and `tests/` for tests
- Submodules: update only when required and pin to known-good commits
- When in doubt, follow the FT8 specification and reference implementations for protocol details
- All mappings, translations, and transformations must be aligned with the FT8 specification. Do not add generic "fallback" logic that tries multiple permutations/sign inversions or alternate paths. Implement a single consistent decoding path used across all inputs.
