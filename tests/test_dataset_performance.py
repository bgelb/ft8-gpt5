import os
import json
import statistics
import time
from pathlib import Path
import re
from typing import List, Tuple

import pytest

from ft8gpt import decode_wav


DATASET_DIR = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
BASELINE_PATH = Path(__file__).resolve().parent / "perf_baseline.json"


def _normalize_msg(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def _parse_wsjt_lines(path: Path) -> List[str]:
    messages: List[str] = []
    if not path.exists():
        return messages
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if "~" in line:
            msg = line.split("~", 1)[1].strip()
        else:
            parts = line.split()
            msg = " ".join(parts[5:]) if len(parts) > 5 else ""
        if msg:
            messages.append(_normalize_msg(msg))
    return messages


def _load_baseline() -> dict:
    # Defaults are intentionally loose; can be tightened over time.
    defaults = {
        "min_match_rate": 0.0,   # fraction [0,1] of expected messages recovered
        "max_mean_seconds": 5.0, # average seconds per 15s WAV
        "max_p95_seconds": 10.0, # p95 seconds per 15s WAV
    }
    # Env overrides take precedence
    env_overrides = {}
    if (v := os.getenv("FT8_PERF_MIN_MATCH_RATE")) is not None:
        env_overrides["min_match_rate"] = float(v)
    if (v := os.getenv("FT8_PERF_MAX_MEAN_S")) is not None:
        env_overrides["max_mean_seconds"] = float(v)
    if (v := os.getenv("FT8_PERF_MAX_P95_S")) is not None:
        env_overrides["max_p95_seconds"] = float(v)

    try:
        with BASELINE_PATH.open("r", encoding="utf-8") as f:
            file_cfg = json.load(f)
    except FileNotFoundError:
        file_cfg = {}
    # Merge with priority: defaults < baseline file < env overrides
    cfg = {**defaults, **file_cfg, **env_overrides}
    return cfg


@pytest.mark.regression
@pytest.mark.slow
def test_dataset_decode_success_rate_and_runtime_distribution(capsys):
    if not DATASET_DIR.exists():
        pytest.skip("dataset not available")

    baselines = _load_baseline()

    wavs = sorted(DATASET_DIR.glob("*.wav"))
    if not wavs:
        pytest.skip("no wav files found")

    total_expected = 0
    total_matched = 0
    total_decoded = 0

    durations: List[Tuple[str, float]] = []

    for wav in wavs:
        expected_msgs = _parse_wsjt_lines(wav.with_suffix(".txt"))
        expected_set = set(expected_msgs)
        total_expected += len(expected_set)

        t0 = time.perf_counter()
        got = decode_wav(str(wav))
        dt = time.perf_counter() - t0
        durations.append((wav.name, dt))

        got_msgs = {_normalize_msg(r.message) for r in got if getattr(r, "message", "")}
        total_decoded += len(got_msgs)
        total_matched += len(got_msgs & expected_set)

    # Compute success metrics
    match_rate = (total_matched / total_expected) if total_expected > 0 else 0.0

    # Compute runtime metrics
    times = [d for _, d in durations]
    mean_s = statistics.fmean(times)
    median_s = statistics.median(times)
    p90_s = statistics.quantiles(times, n=10)[8] if len(times) >= 2 else mean_s
    p95_s = statistics.quantiles(times, n=20)[18] if len(times) >= 2 else mean_s
    max_s = max(times)

    # Print concise, at-a-glance summary
    print(
        f"Decode success: {total_matched}/{total_expected} ({match_rate*100:.1f}%) | "
        f"decoded msgs: {total_decoded} across {len(wavs)} wavs"
    )
    # Topline runtime summary
    print(
        f"Runtime per 15s WAV (s): median={median_s:.3f} mean={mean_s:.3f} p90={p90_s:.3f} p95={p95_s:.3f} max={max_s:.3f} | N={len(wavs)}"
    )
    # Show slowest 5 files for quick inspection
    slowest = sorted(durations, key=lambda kv: kv[1], reverse=True)[:5]
    if slowest:
        lbl = ", ".join([f"{name}:{secs:.3f}s" for name, secs in slowest])
        print(f"Slowest: {lbl}")

    # Flush prints to test output
    capsys.readouterr()

    # Ratchets
    assert match_rate >= baselines["min_match_rate"], (
        f"match_rate {match_rate:.3f} fell below baseline {baselines['min_match_rate']:.3f}"
    )
    assert mean_s <= baselines["max_mean_seconds"], (
        f"mean {mean_s:.3f}s exceeded baseline {baselines['max_mean_seconds']:.3f}s"
    )
    assert p95_s <= baselines["max_p95_seconds"], (
        f"p95 {p95_s:.3f}s exceeded baseline {baselines['max_p95_seconds']:.3f}s"
    )