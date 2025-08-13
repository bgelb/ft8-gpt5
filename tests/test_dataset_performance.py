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
def test_dataset_decode_success_rate_and_runtime_distribution(capsys, tmp_path):
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
    per_file: List[dict] = []

    for wav in wavs:
        expected_msgs = _parse_wsjt_lines(wav.with_suffix(".txt"))
        expected_set = set(expected_msgs)
        total_expected += len(expected_set)

        t0 = time.perf_counter()
        got = decode_wav(str(wav))
        dt = time.perf_counter() - t0
        durations.append((wav.name, dt))

        got_msgs_set = {_normalize_msg(r.message) for r in got if getattr(r, "message", "")}
        got_msgs = sorted(got_msgs_set)
        total_decoded += len(got_msgs)

        matched_set = got_msgs_set & expected_set
        matched = sorted(list(matched_set))
        matched_count = len(matched)
        total_matched += matched_count

        expected_sorted = sorted(list(expected_set))
        expected_not_decoded_set = expected_set - matched_set
        expected_not_decoded = sorted(list(expected_not_decoded_set))

        match_rate_file = (matched_count / len(expected_set)) if len(expected_set) > 0 else 0.0

        per_file.append({
            "file": wav.name,
            "runtime_seconds": dt,
            "expected_count": len(expected_sorted),
            "decoded_count": len(got_msgs),
            "matched_count": matched_count,
            "match_rate": match_rate_file,
            # Expected-centric visibility
            "expected_decoded_success_count": matched_count,
            "expected_not_decoded_count": len(expected_sorted) - matched_count,
            "expected_messages": expected_sorted,
            "expected_not_decoded_messages": expected_not_decoded,
            # Decoded/matched listings for deeper inspection
            "decoded_messages": got_msgs,
            "matched_messages": matched,
        })

    # Compute success metrics (aggregate)
    match_rate = (total_matched / total_expected) if total_expected > 0 else 0.0

    # Compute runtime metrics (aggregate)
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

    # Compose JSON result with per-file details and aggregate
    result = {
        "aggregate": {
            "num_wavs": len(wavs),
            "total_expected": total_expected,
            "total_decoded": total_decoded,
            "total_matched": total_matched,
            "total_expected_decoded_success": total_matched,
            "total_expected_not_decoded": total_expected - total_matched,
            "match_rate": match_rate,
            "runtime_seconds": {
                "median": median_s,
                "mean": mean_s,
                "p90": p90_s,
                "p95": p95_s,
                "max": max_s,
            },
            "slowest": [{"file": n, "runtime_seconds": s} for n, s in slowest],
        },
        "files": per_file,
    }

    # Write JSON artifact to tmp and optional env override path
    json_path = tmp_path / "perf_results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Perf JSON written to: {json_path}")

    if (out_path := os.getenv("FT8_PERF_JSON_OUT")):
        try:
            Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Perf JSON also written to: {out_path}")
        except Exception as e:
            print(f"Warning: failed to write FT8_PERF_JSON_OUT: {e}")

    # Flush prints to test output
    capsys.readouterr()

    # Ratchets (aggregate pass/fail)
    assert match_rate >= baselines["min_match_rate"], (
        f"match_rate {match_rate:.3f} fell below baseline {baselines['min_match_rate']:.3f}"
    )
    assert mean_s <= baselines["max_mean_seconds"], (
        f"mean {mean_s:.3f}s exceeded baseline {baselines['max_mean_seconds']:.3f}s"
    )
    assert p95_s <= baselines["max_p95_seconds"], (
        f"p95 {p95_s:.3f}s exceeded baseline {baselines['max_p95_seconds']:.3f}s"
    )