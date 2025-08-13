#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ft8gpt import decode_wav  # noqa: E402


def parse_wsjt_lines(path: Path) -> Set[str]:
    lines = []
    if not path.exists():
        return set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Extract trailing message after '~' if present
        if "~" in line:
            msg = line.split("~", 1)[1].strip()
        else:
            parts = line.split()
            msg = " ".join(parts[5:]) if len(parts) > 5 else ""
        if msg:
            lines.append(msg)
    return set(lines)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def main() -> int:
    dataset = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
    if not dataset.exists():
        print("Dataset not found:", dataset)
        return 1
    wavs = sorted(dataset.glob("*.wav"))
    out_dir = Path("/workspace/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dataset_metrics.json"

    total_expected = 0
    matched = 0
    missed = 0
    extras = 0
    per_file = []

    for idx, wav in enumerate(wavs, 1):
        exp = {norm(s) for s in parse_wsjt_lines(wav.with_suffix(".txt"))}
        got = decode_wav(str(wav))
        got_msgs = {norm(r.message) for r in got if r.message}
        m = len(exp & got_msgs)
        x = len(got_msgs - exp)
        total_expected += len(exp)
        matched += m
        extras += x
        missed += len(exp) - m
        per_file.append({
            "file": wav.name,
            "expected": len(exp),
            "got": len(got_msgs),
            "matched": m,
            "extras": x,
        })
        print(f"[{idx}/{len(wavs)}] {wav.name}: expected={len(exp)} got={len(got_msgs)} matched={m} extras={x}")
        sys.stdout.flush()

    rate = (matched / total_expected) if total_expected else 0.0
    summary = {
        "files": len(wavs),
        "expected_total": total_expected,
        "matched": matched,
        "missed": missed,
        "extras": extras,
        "match_rate": rate,
        "per_file": per_file,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(f"Match rate: {rate * 100.0:.2f}% ({matched}/{total_expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())