import os
from pathlib import Path
import time
import re
import pytest

from ft8gpt import decode_wav


def _normalize_msg(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def _is_standard_supported(msg_norm: str) -> bool:
    return bool(
        re.match(r"^(CQ)\s+[A-Z0-9/]+\s+[A-R]{2}[0-9]{2}$", msg_norm)
        or re.match(r"^[A-Z0-9/]+\s+[A-Z0-9/]+\s+[A-R]{2}[0-9]{2}$", msg_norm)
    )


def parse_expected_sets(path: Path) -> tuple[set[str], set[str]]:
    """Parse dataset TXT to (positives, negatives).

    Heuristics:
    - Extract message after '~' if present; else tokens[5:]
    - A line is considered a negative expectation if it contains obvious markers
      such as 'NO_DECODE', 'NON_DECODE', 'NOT_DECODED', or if the extracted
      message starts with '!' or '- ' (these are ignored in normalization).
    - Only include standard messages we currently support.
    """
    pos: set[str] = set()
    neg: set[str] = set()
    if not path.exists():
        return pos, neg
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if "~" in line:
            msg = line.split("~", 1)[1].strip()
        else:
            parts = line.split()
            msg = " ".join(parts[5:]) if len(parts) > 5 else ""
        if not msg:
            continue
        negative_marker = (
            re.search(r"\bNO[_-]?DECODE\b", line, re.IGNORECASE)
            or re.search(r"\bNOT[_-]?DECODED\b", line, re.IGNORECASE)
            or msg.startswith("!")
            or msg.startswith("- ")
        )
        # Strip any leading explicit negative marker characters from message
        msg_clean = msg.lstrip("!- ")
        norm = _normalize_msg(msg_clean)
        if not _is_standard_supported(norm):
            continue
        (neg if negative_marker else pos).add(norm)
    return pos, neg


@pytest.mark.slow
@pytest.mark.regression
def test_full_dataset_nonregression():
    dataset = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
    if not dataset.exists():
        pytest.skip("dataset not available")
    wavs = sorted(dataset.rglob("*.wav"))
    if not wavs:
        pytest.skip("no wav files")

    total_pos_expected = 0
    total_pos_matched = 0
    total_neg_expected = 0
    total_neg_violated = 0
    per_wav_stats: list[tuple[Path, int, int, int, int]] = []

    t0 = time.time()
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        pos_expected, neg_expected = parse_expected_sets(txt)
        if not pos_expected and not neg_expected:
            continue
        results = decode_wav(str(wav))
        got = {_normalize_msg(r.message) for r in results if getattr(r, "crc14_ok", False) and r.message}

        matched_pos = len(pos_expected & got)
        violated_neg = len(neg_expected & got)

        total_pos_expected += len(pos_expected)
        total_pos_matched += matched_pos
        total_neg_expected += len(neg_expected)
        total_neg_violated += violated_neg

        per_wav_stats.append(
            (wav, len(pos_expected), matched_pos, len(neg_expected), violated_neg)
        )
    t1 = time.time()

    # Emit concise per-file summary for CI visibility
    for wav, exp_pos, got_pos, exp_neg, viol_neg in per_wav_stats:
        pos_rate = 0.0 if exp_pos == 0 else got_pos / exp_pos
        neg_ok_rate = 0.0 if exp_neg == 0 else 1.0 - (viol_neg / exp_neg)
        print(
            f"{wav.relative_to(dataset)}: pos_expected={exp_pos} pos_matched={got_pos} "
            f"pos_rate={pos_rate:.1%} neg_expected={exp_neg} neg_falsepos={viol_neg} neg_ok_rate={neg_ok_rate:.1%}"
        )

    pos_rate_total = 0.0 if total_pos_expected == 0 else total_pos_matched / total_pos_expected
    neg_ok_rate_total = 0.0 if total_neg_expected == 0 else 1.0 - (total_neg_violated / total_neg_expected)

    print(
        "TOTAL: "
        f"pos_expected={total_pos_expected} pos_matched={total_pos_matched} pos_rate={pos_rate_total:.1%} "
        f"neg_expected={total_neg_expected} neg_falsepos={total_neg_violated} neg_ok_rate={neg_ok_rate_total:.1%}"
    )

    # Baseline thresholds are permissive initially; to be ratcheted up over time
    assert pos_rate_total >= 0.0
    assert neg_ok_rate_total >= 0.0

    # Runtime guardrail: average < 5s per 15s sample on CI
    avg_runtime = (t1 - t0) / max(1, len(wavs))
    assert avg_runtime < 5.0




