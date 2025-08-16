from pathlib import Path
import hashlib
import re
import time
import pytest
import os

from ft8gpt import decode_wav


def _parse_expected_sets(path: Path) -> tuple[set[str], set[str]]:
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
		msg_clean = msg.lstrip("!- ")
		(neg if negative_marker else pos).add(msg_clean)
	return pos, neg


def _select_fixed_20_percent(wavs: list[Path], base: Path) -> list[Path]:
	selected: list[Path] = []
	for p in wavs:
		rel = p.relative_to(base)
		h = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()
		if int(h[-8:], 16) % 5 == 0:
			selected.append(p)
	return selected


def _write_short_summary(total_expected: int, total_matched: int, reason: str | None = None) -> None:
	path = os.environ.get("SHORT_SUMMARY_PATH")
	if not path:
		return
	try:
		p = Path(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		with p.open("w", encoding="utf-8") as f:
			line = f"pos_expected={total_expected} pos_matched={total_matched}"
			if reason:
				line += f" reason={reason}"
			f.write(line + "\n")
	except Exception:
		pass


def test_short_dataset_regression_20pct():
	dataset = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
	if not dataset.exists():
		print("SHORT TOTAL: pos_expected=0 pos_matched=0 (dataset-not-available)")
		_write_short_summary(0, 0, reason="dataset-not-available")
		pytest.skip("dataset not available")
	wavs = sorted(dataset.rglob("*.wav"))
	if not wavs:
		print("SHORT TOTAL: pos_expected=0 pos_matched=0 (no-wav-files)")
		_write_short_summary(0, 0, reason="no-wav-files")
		pytest.skip("no wav files")

	sample = _select_fixed_20_percent(wavs, dataset)
	if not sample:
		print("SHORT TOTAL: pos_expected=0 pos_matched=0 (no-files-selected)")
		_write_short_summary(0, 0, reason="no-files-selected")
		pytest.skip("no files selected in 20% sampling")

	total_pos_expected = 0
	total_pos_matched = 0
	t0 = time.time()
	for wav in sample:
		pos_expected, neg_expected = _parse_expected_sets(wav.with_suffix(".txt"))
		if not pos_expected and not neg_expected:
			continue
		results = decode_wav(str(wav))
		got = {r.message for r in results if getattr(r, "crc14_ok", False) and r.message}
		matched_pos = len(pos_expected & got)
		total_pos_expected += len(pos_expected)
		total_pos_matched += matched_pos
		# Optional: debug print per-file stats
		print(f"short {wav.relative_to(dataset)}: pos_expected={len(pos_expected)} pos_matched={matched_pos}")

	print(f"SHORT TOTAL: pos_expected={total_pos_expected} pos_matched={total_pos_matched}")
	_write_short_summary(total_pos_expected, total_pos_matched)

	# Ratchet for CI: require at least one exact-text, CRC-valid decode across the 20% sample
	assert total_pos_matched >= 16, "expected at least sixteen matched decodes in 20% short regression"

	# Keep runtime guardrail similar to full test but this sample should be much faster
	avg_runtime = (time.time() - t0) / max(1, len(sample))
	assert avg_runtime < 10.0