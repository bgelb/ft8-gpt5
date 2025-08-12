from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Decode:
    start_time_s: float
    snr_db: float
    freq_hz: float
    drift_hz_per_s: float
    message_text: str
    crc_ok: bool
    time_offset_s: Optional[float] = None
