from dataclasses import dataclass


@dataclass(frozen=True)
class FT8Constants:
    # See QEX paper: tone spacing 6.25 Hz, 79 symbols + 21 Costas sync = 100 symbols
    tone_spacing_hz: float = 6.25
    symbol_period_s: float = 0.160
    num_tones: int = 8
    num_message_symbols: int = 79
    num_sync_symbols: int = 21  # 3 Costas arrays of 7 each
    frame_symbols: int = 100
    payload_bits: int = 77
    crc_bits: int = 14
    ldpc_n: int = 174
    ldpc_k: int = 87
    fs_default_hz: int = 12000  # common demod sample rate
