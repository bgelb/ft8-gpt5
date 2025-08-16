from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import (
    LENGTH_SYNC,
    NUM_SYNC,
    SYNC_OFFSET,
    FT8_COSTAS_PATTERN,
    FSK_TONES,
    SYMBOL_PERIOD_S,
    TONE_SPACING_HZ,
)
from scipy.signal import get_window
from numpy.lib.stride_tricks import as_strided


@dataclass(frozen=True)
class StftCandidate:
    frame_start: int
    base_bin: int
    frac: float
    score: float


def _whiten_power_linear(P: NDArray[np.float64]) -> NDArray[np.float64]:
    if P.size == 0:
        return P
    med_f = np.median(P, axis=0, keepdims=True)
    Pw = np.maximum(P - med_f, 0.0)
    med2_f = np.median(Pw, axis=0, keepdims=True)
    mad_f = np.median(np.abs(Pw - med2_f), axis=0, keepdims=True) + 1e-12
    Pw = Pw / mad_f
    med_t = np.median(Pw, axis=1, keepdims=True)
    Pw = np.maximum(Pw - med_t, 0.0)
    return Pw


def _compute_stft_power_linear(signal: NDArray[np.float64], sample_rate_hz: float) -> Tuple[NDArray[np.float64], int, int, int]:
    win_len = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    win_len = max(16, win_len)
    # Half-symbol hop (T/2)
    hop = max(1, win_len // 2)
    # Half-bin frequency resolution via 2x zero-padding
    n_fft = int(2 * win_len)
    win = get_window("hann", win_len, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))

    # Optional time padding for wider search; disabled by default in characterization
    x = signal.astype(np.float64, copy=False)
    if x.size < win_len:
        x = np.pad(x, (0, win_len - x.size))
    n = x.size
    num_frames = 1 + max(0, (n - win_len) // hop)
    if num_frames <= 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float64), n_fft, hop, win_len
    frames = as_strided(x, shape=(num_frames, win_len), strides=(x.strides[0] * hop, x.strides[0]))
    frames_w = frames * win[None, :]
    spec = np.fft.rfft(frames_w, n=n_fft, axis=1)
    p = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
    pw = _whiten_power_linear(p)
    return pw, n_fft, hop, win_len


def find_sync_candidates_stft(signal: NDArray[np.float64], sample_rate_hz: float, top_k: int = 300) -> Tuple[List[StftCandidate], int, int]:
    pwr, n_fft, hop, _win_len = _compute_stft_power_linear(signal, sample_rate_hz)
    num_frames, num_bins = pwr.shape

    bin_hz = sample_rate_hz / float(n_fft) if n_fft > 0 else TONE_SPACING_HZ / 2.0
    tone_stride_bins = max(1, int(round(TONE_SPACING_HZ / max(bin_hz, 1e-9))))
    num_bases = max(0, num_bins - (FSK_TONES - 1) * tone_stride_bins)

    bank = np.empty((num_frames, num_bases, FSK_TONES), dtype=np.float64)
    for j in range(FSK_TONES):
        ofs = j * tone_stride_bins
        bank[:, :, j] = pwr[:, ofs:ofs + num_bases]
    # Half-bin interpolated bank (average adjacent bins)
    num_bases_h = max(0, num_bins - 1 - (FSK_TONES - 1) * tone_stride_bins)
    bankh = np.empty((num_frames, num_bases_h, FSK_TONES), dtype=np.float64)
    for j in range(FSK_TONES):
        ofs = j * tone_stride_bins
        # average between bin ofs and ofs+1 for half-step
        a = pwr[:, ofs:ofs + num_bases_h]
        b = pwr[:, ofs + 1:ofs + 1 + num_bases_h]
        bankh[:, :, j] = 0.5 * (a + b)

    frames_per_symbol = max(1, int(round((SYMBOL_PERIOD_S * sample_rate_hz) / hop)))
    frames_per_symbol = max(frames_per_symbol, 1)
    window_len_symbols = LENGTH_SYNC + (NUM_SYNC - 1) * SYNC_OFFSET
    max_start_frame = num_frames - (window_len_symbols - 1) * frames_per_symbol
    if max_start_frame <= 0:
        return ([], n_fft, hop)

    offs = np.array([SYNC_OFFSET * m + k for m in range(NUM_SYNC) for k in range(LENGTH_SYNC)], dtype=np.int64)
    offs *= frames_per_symbol
    tone_idx = np.array([FT8_COSTAS_PATTERN[k] for _m in range(NUM_SYNC) for k in range(LENGTH_SYNC)], dtype=np.int64)

    def score_grid(bank: NDArray[np.float64], base_limit: int, alpha: float = 1.0) -> NDArray[np.float64]:
        if base_limit <= 0:
            return np.zeros((0, 0), dtype=np.float64)
        grid = np.full((max_start_frame, base_limit), -1e30, dtype=np.float64)
        for t0 in range(max_start_frame):
            frames = t0 + offs
            valid = (frames >= 0) & (frames < num_frames)
            if not np.any(valid):
                continue
            frames = np.clip(frames, 0, num_frames - 1)
            B = bank[frames, :base_limit, :]
            # Compute Csig and Sall across all 21 Costas frames, then form
            # C = Csig - alpha * Cbg where Cbg = (Sall - Csig) / 7
            E_sig = np.take_along_axis(B, tone_idx[:, None, None], axis=2)[..., 0]  # [21, base]
            S_all = B.sum(axis=2)  # [21, base]
            Csig = E_sig.sum(axis=0)  # [base]
            Sall = S_all.sum(axis=0)  # [base]
            Cbg = (Sall - Csig) / max(FSK_TONES - 1, 1)
            E = Csig - alpha * Cbg  # [base]
            nvalid = float(np.sum(valid))
            if nvalid <= 0:
                nvalid = 1.0
            grid[t0, :] = E / nvalid
        return grid

    grid0 = score_grid(bank, num_bases)
    gridh = score_grid(bankh, num_bases_h)

    def peaks_along_bases(G: NDArray[np.float64]) -> NDArray[np.bool_]:
        if G.size == 0:
            return np.zeros_like(G, dtype=bool)
        H, W = G.shape
        if W < 3 or H == 0:
            return np.zeros_like(G, dtype=bool)
        left = G[:, :-2]
        center = G[:, 1:-1]
        right = G[:, 2:]
        mask_mid = (center > left) & (center > right)
        peaks = np.zeros_like(G, dtype=bool)
        peaks[:, 1:-1] = mask_mid
        return peaks

    cand_list: List[StftCandidate] = []
    if grid0.size:
        pk = peaks_along_bases(grid0)
        ys, xs = np.nonzero(pk)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cand_list.append(StftCandidate(frame_start=int(y), base_bin=int(x), frac=0.0, score=float(grid0[y, x])))
    if gridh.size:
        pkh = peaks_along_bases(gridh)
        ys, xs = np.nonzero(pkh)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cand_list.append(StftCandidate(frame_start=int(y), base_bin=int(x), frac=0.5, score=float(gridh[y, x])))

    # Augment with globally best grid cells to reduce miss from local-peak picking
    items: List[Tuple[float, int, int, float]] = []
    k_extra = max(200, (top_k or 80) * 6)
    if grid0.size:
        g = grid0.copy()
        if g.shape[1] >= 2:
            g[:, 0] = -1e30
            g[:, -1] = -1e30
        idx = np.argsort(g, axis=None)[-k_extra:]
        ys, xs = np.unravel_index(idx, g.shape)
        for s, y, x in zip(g[ys, xs], ys, xs):
            items.append((float(s), int(y), int(x), 0.0))
    if gridh.size:
        gh = gridh.copy()
        if gh.shape[1] >= 2:
            gh[:, 0] = -1e30
            gh[:, -1] = -1e30
        idx = np.argsort(gh, axis=None)[-k_extra:]
        ys, xs = np.unravel_index(idx, gh.shape)
        for s, y, x in zip(gh[ys, xs], ys, xs):
            items.append((float(s), int(y), int(x), 0.5))
    items.sort(key=lambda t: t[0], reverse=True)
    for s, y, x, frac in items:
        cand_list.append(StftCandidate(frame_start=y, base_bin=x, frac=frac, score=s))

    # Deduplicate fractional-bin twins (nearest integer bin key)
    def _round_nearest(x: float) -> int:
        return int(np.floor(x + 0.5))

    by_key: dict[tuple[int, int], StftCandidate] = {}
    for c in cand_list:
        key = (int(c.frame_start), _round_nearest(float(c.base_bin) + float(c.frac)))
        if key not in by_key or c.score > by_key[key].score:
            by_key[key] = c
    deduped = list(by_key.values())

    # Time–frequency NMS
    tone_bins = max(1, int(round(TONE_SPACING_HZ / max(bin_hz, 1e-9))))
    # Allow many frequencies at the same time; suppress only in frequency dimension
    dt = 0
    df = int(round(1.0 * tone_bins))
    deduped.sort(key=lambda c: c.score, reverse=True)
    kept: List[StftCandidate] = []
    for c in deduped:
        suppress = False
        for k in kept:
            if abs(c.frame_start - k.frame_start) <= dt and abs((c.base_bin + c.frac) - (k.base_bin + k.frac)) <= df:
                suppress = True
                break
        if not suppress:
            kept.append(c)

    kept.sort(key=lambda c: c.score, reverse=True)
    if top_k is not None and top_k > 0:
        kept = kept[:top_k]
    return (kept, n_fft, hop)
