"""Stateless EEG metric functions for real-time quality monitoring."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import RTConfig
from scipy.signal import welch
from fooof import FOOOF

def line_noise_db(
    window: NDArray[np.float64],
    cfg: RTConfig,
) -> tuple[float, dict]:
    """FOOOF-based periodic amplitude at the line-noise peak (dB)."""


    n_ch, n_samp = window.shape
    nperseg = min(n_samp, int(cfg.sfreq * 2.0))

    freqs, psd = welch(
        window,
        fs=cfg.sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        scaling="density",
    )
    mean_psd = psd.mean(axis=0)

    fit_lo, fit_hi = cfg.fooof_fit_range
    freq_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
    f_fit = freqs[freq_mask]
    p_fit = mean_psd[freq_mask]

    fm = FOOOF(
        peak_width_limits=[1, 8],
        max_n_peaks=8,
        min_peak_height=0.01,
        peak_threshold=2.0,
        aperiodic_mode="fixed",
        verbose=False,
    )
    fm.fit(f_fit, p_fit, cfg.fooof_fit_range)

    lo, hi = cfg.fooof_peak_band
    peaks = fm.get_params("peak_params")
    raw_db = float(np.nan)
    f_peak_hz = float(np.nan)

    if peaks is not None and peaks.ndim == 2:
        in_band = (peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)
        if in_band.any():
            best = int(peaks[in_band, 1].argmax())
            raw_db = float(peaks[in_band, 1][best])
            f_peak_hz = float(peaks[in_band, 0][best])

    ap_params = fm.get_params("aperiodic_params")
    ap_exponent = float(ap_params[-1]) if ap_params is not None else float(np.nan)

    detail = {
        "raw_db": raw_db,
        "f_peak_hz": f_peak_hz,
        "ap_exponent": ap_exponent,
    }
    return raw_db, detail


def channel_kurtosis(window: NDArray[np.float64]) -> tuple[float, dict]:
    """|Excess kurtosis| per channel, mean across channels."""
    from scipy.stats import kurtosis

    per_ch = np.abs(kurtosis(window, axis=1))
    scalar = float(per_ch.mean())
    return scalar, {"per_channel": per_ch.tolist()}


def channel_rms(window: NDArray[np.float64]) -> tuple[float, dict]:
    """RMS amplitude (uV) per channel, mean across channels."""
    per_ch = np.sqrt(np.mean(window ** 2, axis=1))
    scalar = float(per_ch.mean())
    return scalar, {"per_channel": per_ch.tolist()}


def channel_max_amplitude(window: NDArray[np.float64]) -> tuple[float, dict]:
    """Peak-to-peak amplitude per window, mean across channels."""
    per_ch = window.max(axis=1) - window.min(axis=1)
    scalar = float(per_ch.mean())
    return scalar, {"per_channel": per_ch.tolist()}


def channel_bridged(
    window: NDArray[np.float64],
    cfg: RTConfig,
) -> tuple[float, dict]:
    """Number of bridged channel pairs in this window.

    window: (C, T). Each channel is z-scored along time before the electrical
    distance is computed, so the metric is invariant to channel amplitude and
    cannot be fooled by two simultaneously-flat channels (independent
    low-amplitude noise still has near-zero correlation after z-scoring).

    After z-scoring, the closed form
        var(z_a - z_b) = var(z_a) + var(z_b) - 2*cov(z_a, z_b)
    collapses to
        ed[i, j] = 2 * (1 - corr(i, j))
    so cfg.bridge_ed_threshold lives in correlation-distance units (range
    [0, 4], dimensionless). A textbook bridge has corr >= 0.95, i.e. ed <= 0.1.

    Channels whose raw std is below `_BRIDGE_STD_FLOOR_UV` are treated as
    truly-flat and excluded from pair counting -- we cannot speak of a
    correlation for a constant signal.

    Returned scalar is the integer pair count (cast to float so it can flow
    through the EWMA + hysteresis pipeline like every other metric here).
    """
    std = window.std(axis=1)                                # (C,)
    valid = std > _BRIDGE_STD_FLOOR_UV                      # (C,) bool
    z = (window - window.mean(axis=1, keepdims=True)) / np.maximum(
        std[:, None], 1e-12
    )                                                       # (C, T)

    corr = np.cov(z, bias=True)                             # (C, C), == correlation
    ed = 2.0 * (1.0 - corr)                                 # (C, C), in [0, 4]

    valid_pair = valid[:, None] & valid[None, :]            # (C, C)
    mask = np.triu((ed < cfg.bridge_ed_threshold) & valid_pair, k=1)
    pairs = [(int(i), int(j)) for i, j in np.argwhere(mask)]
    n_pairs = len(pairs)
    return float(n_pairs), {"n_pairs": n_pairs, "pairs": pairs}


# Channels with std below this floor (in input units, typically uV) are
# excluded from bridge pair counting. Set well above float-precision noise
# but below any plausible real-EEG noise floor.
_BRIDGE_STD_FLOOR_UV: float = 0.1


