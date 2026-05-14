"""EEG simulation functions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from collections import deque
from scipy.ndimage import gaussian_filter1d

# ── CONFIG ──────────────────────────────────────────────────────────────────
N_CHANNELS = 10
SFREQ = 1000
DURATION = 180
CHUNK_SEC = 5.0
NOISE_RMS_UV = 20
SEED = 42

LINE_FREQ_HZ = 60.0
LINE_AMP_UV = 40
LINE_ON_SEC = (35, 55)
LINE_BASELINE_FRAC = 1.2   # must be > 1
LINE_RAMP_SIGMA_SEC = 0.1

DEAD_CHANNELS = 8
FLAT_ON_SEC = (60, 65)
PEAK_FRAC = 2.0
MOVEMENT_UV = 300
MOVE_ON_SEC = (85, 90)

SPIKE_PROB = 0.05
SPIKE_ON_SEC = (100, 115)

# Bridge: two channels share a 5 Hz sinusoid + small independent jitter,
# producing near-perfect correlation while preserving plausible amplitudes.
BRIDGE_FREQ_HZ = 5.0
BRIDGE_AMP_UV = 60.0
BRIDGE_JITTER_UV = 2.0
BRIDGE_ON_SEC = (130, 150)
BRIDGE_CHANNELS = (4, 5)


# ── EEG simulation functions ────────────────────────────────────────────────

def _pink_noise(n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal((n_samples, n_channels))
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0
    filt = 1.0 / np.sqrt(freqs)
    filt[0] = 0.0
    spectrum = np.fft.rfft(white, axis=0) * filt[:, np.newaxis]
    sig = np.fft.irfft(spectrum, n=n_samples, axis=0)

    sig -= sig.mean(axis=0, keepdims=True)
    sig /= (sig.std(axis=0, keepdims=True) + 1e-12)
    return sig.astype(np.float64)


def _smooth_line_envelope(
    t: np.ndarray,
    on_sec: tuple[float, float],
    baseline_frac: float = 0.15,
    peak_frac: float = 1.0,
    ramp_sigma_sec: float = 1.0,
) -> np.ndarray:
    env = np.full_like(t, baseline_frac, dtype=float)
    env[(t >= on_sec[0]) & (t < on_sec[1])] = peak_frac

    dt = np.median(np.diff(t))
    sigma_samples = max(ramp_sigma_sec / dt, 1.0)
    env = gaussian_filter1d(env, sigma=sigma_samples, mode="nearest")
    return env


def _make_signal():
    rng = np.random.default_rng(SEED)
    n_samples = int(DURATION * SFREQ)
    t = np.arange(n_samples) / SFREQ

    signal = _pink_noise(n_samples, N_CHANNELS, rng) * NOISE_RMS_UV

    # Simulate Line noise: always present, but stronger during LINE_ON_SEC
    line_env = _smooth_line_envelope(
        t,
        on_sec=LINE_ON_SEC,
        baseline_frac=LINE_BASELINE_FRAC,
        peak_frac=PEAK_FRAC,
        ramp_sigma_sec=LINE_RAMP_SIGMA_SEC,
    )
    line_gt = (t >= LINE_ON_SEC[0]) & (t < LINE_ON_SEC[1])

    phase = rng.uniform(0, 2 * np.pi, size=(1, N_CHANNELS))
    channel_gain = rng.uniform(0.8, 1.2, size=(1, N_CHANNELS))
    sin60 = np.sin(2 * np.pi * LINE_FREQ_HZ * t[:, None] + phase)
    signal += LINE_AMP_UV * line_env[:, None] * sin60 * channel_gain

    # Simulate Flat/dead channels
    flat_on = (t >= FLAT_ON_SEC[0]) & (t < FLAT_ON_SEC[1])
    signal[np.ix_(flat_on, np.arange(DEAD_CHANNELS))] *= 1e-2

    # Simulate Movement burst
    move_on = (t >= MOVE_ON_SEC[0]) & (t < MOVE_ON_SEC[1])
    signal[move_on, :] += rng.standard_normal((move_on.sum(), N_CHANNELS)) * MOVEMENT_UV

    # Simulate Spiking artifacts (e.g. head movement, electrode pops, sharp change in impedance)
    spike_on = (t >= SPIKE_ON_SEC[0]) & (t < SPIKE_ON_SEC[1])
    spike_idx = np.where(spike_on)[0]
    n_spikes = int(len(spike_idx) * SPIKE_PROB)
    chosen = rng.choice(spike_idx, size=n_spikes, replace=False)
    signal[chosen, :] += rng.choice([-1, 1], size=(n_spikes, N_CHANNELS)) * 800.0

    # Simulate Bridged electrodes: two channels share a 5 Hz sinusoid plus
    # small independent jitter -> very high correlation, low electrical
    # distance after z-scoring.
    bridge_on = (t >= BRIDGE_ON_SEC[0]) & (t < BRIDGE_ON_SEC[1])
    n_bridge = int(bridge_on.sum())
    bridge_sine = BRIDGE_AMP_UV * np.sin(
        2 * np.pi * BRIDGE_FREQ_HZ * t[bridge_on]
    )
    for ci in BRIDGE_CHANNELS:
        signal[bridge_on, ci] = bridge_sine + (
            rng.standard_normal(n_bridge) * BRIDGE_JITTER_UV
        )

    return (
        signal.astype(np.float64),
        t,
        line_gt,
        flat_on,
        move_on,
        spike_on,
        bridge_on,
    )
