"""Real-time pipeline configuration — single flat dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RTConfig:
    """All tunable constants for the real-time EEG quality monitor.

    Metric names emitted by the engine (the `metric` field on WindowState):
        "excess line noise", "low RMS", "excess p2p", "bridge detected",
        "excess kurtosis".
    Internally the engine uses identifier-friendly short keys
    (line_noise_detected, is_flat, high_peak2peak, is_bridged, kurtosis)
    which name the matching <name>_window_sec, <name>_step_sec,
    ewma_beta_<name>, <name>_enter, <name>_exit attributes below.
    """

    # Signal
    sfreq: float = 500.0
    n_channels: int = 128

    # Global defaults — every per-metric window/step/beta below falls back
    # to these unless explicitly overridden.
    global_step_sec: float = 0.1
    global_window_sec: float = 4.0
    global_beta: float = 0.8

    # Window sizing — per metric (seconds of EEG fed to the metric function)
    line_noise_detected_window_sec: float = global_window_sec
    kurtosis_window_sec: float = global_window_sec
    is_flat_window_sec: float = global_window_sec
    high_peak2peak_window_sec: float = global_window_sec
    is_bridged_window_sec: float = global_window_sec

    # Step sizing — per metric (how often the metric is re-evaluated)
    line_noise_detected_step_sec: float = global_step_sec
    kurtosis_step_sec: float = global_step_sec
    is_flat_step_sec: float = global_step_sec
    high_peak2peak_step_sec: float = global_step_sec
    is_bridged_step_sec: float = global_step_sec

    # FOOOF fitting parameters (line-noise metric only)
    fooof_fit_range: tuple[float, float] = (20.0, 80.0)
    fooof_peak_band: tuple[float, float] = (59.0, 61.0)

    # EWMA smoothing beta — one per EEG metric
    ewma_beta_line_noise_detected: float = global_beta
    ewma_beta_kurtosis: float = global_beta
    ewma_beta_is_flat: float = global_beta
    ewma_beta_high_peak2peak: float = global_beta
    # Bridge metric bypasses EWMA (see _NO_EWMA_METRICS in alerts.py).
    # This value is therefore unused; left here so the per-metric beta map
    # in alerts.py can still be built without a special case.
    ewma_beta_is_bridged: float = global_beta

    # Hysteresis thresholds (applied to EWMA-smoothed values)
    line_noise_detected_enter: float = 4.0
    line_noise_detected_exit: float = 3.5

    high_peak2peak_enter: float = 200.0
    high_peak2peak_exit: float = 150.0

    kurtosis_enter: float = 5.0
    kurtosis_exit: float = 4.0

    # is_flat uses inverted logic: BAD when smoothed RMS < enter, GOOD when >= exit.
    is_flat_enter: float = 1e-3
    is_flat_exit: float = 1e-3

    # Bridge detection.
    # Channels are z-scored before computing electrical distance, so
    # bridge_ed_threshold is in correlation-distance units:
    #     ed = 2 * (1 - corr)   ->  range [0, 4], dimensionless.
    # 0.1 corresponds to corr >= 0.95 (textbook bridge cutoff).
    bridge_ed_threshold: float = 0.1
    # Pair-count thresholds, applied directly to the (un-smoothed) integer
    # pair count. enter=0.5 means "raw count >= 1 trips" (strict-`>` in
    # alerts.py); exit=0.5 means "raw count == 0 clears". Hysteresis is
    # provided entirely by min_steps_to_enter / min_steps_to_clear.
    is_bridged_enter: float = 0.5
    is_bridged_exit: float = 0.5

    # Persistence — minimum consecutive steps before state transitions
    min_steps_to_enter: int = 2
    min_steps_to_clear: int = 3

    # Cooldown — minimum seconds between repeated alerts for same metric
    alert_cooldown_sec: float = 30.0

    # Impedance (no EWMA, no hysteresis)
    impedance_bad_threshold_kohm: float = 30.0
