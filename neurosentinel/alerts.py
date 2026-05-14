"""Stateful alerting: EWMA smoothing, hysteresis, persistence, and cooldown."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import RTConfig
from .ewma import EWMA

_EEG_METRICS = (
    "line_noise_detected",
    "kurtosis",
    "is_flat",
    "high_peak2peak",
    "is_bridged",
)

_BETA_MAP = {
    "line_noise_detected": "ewma_beta_line_noise_detected",
    "kurtosis": "ewma_beta_kurtosis",
    "is_flat": "ewma_beta_is_flat",
    "high_peak2peak": "ewma_beta_high_peak2peak",
    "is_bridged": "ewma_beta_is_bridged",
}

_THRESHOLD_MAP = {
    "line_noise_detected": ("line_noise_detected_enter", "line_noise_detected_exit"),
    "kurtosis": ("kurtosis_enter", "kurtosis_exit"),
    "is_flat": ("is_flat_enter", "is_flat_exit"),
    "high_peak2peak": ("high_peak2peak_enter", "high_peak2peak_exit"),
    "is_bridged": ("is_bridged_enter", "is_bridged_exit"),
}

# Metrics whose "bad" condition is smoothed < enter_th (not >).
# is_flat = low RMS means the channel is dead/flat.
_INVERTED_METRICS = frozenset({"is_flat"})

# Metrics whose raw value is structurally an integer (pair counts, etc.)
# and should not be EWMA-smoothed -- smoothing produces fractional values
# that cannot be interpreted on the metric's native scale. Persistence
# counters (min_steps_to_enter / min_steps_to_clear) handle debouncing.
_NO_EWMA_METRICS = frozenset({"is_bridged"})


@dataclass
class WindowState:
    """Outcome of a single metric evaluation step."""

    metric: str
    state: str
    value: float
    timestamp: float
    message: str | None


class AlertEngine:
    """EWMA smoothing + hysteresis state machine for all EEG metrics."""

    def __init__(self, cfg: RTConfig) -> None:
        self._cfg = cfg
        self._ewmas: dict[str, EWMA] = {
            m: EWMA(beta=getattr(cfg, _BETA_MAP[m])) for m in _EEG_METRICS
        }
        self._states: dict[str, dict] = {}
        for m in _EEG_METRICS:
            self._states[m] = self._fresh_state()

    def update(self, metric: str, raw_value: float, timestamp: float) -> WindowState:
        """Feed a raw metric scalar, apply EWMA + hysteresis, return state."""
        out_name = _OUTPUT_NAMES.get(metric, metric)
        if np.isnan(raw_value):
            s = self._states[metric]
            return WindowState(
                metric=out_name,
                state="bad" if s["in_bad_state"] else "good",
                value=raw_value,
                timestamp=timestamp,
                message=None,
            )
        # Bridge pair count is an integer; smoothing it produces fractional
        # values that aren't interpretable as a pair count. Skip EWMA --
        # persistence counters below provide all the debouncing we need.
        if metric in _NO_EWMA_METRICS:
            smoothed = float(raw_value)
        else:
            smoothed = self._ewmas[metric].update(raw_value)
        s = self._states[metric]
        cfg = self._cfg
        enter_attr, exit_attr = _THRESHOLD_MAP[metric]
        enter_th = getattr(cfg, enter_attr)
        exit_th = getattr(cfg, exit_attr)

        inverted = metric in _INVERTED_METRICS
        is_bad_now = smoothed < enter_th if inverted else smoothed > enter_th
        is_good_now = smoothed >= exit_th if inverted else smoothed < exit_th

        prev_bad = s["in_bad_state"]
        message: str | None = None

        if not prev_bad:
            if is_bad_now:
                s["consecutive_bad"] += 1
                s["consecutive_good"] = 0
            else:
                s["consecutive_bad"] = 0

            if s["consecutive_bad"] >= cfg.min_steps_to_enter:
                s["in_bad_state"] = True
                s["consecutive_good"] = 0
                s["last_alert_time"] = timestamp
                message = self._enter_message(metric, smoothed)
        else:
            if is_good_now:
                s["consecutive_good"] += 1
                s["consecutive_bad"] = 0
            else:
                s["consecutive_good"] = 0

            if s["consecutive_good"] >= cfg.min_steps_to_clear:
                s["in_bad_state"] = False
                s["consecutive_bad"] = 0
                message = self._exit_message(metric, smoothed)
            elif not is_good_now:
                elapsed = timestamp - s["last_alert_time"]
                if elapsed >= cfg.alert_cooldown_sec:
                    s["last_alert_time"] = timestamp

        state_str = "bad" if s["in_bad_state"] else "good"
        return WindowState(
            metric=out_name,
            state=state_str,
            value=smoothed,
            timestamp=timestamp,
            message=message,
        )

    def check_impedance(
        self, impedance_vals: NDArray[np.float64], timestamp: float
    ) -> WindowState:
        """Direct threshold check on median impedance — no EWMA or hysteresis."""
        median = float(np.median(impedance_vals))
        th = self._cfg.impedance_bad_threshold_kohm
        if median > th:
            return WindowState(
                metric="impedance",
                state="bad",
                value=median,
                timestamp=timestamp,
                message=f"Median impedance {median:.1f} kOhm exceeds threshold.",
            )
        return WindowState(
            metric="impedance",
            state="good",
            value=median,
            timestamp=timestamp,
            message=None,
        )

    def reset(self) -> None:
        """Reset all EWMA instances and state machines."""
        for m in _EEG_METRICS:
            self._ewmas[m].reset()
            self._states[m] = self._fresh_state()

    @staticmethod
    def _fresh_state() -> dict:
        return {
            "in_bad_state": False,
            "consecutive_bad": 0,
            "consecutive_good": 0,
            "last_alert_time": -1e9,
        }

    @staticmethod
    def _enter_message(metric: str, value: float) -> str:
        return f"{_OUTPUT_NAMES[metric]} entered BAD state (smoothed={value:.4g})."

    @staticmethod
    def _exit_message(metric: str, value: float) -> str:
        return f"{_OUTPUT_NAMES[metric]} returned to GOOD state (smoothed={value:.4g})."


# Internal short keys -> user-facing names emitted on WindowState.metric.
_OUTPUT_NAMES = {
    "line_noise_detected": "excess line noise",
    "kurtosis":            "excess kurtosis",
    "is_flat":             "low RMS",
    "high_peak2peak":      "excess p2p",
    "is_bridged":          "bridge detected",
}
