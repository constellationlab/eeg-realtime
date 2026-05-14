"""RealtimeMonitor — the single public entry point for callers."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .alerts import AlertEngine, WindowState
from .config import RTConfig

from .metrics import (
    channel_kurtosis,
    channel_max_amplitude,
    channel_rms,
    line_noise_db,
    channel_bridged,
)

_METRIC_DEFS = {
    "line_noise_detected": {
        "window_attr": "line_noise_detected_window_sec",
        "step_attr": "line_noise_detected_step_sec",
        "fn": line_noise_db,
        "needs_cfg": True,
    },
    "kurtosis": {
        "window_attr": "kurtosis_window_sec",
        "step_attr": "kurtosis_step_sec",
        "fn": channel_kurtosis,
        "needs_cfg": False,
    },
    "is_flat": {
        "window_attr": "is_flat_window_sec",
        "step_attr": "is_flat_step_sec",
        "fn": channel_rms,
        "needs_cfg": False,
    },
    "high_peak2peak": {
        "window_attr": "high_peak2peak_window_sec",
        "step_attr": "high_peak2peak_step_sec",
        "fn": channel_max_amplitude,
        "needs_cfg": False,
    },
    "is_bridged": {
        "window_attr": "is_bridged_window_sec",
        "step_attr": "is_bridged_step_sec",
        "fn": channel_bridged,
        "needs_cfg": True,
    },
}


class RealtimeMonitor:
    """Stateful real-time EEG quality monitor."""

    def __init__(self, cfg: RTConfig | None = None) -> None:
        self._cfg = cfg or RTConfig()
        c = self._cfg

        max_window_sec = max(
            c.line_noise_detected_window_sec,
            c.kurtosis_window_sec,
            c.is_flat_window_sec,
            c.high_peak2peak_window_sec,
            c.is_bridged_window_sec,
        )
        self._capacity = math.ceil(max_window_sec * c.sfreq)
        self._buf = np.zeros((c.n_channels, self._capacity), dtype=np.float64)
        self._write_pos = 0
        self._n_filled = 0

        self._alert_engine = AlertEngine(c)

        self._counters: dict[str, int] = {m: 0 for m in _METRIC_DEFS}
        self._ready: dict[str, bool] = {m: False for m in _METRIC_DEFS}
        self._last_state: dict[str, WindowState | None] = {m: None for m in _METRIC_DEFS}
        self._window_samples: dict[str, int] = {}
        self._step_samples: dict[str, int] = {}
        for name, defn in _METRIC_DEFS.items():
            self._window_samples[name] = math.ceil(
                getattr(c, defn["window_attr"]) * c.sfreq
            )
            self._step_samples[name] = math.ceil(
                getattr(c, defn["step_attr"]) * c.sfreq
            )

    def update(
        self, chunk: NDArray[np.float64], timestamp: float
    ) -> list[WindowState]:
        """Ingest an EEG chunk (n_channels, n_new_samples) and return any state changes."""
        n_new = chunk.shape[1]
        self._append(chunk)

        results: list[WindowState] = []
        for name, defn in _METRIC_DEFS.items():
            self._counters[name] += n_new
            ws = self._window_samples[name]
            ss = self._step_samples[name]

            if not self._ready[name]:
                if self._n_filled >= ws:
                    self._ready[name] = True
                else:
                    continue

            if self._counters[name] >= ss:
                window = self._get_latest(ws)
                fn = defn["fn"]
                scalar, detail = fn(window, self._cfg) if defn["needs_cfg"] else fn(window)

                state = self._alert_engine.update(name, scalar, timestamp)
                self._counters[name] = 0
                self._last_state[name] = state
            elif self._last_state[name] is not None:
                # Between steps — re-emit last known state with updated timestamp
                prev = self._last_state[name]
                state = WindowState(
                    metric=prev.metric,
                    state=prev.state,
                    value=prev.value,
                    timestamp=timestamp,
                    message=None,  # no transition
                )
            else:
                continue  # not yet ready and no prior state

            results.append(state)

        return results

    def update_impedance(
        self, impedance_vals: NDArray[np.float64], timestamp: float
    ) -> WindowState:
        """Check impedance against threshold and return state."""
        return self._alert_engine.check_impedance(impedance_vals, timestamp)

    def reset(self) -> None:
        """Reset all internal state."""
        self._buf[:] = 0.0
        self._write_pos = 0
        self._n_filled = 0
        for m in self._counters:
            self._counters[m] = 0
            self._ready[m] = False
            self._last_state[m] = None
        self._alert_engine.reset()

    def _append(self, chunk: NDArray[np.float64]) -> None:
        n_new = chunk.shape[1]
        cap = self._capacity

        if n_new >= cap:
            self._buf[:] = chunk[:, -cap:]
            self._write_pos = 0
            self._n_filled = cap
            return

        space = cap - self._write_pos
        if n_new <= space:
            self._buf[:, self._write_pos : self._write_pos + n_new] = chunk
        else:
            self._buf[:, self._write_pos :] = chunk[:, :space]
            self._buf[:, : n_new - space] = chunk[:, space:]

        self._write_pos = (self._write_pos + n_new) % cap
        self._n_filled = min(self._n_filled + n_new, cap)

    def _get_latest(self, n_samples: int) -> NDArray[np.float64]:
        end = self._write_pos
        start = (end - n_samples) % self._capacity
        if start < end:
            return self._buf[:, start:end].copy()
        return np.concatenate(
            [self._buf[:, start:], self._buf[:, :end]], axis=1
        )
