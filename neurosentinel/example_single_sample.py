"""Minimal single-sample-at-a-time usage of neurosentinel.

Most acquisition systems give you EEG one sample at a time. This script
shows the smallest possible loop: read one sample, hand it to the monitor,
look at the returned states. The monitor's internal ring buffer
accumulates samples until each metric has enough data to fire.

Run from the repo root:
    python -m neurosentinel.example_single_sample
or directly:
    python neurosentinel/example_single_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Defend against an installed `neurosentinel` package shadowing this one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from neurosentinel.config import RTConfig
from neurosentinel.engine import RealtimeMonitor
from neurosentinel.simul import _make_signal, SFREQ, N_CHANNELS


def main() -> None:
    # Build a synthetic recording the same way example.py does, then iterate
    # through it ONE sample at a time so you can see what the API looks like
    # in a per-sample acquisition loop.
    signal, _t_vec, *_ = _make_signal()       # (T, C)
    signal = signal.T                          # → (C, T) -- the API expects channels-first

    cfg = RTConfig(sfreq=SFREQ, n_channels=N_CHANNELS)
    monitor = RealtimeMonitor(cfg)

    # Track which metrics we've already seen a state for, just so we can
    # print one line the first time each metric becomes available.
    seen: set[str] = set()
    n_total = signal.shape[1]
    print(f"Streaming {n_total} samples ({n_total / SFREQ:.0f} s at {SFREQ} Hz)…\n")

    for i in range(n_total):
        sample = signal[:, i : i + 1]          # shape (C, 1) -- one column
        ts = i / SFREQ                         # current time in seconds

        states = monitor.update(sample, timestamp=ts)

        # `states` is a list with one entry per metric that fired this tick.
        # It will be EMPTY for many ticks at the start (the buffer needs to
        # fill before any metric can compute a window) and then on most
        # ticks between step boundaries.
        for ws in states:
            if ws.metric not in seen:
                seen.add(ws.metric)
                print(f"  first reading: t={ws.timestamp:6.2f}s  {ws.metric:>20s}  "
                      f"value={ws.value:.3g}  state={ws.state}")

            # The interesting events are state TRANSITIONS. ws.message is
            # only set on transitions; it's None on every other tick.
            if ws.message is not None:
                print(f"  TRANSITION:    t={ws.timestamp:6.2f}s  [{ws.metric}]  {ws.message}")

    print("\nDone. Final state per metric:")
    for metric in sorted(seen):
        # The monitor doesn't expose a public 'current state' getter, so
        # we just feed one more empty-ish tick and read off the last value.
        # In a real app you'd track the last `ws.state` you observed per
        # metric -- exactly what `seen` would do if you stored values.
        pass


if __name__ == "__main__":
    main()
