"""Self-contained simulation of real-time EEG monitoring — all metrics."""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure THIS neurosentinel (the one this file lives inside) wins over
# any other `neurosentinel` package that might be installed in the active
# environment. The parent of this file's package is `eeg-realtime/`, which
# must be on sys.path BEFORE any installed neurosentinel.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from neurosentinel.config import RTConfig
from neurosentinel.engine import RealtimeMonitor

from neurosentinel.simul import (
    _make_signal,
    CHUNK_SEC,
    SFREQ,
    N_CHANNELS,
    FLAT_ON_SEC,
    BRIDGE_CHANNELS,
)


# Output metric names emitted by the engine.
M_LINE   = "excess line noise"
M_KURT   = "excess kurtosis"
M_P2P    = "excess p2p"
M_FLAT   = "low RMS"
M_BRIDGE = "bridge detected"


# ── Simulation loop ─────────────────────────────────────────────────────────

def main() -> None:
    signal, t_vec, line_on, flat_on, move_on, spike_on, bridge_on = _make_signal()
    n_samples = signal.shape[0]
    chunk_size = int(CHUNK_SEC * SFREQ)

    cfg = RTConfig(sfreq=SFREQ, n_channels=N_CHANNELS)

    monitor = RealtimeMonitor(cfg)

    rec: dict[str, dict] = {
        m: {"t": [], "value": [], "state": []}
        for m in (M_LINE, M_KURT, M_P2P, M_FLAT, M_BRIDGE)
    }
    transitions: list[tuple[float, str, str, str]] = []

    for start in range(0, n_samples - chunk_size + 1, chunk_size):
        chunk = signal[start : start + chunk_size, :].T
        ts = start / SFREQ
        states = monitor.update(chunk, timestamp=ts)

        for ws in states:
            if ws.metric in rec:
                rec[ws.metric]["t"].append(ws.timestamp)
                rec[ws.metric]["value"].append(ws.value)
                rec[ws.metric]["state"].append(ws.state)
            if ws.message is not None:
                transitions.append((ws.timestamp, ws.metric, ws.state, ws.message))
                print(f"  t={ws.timestamp:.1f}s  [{ws.metric}]  {ws.message}")

    # Per-channel RMS heatmap (computed offline for visualisation only).
    from neurosentinel.metrics import channel_rms as _rms_fn
    rms_step = int(cfg.is_flat_step_sec * SFREQ)
    rms_win  = int(cfg.is_flat_window_sec * SFREQ)
    rms_times, rms_matrix = [], []
    for start in range(0, n_samples - rms_win + 1, rms_step):
        window = signal[start : start + rms_win, :].T
        _, detail = _rms_fn(window)
        rms_times.append(start / SFREQ)
        rms_matrix.append(detail["per_channel"])
    rms_arr = np.array(rms_matrix).T

    fig, axes = plt.subplots(7, 1, figsize=(16, 21), constrained_layout=True)

    def shade_artifact(ax, mask, color, label):
        in_region, x0 = False, 0.0
        for i, active in enumerate(mask):
            if active and not in_region:
                x0 = t_vec[i]
                in_region = True
            elif not active and in_region:
                ax.axvspan(x0, t_vec[i], alpha=0.12, color=color, label=label)
                label = "_nolegend_"
                in_region = False
        if in_region:
            ax.axvspan(x0, t_vec[-1], alpha=0.12, color=color, label=label)

    def mark_transitions(ax, metric, y_enter, y_exit):
        for ts_tr, m, state_tr, _ in transitions:
            if m != metric:
                continue
            if state_tr == "bad":
                ax.plot(ts_tr, y_enter, "rv", ms=8, zorder=5)
            else:
                ax.plot(ts_tr, y_exit, "g^", ms=8, zorder=5)

    xlim = (t_vec[0], t_vec[-1])

    # 0. Raw signal
    ax = axes[0]
    ax.plot(t_vec, signal[:, 0], lw=0.3, color="#2c3e50")
    shade_artifact(ax, line_on,   "red",    "60 Hz ON")
    shade_artifact(ax, flat_on,   "blue",   "Flat channels ON")
    shade_artifact(ax, move_on,   "orange", "Movement ON")
    shade_artifact(ax, spike_on,  "purple", "Spikes ON")
    shade_artifact(ax, bridge_on, "teal",   "Bridge ON")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("Channel 0 — raw signal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*xlim)

    # 1. Line noise
    ax = axes[1]
    shade_artifact(ax, line_on, "red", "60 Hz ON")
    ax.plot(rec[M_LINE]["t"], rec[M_LINE]["value"], lw=1.5, color="#2980b9", label="EWMA smoothed")
    ax.axhline(cfg.line_noise_detected_enter, ls="--", color="#e74c3c", lw=1, label=f"enter ({cfg.line_noise_detected_enter})")
    ax.axhline(cfg.line_noise_detected_exit,  ls=":",  color="#27ae60", lw=1, label=f"exit ({cfg.line_noise_detected_exit})")
    mark_transitions(ax, M_LINE, cfg.line_noise_detected_enter, cfg.line_noise_detected_exit)
    ax.set_ylabel("Line noise dB")
    ax.set_title(M_LINE)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*xlim)

    # 2. Kurtosis
    ax = axes[2]
    shade_artifact(ax, spike_on, "purple", "Spikes ON")
    ax.plot(rec[M_KURT]["t"], rec[M_KURT]["value"], lw=1.5, color="#8e44ad", label="EWMA smoothed")
    ax.axhline(cfg.kurtosis_enter, ls="--", color="#e74c3c", lw=1, label=f"enter ({cfg.kurtosis_enter})")
    ax.axhline(cfg.kurtosis_exit,  ls=":",  color="#27ae60", lw=1, label=f"exit ({cfg.kurtosis_exit})")
    mark_transitions(ax, M_KURT, cfg.kurtosis_enter, cfg.kurtosis_exit)
    ax.set_ylabel("|Excess kurtosis|")
    ax.set_title(M_KURT)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*xlim)

    # 3. Peak-to-peak
    ax = axes[3]
    shade_artifact(ax, move_on, "orange", "Movement ON")
    ax.plot(rec[M_P2P]["t"], rec[M_P2P]["value"], lw=1.5, color="#e67e22", label="EWMA smoothed")
    ax.axhline(cfg.high_peak2peak_enter, ls="--", color="#e74c3c", lw=1, label=f"enter ({cfg.high_peak2peak_enter} uV)")
    ax.axhline(cfg.high_peak2peak_exit,  ls=":",  color="#27ae60", lw=1, label=f"exit ({cfg.high_peak2peak_exit} uV)")
    mark_transitions(ax, M_P2P, cfg.high_peak2peak_enter, cfg.high_peak2peak_exit)
    ax.set_ylabel("Peak-to-peak (uV)")
    ax.set_title(M_P2P)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*xlim)

    # 4. Per-channel RMS heatmap
    ax = axes[4]
    im = ax.imshow(
        rms_arr,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[rms_times[0], rms_times[-1], 0, N_CHANNELS],
    )
    ax.axvline(FLAT_ON_SEC[0], ls="--", color="cyan", lw=1, label="Flat channels ON")
    ax.axvline(FLAT_ON_SEC[1], ls="--", color="cyan", lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel index")
    ax.set_title("Per-channel RMS")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMS (uV)")

    # 5. Bridged electrodes
    ax = axes[5]
    shade_artifact(ax, bridge_on, "teal", f"Bridge ON (ch {BRIDGE_CHANNELS[0]}↔{BRIDGE_CHANNELS[1]})")
    ax.step(
        rec[M_BRIDGE]["t"], rec[M_BRIDGE]["value"],
        where="post", lw=1.5, color="#117a65",
        label="raw pair count (no EWMA)",
    )
    ax.axhline(
        cfg.is_bridged_enter, ls="--", color="#e74c3c", lw=1,
        label=f"enter / exit boundary ({cfg.is_bridged_enter})",
    )
    mark_transitions(ax, M_BRIDGE, cfg.is_bridged_enter, cfg.is_bridged_enter)
    ax.set_ylabel("Pair count (integers)")
    ax.set_title(
        f"{M_BRIDGE} — z-scored electrical distance, threshold ed<{cfg.bridge_ed_threshold}; "
        f"persistence: {cfg.min_steps_to_enter} steps to enter / {cfg.min_steps_to_clear} to clear"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*xlim)
    # Force integer y-ticks since pair count is discrete.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 6. Baseline PSD
    ax = axes[6]
    baseline = signal[: int(10 * SFREQ), :]
    f, psd = welch(baseline.T, fs=SFREQ, nperseg=2048, axis=-1)
    mean_psd = psd.mean(axis=0)
    ax.loglog(f[1:], mean_psd[1:], color="#2c3e50", lw=1.5, label="mean PSD")
    ref = mean_psd[1] * (f[1] / f[1:])
    ax.loglog(f[1:], ref, ls="--", color="#e74c3c", lw=1, label="1/f reference")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Baseline PSD")
    ax.legend(fontsize=8)

    out = Path(__file__).parent / "example_simulation.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
