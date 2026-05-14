# neurosentinel `_realtime` — Real-Time EEG Signal Quality Monitor

A lightweight, stateful Python module for monitoring EEG signal quality in real time. Feed it EEG chunks as they arrive from your device and it continuously tracks five signal quality metrics, raising alerts when quality degrades.

---

## Installation

```bash
pip install numpy scipy fooof pycatch22 matplotlib
```

The module is a local package, place the `_realtime/` folder inside your `neurosentinel` package and ensure `neurosentinel` is on your Python path.

---

## Quickstart

```python
from neurosentinel._realtime import RealtimeMonitor, RTConfig

# 1. Configure for your hardware
cfg = RTConfig(
    sfreq=1000.0,      # your EEG sampling rate in Hz
    n_channels=64,     # number of EEG channels
)

# 2. Initialise the real-time monitor
monitor = RealtimeMonitor(cfg)

# In your data acquisition loop, feed chunks as they arrive
while recording:
    chunk = get_next_eeg_chunk()    # shape: (n_channels, n_new_samples)
    timestamp = get_current_time()  # seconds since recording start

    states = monitor.update(chunk, timestamp)

    for state in states:
        if state.message:           # only set on state transitions
            print(f"[{state.metric}] {state.message}")

```

---

### Chunk size

Current patch works best when the chunk size is 1–2 seconds of EEG data. You can change `step_sec`, to control how often the monitor computes each metrics `step_sec=1.0` means the monitor fires one metric update per second, irrespective of the chunk size.

---

## What Gets Monitored

| Metric | What it detects | Conditions for state change |
|--------|----------------|----------|
| `line_noise` | 60 Hz electrical interference | State changes when FOOOF peak at 60 Hz > 4 dB above aperiodic background activity |
| `kurtosis` | Spike artifacts, eye blinks, electrode pops | State changes when Mean absolute excess kurtosis > 5 |
| `rms` | Flat / dead channels | State changes when Mean RMS drops below threshold (signal too weak) |
| `max_amp` | Movement artifact, muscle noise | State changes when Mean peak-to-peak amplitude > 200 µV |
| `catch22` | General anomaly detection algorithm based on catch22 features | State changes when cosine dissimilarity between consecutive windows > 0.15 |

Each metric runs on a **sliding window** (default 4 seconds), evaluated every **step** (default 1 second). Results are **EWMA-smoothed** to reduce noise and **hysteresis thresholds** prevent rapid flickering between states. You can modify these parameters in `config.py`

---

## The `WindowState` Object

Every call to `monitor.update()` returns a list of `WindowState` objects — one per metric that was evaluated. Each has:

| Field | Type | Description |
|-------|------|-------------|
| `metric` | str | `"line_noise"`, `"kurtosis"`, `"rms"`, `"max_amp"`, or `"catch22"` |
| `state` | str | `"good"` or `"bad"` |
| `value` | float | Current EWMA-smoothed value of the metric |
| `timestamp` | float | Timestamp passed to `update()` |
| `message` | str or None | Human-readable alert. **Only set on state transitions**, `None` otherwise |

**The key pattern** — check `state.message is not None` to detect events:

```python
for state in states:
    # Always available — use for dashboards / logging
    print(f"{state.metric}: {state.value:.3f} ({state.state})")

    # Only fires on transitions — use for alerts
    if state.message:
        send_alert(state.message)
```

---

## Configuration

All settings live in `RTConfig`. The most important ones:

```python
cfg = RTConfig(
    # Hardware
    sfreq=1000.0,                    # sampling rate Hz
    n_channels=128,                  # number of channels

    # How much data to analyse and how often (per metric)
    line_noise_window_sec=5.0,       # seconds of data per FOOOF fit
    line_noise_step_sec=1.0,         # re-evaluate every N seconds

    # EWMA smoothing — higher beta = slower response, smoother trace
    # Effective time constant: tau = step_sec / (1 - beta)
    ewma_beta_line_noise=0.85,       # tau ≈ 6.7s at step=1s

    # Hysteresis thresholds
    line_noise_enter=4.0,            # enter bad state above this
    line_noise_exit=3.5,             # recover below this

    # Persistence — require N consecutive bad steps before alerting
    min_steps_to_enter=3,
    min_steps_to_clear=3,

    # Cooldown — minimum seconds between repeated alerts
    alert_cooldown_sec=30.0,

    # Impedance
    impedance_bad_threshold_kohm=30.0,
)
```

### Global defaults shorthand

`config.py` exposes `global_window_sec`, `global_step_sec`, and `global_beta` which all per-metric settings default to. Override them to change everything at once:

```python
cfg = RTConfig(
    sfreq=500.0,
    n_channels=64,
    global_window_sec=6.0,   # all metrics use 6s windows
    global_step_sec=0.5,     # all metrics fire every 0.5s
    global_beta=0.9,         # all EWMA use beta=0.9
)
```

### EWMA and step_sec are coupled

Because EWMA counts steps (not time), the effective smoothing time constant depends on both `beta` and `step_sec`. If you change `step_sec`, adjust `beta` accordingly to maintain the same response speed:

```
beta = 1 - (step_sec / tau_desired_seconds)
# e.g. 10s time constant at step=1s → beta = 0.90
# e.g. 10s time constant at step=0.5s → beta = 0.95
```

---

## Resetting Between Recordings

Call `monitor.reset()` between subjects or recording blocks to clear all internal state — the ring buffer, EWMA values, hysteresis state machines, and the catch22 tracker:

```python
monitor.reset()
```

---

## Running the Simulation

You can test changes to the real-time monitor using the module that simualtes 1/f pink noise signals mixed with various artefact sources:

```bash
python -m neurosentinel._realtime.example
```

This runs a 180-second simulation with synthetic EEG containing four injected artifact periods (line noise, flat channels, movement, spikes) and saves a 7-panel figure to `example_simulation.png`.

---

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | All tunable parameters in one flat dataclass |
| `engine.py` | `RealtimeMonitor` — the main class you interact with |
| `alerts.py` | EWMA smoothing + hysteresis state machine |
| `ewma.py` | Stateful online exponentially weighted moving average |
| `metrics.py` | Stateless metric functions (one per quality metric) |
| `simul.py` | Synthetic EEG signal generation for testing |
| `example.py` | End-to-end simulation with plots |
| `test_rt.py` | Unit tests |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Core array operations |
| `scipy` | Welch PSD, kurtosis |
| `fooof` | Spectral parameterisation for line noise detection |
| `pycatch22` | Time series feature extraction (catch22 metric) |
| `matplotlib` | Plotting in `example.py` only |
