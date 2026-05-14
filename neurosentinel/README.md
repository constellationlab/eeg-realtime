# neurosentinel — Real-Time EEG Signal Quality Monitor

A small, stateful Python package for monitoring EEG signal quality in real time. Feed it EEG samples (or chunks) as they arrive from the amplifier; it tracks five signal-quality metrics on a sliding window and raises typed state transitions when quality crosses a threshold.

---

## Installation

The package lives at `eeg-realtime/neurosentinel/`. Editable install from the project root:

```bash
cd /path/to/eeg-realtime
pip install -e .
```

This installs `numpy`, `scipy`, `fooof`, and `matplotlib` (matplotlib only used by the example scripts).

If a different `neurosentinel` is already installed in your environment, uninstall it first or it will shadow this one:

```bash
pip uninstall neurosentinel
python -c "import neurosentinel; print(neurosentinel.__file__)"
# Confirm the path is .../eeg-realtime/neurosentinel/__init__.py
```

---

## Quickstart

```python
from neurosentinel import RealtimeMonitor, RTConfig

cfg = RTConfig(
    sfreq=1000.0,    # EEG sampling rate in Hz
    n_channels=64,
)
monitor = RealtimeMonitor(cfg)

while recording:
    chunk = get_next_eeg_chunk()      # shape: (n_channels, n_new_samples)
    timestamp = get_current_time()    # seconds since recording start

    states = monitor.update(chunk, timestamp)

    for ws in states:
        if ws.message is not None:    # only set on state transitions
            print(f"[{ws.metric}] {ws.message}")
```

Chunks can be any width, including a single sample (`n_new_samples=1`). See `example_single_sample.py` for a per-sample loop.

---

## What gets monitored

| Emitted `metric` name | Detects | Raw quantity | Default threshold |
|---|---|---|---|
| `excess line noise` | 60 Hz electrical interference | FOOOF peak height (dB above aperiodic) in 59–61 Hz | enter > 4.0, exit < 3.5 |
| `excess kurtosis` | spike artifacts, blinks, electrode pops | mean abs(excess kurtosis) across channels | enter > 5.0, exit < 4.0 |
| `low RMS` | dead / disconnected channels | mean RMS across channels (µV); inverted polarity | enter < 1e-3, exit ≥ 1e-3 |
| `excess p2p` | movement, muscle bursts | mean peak-to-peak across channels (µV) | enter > 200, exit < 150 |
| `bridge detected` | bridged electrodes (shared conduction path) | integer count of pairs with z-scored ED below `bridge_ed_threshold=0.1` (i.e. corr ≥ ~0.95) | enter > 0.5, exit < 0.5 (i.e. ≥1 pair vs. 0 pairs) |

All five metrics flow through the same pipeline (described next). Two specifics worth knowing:

- **`low RMS` uses inverted threshold logic** — it goes BAD when the smoothed value falls *below* `enter`, GOOD when it rises back *above* `exit`. Sign-flipped via `_INVERTED_METRICS` in `alerts.py`.
- **`bridge detected` bypasses EWMA smoothing.** The raw value is an integer pair count (0, 1, 2, …) and EWMA-smoothing it produces fractional values that aren't interpretable as a count. Persistence (`min_steps_to_enter` / `min_steps_to_clear`) does all the debouncing. Listed in `_NO_EWMA_METRICS` in `alerts.py`.

---

## How a metric becomes a state

Every metric goes through the same four stages:

1. **Buffer.** `monitor.update(chunk, t)` appends new samples to a ring buffer big enough to hold `max(window_sec)` of EEG.
2. **Window + step.** Each metric has a `_window_sec` (how much data the metric function sees) and a `_step_sec` (how often it's re-evaluated). When `step_sec` worth of new samples have arrived, the engine grabs the most recent `window_sec` and runs the metric function. Defaults: 4 s window, 0.1 s step.
3. **EWMA smoothing.** The raw scalar is fed through `smoothed = beta * smoothed_prev + (1 - beta) * raw`. Larger `beta` = slower response. Time constant: `tau ≈ step_sec / (1 - beta)` seconds. The bridge metric skips this stage.
4. **Hysteresis + persistence.** The smoothed value is compared against `enter` / `exit` thresholds, then a state transition only fires after `min_steps_to_enter` (or `_clear`) consecutive steps on the new side.

So the full chain for one metric is:

```
new samples → ring buffer → window slice → metric function
            → raw scalar → EWMA → smoothed scalar
            → enter/exit threshold → persistence counter → BAD or GOOD
```

### EWMA depends on update rate, not sampling rate

EWMA itself doesn't know about `sfreq`. It cares about how often it's *updated*, which is `step_sec`. If you change `step_sec`, the time constant changes proportionally. To keep response speed constant when changing `step_sec`, adjust `beta`:

```
beta = 1 - step_sec / tau_desired_seconds
# tau = 2 s, step_sec = 0.1 → beta = 0.95
# tau = 2 s, step_sec = 1.0 → beta = 0.5
```

There's a hidden coupling worth knowing: the engine fires each metric at most once per `update()` call. If you pass chunks much larger than `step_sec`, the effective update rate becomes the chunk rate, not `step_sec`. For honest step timing, call `update()` with chunks no larger than `step_sec` — single samples work great.

---

## The `WindowState` object

Every `monitor.update()` call returns a `list[WindowState]`. Between step boundaries the engine re-emits the previously-known state with `message=None`, so a dashboard view always has fresh data.

| Field | Type | Description |
|---|---|---|
| `metric` | str | One of the five emitted names above |
| `state` | str | `"good"` or `"bad"` |
| `value` | float | Smoothed metric value (raw integer count for `bridge detected`) |
| `timestamp` | float | The timestamp passed to `update()` |
| `message` | str or None | Human-readable alert. **Only set on state transitions**, `None` otherwise |

Typical pattern — separate "current state for dashboard" from "alert on transition":

```python
last_state: dict[str, str] = {}
for ws in states:
    last_state[ws.metric] = ws.state          # always-current view
    if ws.message is not None:
        send_alert(ws.message)                # transition-only alerts
```

---

## Configuration

All knobs live in `RTConfig` (`config.py`). Per-metric attributes follow the pattern `<short_key>_<param>`, where the short keys are:

| short key | emitted name |
|---|---|
| `line_noise_detected` | `excess line noise` |
| `kurtosis` | `excess kurtosis` |
| `is_flat` | `low RMS` |
| `high_peak2peak` | `excess p2p` |
| `is_bridged` | `bridge detected` |

For each short key there is `<key>_window_sec`, `<key>_step_sec`, `ewma_beta_<key>` (unused for `is_bridged`), `<key>_enter`, and `<key>_exit`. Three persistence/cooldown fields apply to all metrics:

```python
min_steps_to_enter: int = 2     # consecutive bad steps before BAD fires
min_steps_to_clear: int = 3     # consecutive good steps before GOOD fires
alert_cooldown_sec: float = 30  # min seconds between repeat alerts
```

### Global defaults

`global_window_sec`, `global_step_sec`, and `global_beta` set the per-metric defaults. Override them to change everything at once:

```python
cfg = RTConfig(
    sfreq=500.0,
    n_channels=64,
    global_window_sec=6.0,
    global_step_sec=0.5,
    global_beta=0.9,
)
```

### Bridge-specific config

```python
bridge_ed_threshold: float = 0.1   # correlation distance ed = 2*(1 - corr); 0.1 ≈ corr ≥ 0.95
is_bridged_enter: float = 0.5      # raw count > 0.5 → ≥ 1 pair
is_bridged_exit:  float = 0.5      # raw count < 0.5 → 0 pairs
```

The bridge metric z-scores each channel along time before computing the electrical distance, then excludes channels with std below an internal floor (`_BRIDGE_STD_FLOOR_UV = 0.1` in `metrics.py`) so simultaneously-flat channels can't be misclassified as bridged.

---

## Resetting between recordings

```python
monitor.reset()
```

Clears the ring buffer, EWMA state, hysteresis counters, and step counters. Use between subjects or recording blocks.

---

## Impedance check

Impedance is a one-shot direct comparison, not part of the streaming loop:

```python
ws = monitor.update_impedance(impedance_kohm_array, timestamp)
# ws.state is "bad" if median impedance > cfg.impedance_bad_threshold_kohm (default 30 kOhm)
```

No EWMA, no hysteresis, no persistence — just `median > threshold`.

---

## Running the simulation

The chunked end-to-end demo (180 s of synthetic EEG with 5 injected artifacts):

```bash
python -m neurosentinel.example
```

Saves a 7-panel figure to `example_simulation.png` showing the raw signal plus the smoothed/raw trace for each metric with the artifact ground-truth shaded behind it.

The single-sample-at-a-time variant:

```bash
python -m neurosentinel.example_single_sample
```

Useful for understanding the `step_sec` / EWMA timing — alerts arrive within a few hundred ms of the artifact starting, instead of lagging by one chunk.

---

## File structure

| File | Purpose |
|---|---|
| `__init__.py` | Public surface: `RealtimeMonitor`, `RTConfig`, `WindowState` |
| `config.py` | `RTConfig` dataclass — every tunable parameter in one place |
| `engine.py` | `RealtimeMonitor` — ring buffer + per-metric step timing + dispatch |
| `metrics.py` | Stateless metric functions (one per quality metric) |
| `alerts.py` | `AlertEngine` — EWMA smoothing + hysteresis + persistence + cooldown |
| `ewma.py` | `EWMA` — stateful exponentially-weighted moving average |
| `simul.py` | Synthetic 1/f-pink-noise EEG with five injected artifact types |
| `example.py` | Chunked end-to-end simulation with 7-panel figure |
| `example_single_sample.py` | Per-sample streaming demo |
| `test_rt.py` | **Stale.** References an older API (`neurosentinel.rt.*`) and does not run against this codebase. Slated for rewrite. |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Core array operations |
| `scipy` | Welch PSD, kurtosis |
| `fooof` | Spectral parameterisation for `excess line noise` |
| `matplotlib` | Plotting in the example scripts only |
