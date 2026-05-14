# eeg-realtime

Lightweight real-time EEG QC: bandpass filter and per-channel checks for flat,
bridged, and high-amplitude electrodes.

## Install

```
pip install -r requirements.txt
```

## Run the demo

```
python example.py
```

The pipeline runs on a `T x C` matrix (samples by channels):

1. `filter_eeg_bandpass` — zero-phase Butterworth bandpass (`config.yaml: bandpass`).
2. `truncate_edges` — drop the filtfilt transient.
3. `compute_rms`, `electrical_distance`, `max_peak_to_peak` — per-channel metrics.
4. `is_flat`, `is_bridged`, `is_high_ptp` — threshold each metric against `config.yaml: qc`. `is_bridged` returns a `(C, C)` symmetric boolean matrix.

Tune thresholds in `config.yaml`. See `example.py` for the wiring.
