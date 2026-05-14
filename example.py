"""
Example: run the streaming-EEG QC pipeline on a synthetic T x C matrix.

Pipeline:
  1) bandpass filter 0.5-50 Hz
  2) remove edge artefacts
  3) compute RMS, electrical distance, peak-to-peak
  4) flag bad channels:
       - flat              (RMS below threshold)
       - high peak-to-peak (motion / large transient artifact)
       - bridged           (low electrical distance to another channel)

Defaults live in config.yaml.
"""
from pathlib import Path

import numpy as np
import yaml

from sigproc import (
    filter_eeg_bandpass,
    truncate_edges,
    compute_rms,
    electrical_distance,
    max_peak_to_peak,
    is_flat,
    is_bridged,
    is_high_ptp,
)


def make_synthetic_eeg(sampling_rate, duration_s=30, n_channels=8, seed=0):
    """T x C synthetic EEG with three planted bad channels."""
    rng = np.random.default_rng(seed)
    n = int(sampling_rate * duration_s)
    t = np.arange(n) / sampling_rate

    # baseline: white noise + 10 Hz "alpha" common to all channels
    x = 20.0 * rng.standard_normal((n, n_channels))
    x += 30.0 * np.sin(2 * np.pi * 10 * t)[:, None]

    # planted faults
    x[:, 1] = 0.1 * rng.standard_normal(n)                  # ch1: flat
    x[:, 3] = x[:, 2] + 0.5 * rng.standard_normal(n)        # ch3: bridged to ch2
    x[:, 5] += 400.0 * np.sin(2 * np.pi * 3 * t)            # ch5: huge artifact
    return x


def main():
    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    sr = cfg["sampling_rate"]
    bp = cfg["bandpass"]
    qc = cfg["qc"]

    # synthetic T x C EEG data
    x = make_synthetic_eeg(sr, duration_s=30, n_channels=128)
    print(f"input: T={x.shape[0]} samples ({x.shape[0]/sr:.1f}s) x C={x.shape[1]} channels")

    # broadband filter 
    x_f = filter_eeg_bandpass(
        x,
        sampling_rate=sr,
        low_freq=bp["low_freq"],
        high_freq=bp["high_freq"],
        order=bp["order"],
    )

    # remove edges
    x_f = truncate_edges(
        x_f,
        sampling_rate=sr,
        low_freq=bp["low_freq"],
        num_cycles=cfg["truncate_edges"]["num_cycles"],
    )

    # QC metrics per channel
    rms = compute_rms(x_f)
    ed  = electrical_distance(x_f)
    ptp = max_peak_to_peak(x_f)

    # flags
    flat_mask    = is_flat(x_f, threshold=qc["flat_rms_threshold"])
    high_ptp     = is_high_ptp(x_f, threshold=qc["ptp_threshold"])
    bridged_mat  = is_bridged(x_f, threshold=qc["bridge_ed_threshold"], ed=ed)
    bridges      = [(int(i), int(j))
                    for i, j in np.argwhere(np.triu(bridged_mat, k=1))]

    # report
    print(f"{'ch':>3} {'rms (uV)':>10} {'ptp (uV)':>10}  flags")
    for c in range(x_f.shape[1]):
        tags = []
        if flat_mask[c]: tags.append("FLAT")
        if high_ptp[c]:  tags.append("HIGH_PTP")
        print(f"{c:>3} {rms[c]:>10.2f} {ptp[c]:>10.2f}  {','.join(tags)}")

    print("\nbridged channel pairs (low electrical distance):")
    if bridges:
        for i, j in bridges:
            print(f"  ch{i} <-> ch{j}  ed={ed[i, j]:.3f} uV^2")
    else:
        print("  none")


if __name__ == "__main__":
    main()
