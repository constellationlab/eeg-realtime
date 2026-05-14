import numpy as np
from scipy.signal import butter, sosfiltfilt


def filter_eeg_bandpass(x, sampling_rate, low_freq, high_freq, order=2):
    """Zero-phase Butterworth filter.

    x: array of shape (T, C). Returns filtered array of same shape.
    """
    sos = butter(order, [low_freq, high_freq], btype="bandpass",
                 fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, np.asarray(x), axis=0)


def truncate_edges(x_filt, sampling_rate, low_freq, num_cycles=3):
    """Drop num_cycles cycles of low_freq from each edge (filtfilt transient)."""
    num_samples = int(np.ceil(num_cycles * sampling_rate / low_freq))
    if 2 * num_samples >= x_filt.shape[0]:
        raise ValueError(
            f"Cannot remove {num_samples} samples from each edge; "
            f"signal only has {x_filt.shape[0]} samples."
        )
    return x_filt[num_samples:-num_samples, :]


# compute RMS
def compute_rms(x):
    """Root-mean-square per channel. x: (T, C) -> (C,)."""
    return np.sqrt(np.mean(x ** 2, axis=0))


# compute electrical distance (variance of pairwise diff)
def electrical_distance(x):
    """Electrical-distance matrix, ed[i, j] = var(x[:, i] - x[:, j]).

    x: (T, C). Returns (C, C) with NaN on the diagonal.
    Low values indicate possible electrode bridging.

    Closed-form: var(a - b) = var(a) + var(b) - 2*cov(a, b).
    Uses O(C^2) memory instead of materializing a (T, C, C) tensor.
    """
    x = np.asarray(x)
    v = np.var(x, axis=0)                    # ddof=0
    c = np.cov(x, rowvar=False, bias=True)   # bias=True -> ddof=0
    ed = v[:, None] + v[None, :] - 2 * c
    np.fill_diagonal(ed, np.nan)
    return ed


# compute max peak-to-peak amplitude per channel within a window
def max_peak_to_peak(x):
    """Peak-to-peak amplitude per channel in the given window.

    x: (T, C) -> (C,). max(x) - min(x) along time.
    """
    return np.ptp(np.asarray(x), axis=0)


# detect electrode bridges
def is_bridged(x, threshold, ed=None):
    """Symmetric (C, C) boolean matrix of bridged channel pairs.

    mat[i, j] is True iff the electrical distance between channels i and j
    is below threshold. Diagonal is False (NaN < threshold -> False).

    threshold has units of the signal squared. If x is in microvolts,
    a threshold of 5 means 5 uV^2.

    If ed (a precomputed electrical_distance(x) matrix) is passed in, it is
    reused instead of recomputed -- useful when the caller already has it.
    """
    if ed is None:
        ed = electrical_distance(x)
    with np.errstate(invalid="ignore"):  # NaN diagonal triggers a compare warning
        mat = ed < threshold
    return mat


# detect flat electrodes
def is_flat(x, threshold):
    """Flag channels whose RMS is below threshold. x: (T, C) -> (C,) bool."""
    return compute_rms(x) < threshold


# detect high amplitude channels
def is_high_ptp(x, threshold):
    """Flag channels whose peak-to-peak amplitude exceeds threshold."""
    return max_peak_to_peak(x) > threshold
