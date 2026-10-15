from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


@dataclass
class FeatureSpec:
    # Classic sleep staging bands
    bands: List[Tuple[str, float, float]] = (
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 12.0),
        ("sigma", 12.0, 16.0),
        ("beta", 16.0, 30.0),
    )
    fmin: float = 0.5
    fmax: float = 30.0
    welch_sec: float = 4.0   # window length for Welch
    welch_olap: float = 0.5  # overlap fraction
    use_relative: bool = True  # relative bandpower (robust across subjects)


def _bandpower(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def extract_epoch_features(epoch: np.ndarray, sfreq: float, spec: FeatureSpec) -> np.ndarray:
    """
    epoch: 1D array (samples,) for one 30s epoch
    Returns feature vector: log bandpowers (+ optionally relative powers) + simple stats
    """
    x = np.asarray(epoch, dtype=float)

    # Welch PSD
    nperseg = int(round(spec.welch_sec * sfreq))
    nperseg = max(32, min(nperseg, len(x)))
    noverlap = int(round(spec.welch_olap * nperseg))

    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap, scaling="density")
    # Restrict to [fmin, fmax]
    keep = (freqs >= spec.fmin) & (freqs <= spec.fmax)
    freqs = freqs[keep]
    psd = psd[keep]

    total = _bandpower(psd, freqs, spec.fmin, spec.fmax) + 1e-20

    feats = []
    for _, lo, hi in spec.bands:
        bp = _bandpower(psd, freqs, lo, hi) + 1e-20
        if spec.use_relative:
            bp = bp / total
        feats.append(np.log10(bp))

    # A couple of time-domain stats (cheap + sometimes helpful)
    x_uv = x * 1e6
    feats.extend([
        float(np.std(x_uv)),
        float(np.mean(np.abs(np.diff(x_uv)))),  # line length-ish
    ])

    return np.asarray(feats, dtype=float)
