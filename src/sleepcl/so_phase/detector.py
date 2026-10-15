from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from scipy.signal import hilbert


@dataclass
class SOPhaseSpec:
    sfreq: float
    window_sec: float = 12.0          # trailing window used for phase estimate
    target_phase_rad: float = 0.0     # adjust after preview (0 often ~positive peak)
    tolerance_rad: float = 0.35       # how close to target counts as "in window"
    min_amp_uv: float = 60.0          # amplitude gate (robust peak-to-peak in window)
    robust_amp: bool = True           # use percentiles instead of strict ptp
    edge_detect: bool = True          # only fire when entering the phase window


def _wrap_to_pi(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.angle(np.exp(1j * x)))


class SOPhaseDetector:
    """
    Trailing-window phase estimator for slow oscillations.

    Assumes input is already SO-band filtered (e.g., 0.3–1.2 Hz), in VOLTS.
    Phase is computed using Hilbert transform on the trailing window, and we use
    the phase of the *latest* sample only (pseudo-online: no future peeking).
    """

    def __init__(self, spec: SOPhaseSpec):
        self.spec = spec
        self.sfreq = float(spec.sfreq)
        self.win_n = int(round(self.spec.window_sec * self.sfreq))
        self.win_n = max(10, self.win_n)

        self._buf = np.zeros(self.win_n, dtype=float)
        self._filled = 0
        self._idx = 0

        self._prev_in_window = False

    def _append(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).ravel()
        for v in x:
            self._buf[self._idx] = v
            self._idx = (self._idx + 1) % self.win_n
            self._filled = min(self.win_n, self._filled + 1)

    def _get_window(self) -> np.ndarray:
        if self._filled < self.win_n:
            return self._buf[:self._filled].copy()
        # return in chronological order
        return np.concatenate([self._buf[self._idx:], self._buf[:self._idx]]).copy()

    def update(self, x_so_volts: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Update detector with a new chunk of SO-band filtered samples (VOLTS).
        Returns a dict for the *latest* sample in the window once enough history exists.
        """
        self._append(x_so_volts)

        if self._filled < self.win_n:
            return None

        w = self._get_window()

        # Robust amplitude estimate in µV
        w_uv = w * 1e6
        if self.spec.robust_amp:
            amp_uv = float(np.percentile(w_uv, 95) - np.percentile(w_uv, 5))
        else:
            amp_uv = float(np.ptp(w_uv))

        # Hilbert phase (use only last sample)
        analytic = hilbert(w)
        phase = float(np.angle(analytic[-1]))

        # Distance to target on circle
        diff = _wrap_to_pi(phase - self.spec.target_phase_rad)
        in_window = (abs(diff) <= self.spec.tolerance_rad) and (amp_uv >= self.spec.min_amp_uv)

        if self.spec.edge_detect:
            candidate = bool(in_window and (not self._prev_in_window))
        else:
            candidate = bool(in_window)

        self._prev_in_window = bool(in_window)

        return {
            "phase": phase,
            "diff_to_target": diff,
            "amp_uv": amp_uv,
            "in_window": bool(in_window),
            "candidate": bool(candidate),
        }
