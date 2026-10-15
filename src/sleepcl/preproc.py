from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


@dataclass
class FilterSpec:
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0
    band_hz: Tuple[float, float] = (0.3, 35.0)
    order: int = 4


class CausalIIR:
    """
    Stateful causal IIR filtering for streaming.

    - Optional notch (IIR notch at notch_hz)
    - Band-pass (Butterworth band-pass)
    Uses SOS form and preserves zi states across chunks.
    """

    def __init__(self, sfreq: float, spec: FilterSpec):
        self.sfreq = float(sfreq)
        self.spec = spec

        self._sos_notch = None
        self._zi_notch = None

        self._sos_band = None
        self._zi_band = None

        self._is_init = False
        self._build_filters()

    def _build_filters(self) -> None:
        nyq = 0.5 * self.sfreq

        # Notch (optional)
        if self.spec.notch_hz is not None:
            w0 = float(self.spec.notch_hz) / nyq
            if 0.0 < w0 < 1.0:
                b, a = iirnotch(w0=w0, Q=self.spec.notch_q)
                self._sos_notch = tf2sos(b, a)
                self._zi_notch = sosfilt_zi(self._sos_notch)
            else:
                self._sos_notch = None
                self._zi_notch = None

        # Band-pass
        lo, hi = self.spec.band_hz
        lo = max(1e-6, float(lo))
        hi = float(hi)

        if not (0 < lo < hi < nyq):
            raise ValueError(
                f"Invalid band_hz={self.spec.band_hz} for sfreq={self.sfreq}. "
                f"Need 0 < lo < hi < {nyq:.3f}."
            )

        self._sos_band = butter(
            N=self.spec.order,
            Wn=(lo, hi),
            btype="bandpass",
            fs=self.sfreq,
            output="sos",
        )
        self._zi_band = sosfilt_zi(self._sos_band)

    def reset(self) -> None:
        if self._sos_notch is not None:
            self._zi_notch = sosfilt_zi(self._sos_notch)
        self._zi_band = sosfilt_zi(self._sos_band)
        self._is_init = False

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        if x.size == 0:
            return x

        # Initialise filter states to first sample to reduce start transient
        if not self._is_init:
            x0 = float(x[0])
            if self._zi_notch is not None:
                self._zi_notch = self._zi_notch * x0
            if self._zi_band is not None:
                self._zi_band = self._zi_band * x0
            self._is_init = True

        # Notch
        if self._sos_notch is not None:
            y, self._zi_notch = sosfilt(self._sos_notch, x, zi=self._zi_notch)
        else:
            y = x

        # Band-pass
        y, self._zi_band = sosfilt(self._sos_band, y, zi=self._zi_band)
        return y
