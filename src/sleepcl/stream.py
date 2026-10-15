# A streamer that yields fixed-size chunks and a ring buffer that can return the last N seconds on demand (for staging epochs and SO phase windows).

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StreamConfig:
    sfreq: float
    chunk_sec: float = 1.0


class ChunkStreamer:
    """
    Iterates over a 1D signal in fixed-size chunks.
    Yields (t_end_sec, chunk) where t_end_sec is the end time of the chunk.
    """

    def __init__(self, x: np.ndarray, cfg: StreamConfig):
        self.x = np.asarray(x, dtype=float)
        self.sfreq = float(cfg.sfreq)
        self.chunk_n = int(round(float(cfg.chunk_sec) * self.sfreq))
        if self.chunk_n <= 0:
            raise ValueError("chunk_sec too small for this sfreq")

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.x):
            raise StopIteration

        j = min(self._i + self.chunk_n, len(self.x))
        chunk = self.x[self._i:j]
        self._i = j

        t_end_sec = self._i / self.sfreq
        return t_end_sec, chunk


class RingBuffer1D:
    """
    Fixed-size ring buffer for 1D streaming signals.
    Stores the most recent max_sec seconds.
    """

    def __init__(self, sfreq: float, max_sec: float):
        self.sfreq = float(sfreq)
        self.max_n = int(round(float(max_sec) * self.sfreq))
        if self.max_n <= 1:
            raise ValueError("max_sec too small")
        self.buf = np.zeros(self.max_n, dtype=float)
        self.n_filled = 0
        self.write_idx = 0

    def append(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).ravel()
        n = len(x)
        if n == 0:
            return

        if n >= self.max_n:
            # Only keep last max_n samples
            self.buf[:] = x[-self.max_n:]
            self.n_filled = self.max_n
            self.write_idx = 0
            return

        end = self.write_idx + n
        if end <= self.max_n:
            self.buf[self.write_idx:end] = x
        else:
            k = self.max_n - self.write_idx
            self.buf[self.write_idx:] = x[:k]
            self.buf[: n - k] = x[k:]

        self.write_idx = (self.write_idx + n) % self.max_n
        self.n_filled = min(self.max_n, self.n_filled + n)

    def get_last(self, sec: float) -> np.ndarray:
        """Return the last `sec` seconds (or fewer if not enough data yet)."""
        n = int(round(float(sec) * self.sfreq))
        n = min(n, self.n_filled)
        if n <= 0:
            return np.array([], dtype=float)

        start = (self.write_idx - n) % self.max_n
        if start < self.write_idx:
            return self.buf[start:self.write_idx].copy()
        else:
            return np.concatenate([self.buf[start:], self.buf[:self.write_idx]]).copy()

    def seconds_available(self) -> float:
        return self.n_filled / self.sfreq
