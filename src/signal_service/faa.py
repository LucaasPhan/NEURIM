"""Frontal alpha asymmetry: the entire "reward" signal.

FAA = ln(alpha_power(F4)) - ln(alpha_power(F3)), z-scored against a per-subject
resting baseline and clipped to [-1, 1]. Higher = more left-frontal activation
= approach motivation, per the standard EEG asymmetry literature. This module
only knows about numbers in, numbers out - it never sees the image generator,
the optimizer, or anything about images.

F3 = Channel_3.csv and F4 = Channel_12.csv.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


EPOC_X_POSITIONS: dict[str, tuple[float, float, float]] = {
    "AF3": (-0.42, 0.88, 0.22),
    "F7": (-0.86, 0.58, 0.04),
    "F3": (-0.46, 0.55, 0.36),
    "FC5": (-0.72, 0.22, 0.22),
    "T7": (-0.95, -0.08, 0.0),
    "P7": (-0.78, -0.58, 0.1),
    "O1": (-0.34, -0.9, 0.18),
    "O2": (0.34, -0.9, 0.18),
    "P8": (0.78, -0.58, 0.1),
    "T8": (0.95, -0.08, 0.0),
    "FC6": (0.72, 0.22, 0.22),
    "F4": (0.46, 0.55, 0.36),
    "F8": (0.86, 0.58, 0.04),
    "AF4": (0.42, 0.88, 0.22),
}


def band_power(samples: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Welch PSD power in `band` (Hz) for a single-channel 1D signal."""
    if samples.size < 8:
        return 0.0
    nperseg = min(samples.size, max(int(fs * 1.0), 8))
    freqs, psd = welch(samples, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    band_freqs, band_psd = freqs[mask], psd[mask]
    if band_freqs.size < 2:
        return float(band_psd.sum())
    return float(np.sum((band_psd[1:] + band_psd[:-1]) * np.diff(band_freqs)) / 2.0)


def raw_faa(
    window: dict[str, np.ndarray],
    fs: float,
    channel_left: str = "F3",
    channel_right: str = "F4",
    band: tuple[float, float] = (8.0, 13.0),
    eps: float = 1e-12,
) -> float:
    """ln(power_right) - ln(power_left) over one window of samples per channel."""
    p_left = band_power(window[channel_left], fs, band) + eps
    p_right = band_power(window[channel_right], fs, band) + eps
    return float(np.log(p_right) - np.log(p_left))


@dataclass
class RunningBaseline:
    """Mean/std of raw FAA collected during the rest period, for z-scoring."""

    mean: float = 0.0
    std: float = 1.0
    n: int = 0

    def fit(self, samples: list[float]) -> None:
        arr = np.asarray(samples, dtype=float)
        self.mean = float(arr.mean())
        self.std = float(arr.std()) or 1.0
        self.n = len(samples)

    def z_score(self, value: float) -> float:
        return (value - self.mean) / self.std


class FAARewardComputer:
    """Sliding-window FAA -> baseline z-score -> clip to [-1, 1] = r(t).

    Feed it raw multi-channel samples as they arrive; call `update()` on the
    cadence you want reward readings (every ~250ms per the spec) and it slices
    the trailing `window_s` seconds out of its ring buffer.
    """

    def __init__(
        self,
        fs: float,
        channel_left: str = "F3",
        channel_right: str = "F4",
        band: tuple[float, float] = (8.0, 13.0),
        window_s: float = 2.0,
        clip: tuple[float, float] = (-1.0, 1.0),
        channels: list[str] | None = None,
    ):
        self.fs = fs
        self.channel_left = channel_left
        self.channel_right = channel_right
        self.band = band
        self.window_s = window_s
        self.clip = clip
        self._maxlen = int(fs * window_s) + 1
        channel_names = list(dict.fromkeys([*(channels or []), channel_left, channel_right]))
        self._buffers: dict[str, deque[float]] = {
            ch: deque(maxlen=self._maxlen) for ch in channel_names
        }
        self.baseline = RunningBaseline()

    def push_sample(self, channel_values: dict[str, float]) -> None:
        for ch, value in channel_values.items():
            if ch not in self._buffers:
                self._buffers[ch] = deque(maxlen=self._maxlen)
            if isinstance(value, int | float):
                self._buffers[ch].append(float(value))

    def ready(self) -> bool:
        return len(self._buffers[self.channel_left]) >= self._maxlen - 1

    def raw_value(self) -> float | None:
        if not self.ready():
            return None
        window = {ch: np.asarray(buf) for ch, buf in self._buffers.items()}
        return raw_faa(window, self.fs, self.channel_left, self.channel_right, self.band)

    def band_powers(self) -> tuple[float, float] | None:
        """(power_left, power_right) in the configured band over the current
        window, or None if the buffer isn't full yet. Diagnostic helper - the
        FAA index is ln(power_right) - ln(power_left)."""
        if not self.ready():
            return None
        p_left = band_power(np.asarray(self._buffers[self.channel_left]), self.fs, self.band)
        p_right = band_power(np.asarray(self._buffers[self.channel_right]), self.fs, self.band)
        return p_left, p_right

    def eeg_features(self, reward: float | None = None, raw: float | None = None) -> dict | None:
        """Compact EEG visualization payload for the frontend.

        Uses alpha-band power over the same sliding window as FAA. This is
        intentionally low-rate derived telemetry, not raw high-frequency EEG.
        """
        if not self.ready():
            return None

        channels = []
        for name, buf in self._buffers.items():
            if not buf:
                continue
            arr = np.asarray(buf, dtype=float)
            alpha = band_power(arr, self.fs, self.band) if arr.size >= 8 else 0.0
            channels.append(
                {
                    "name": name,
                    "value": float(arr[-1]),
                    "alpha_power": float(alpha),
                    "quality": min(1.0, len(buf) / max(1, self._maxlen - 1)),
                    "position": list(EPOC_X_POSITIONS.get(name, (0.0, 0.0, 0.0))),
                }
            )

        return {
            "channels": channels,
            "faa": {
                "raw": raw if raw is not None else self.raw_value(),
                "reward": reward if reward is not None else self.reward(),
                "left_channel": self.channel_left,
                "right_channel": self.channel_right,
            },
        }

    def reward(self) -> float | None:
        """Baseline-normalized r(t), or None if the buffer isn't full yet."""
        value = self.raw_value()
        if value is None:
            return None
        z = self.baseline.z_score(value)
        lo, hi = self.clip
        return float(np.clip(z, lo, hi))
