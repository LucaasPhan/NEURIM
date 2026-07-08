"""Fake reward sources with the exact same interface FAA reward has: a scalar
in [-1, 1], read a few times a second. This is what build-order step 1 uses -
prove the optimizer/generator loop converges before EEG ever touches it.

Downstream code (Optimizer, Orchestrator) only ever sees RewardMessage, so
swapping this out for real FAA later is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.common.messages import RewardMessage


class RewardSource:
    """Common interface: FAARewardComputer-backed or fake, doesn't matter."""

    def read_reward(self) -> RewardMessage | None:
        raise NotImplementedError


class KeyboardRewardSource(RewardSource):
    """Up/down arrow keys nudge reward; it decays toward 0 between presses.

    Uses `pynput` for global key capture (works without terminal focus, which
    matters once the frontend/pyramid window has focus instead). Falls back
    to raising ImportError with a clear message if pynput isn't installed or
    the OS denies the accessibility permission it needs on macOS.
    """

    def __init__(self, decay: float = 0.92, step: float = 0.15):
        from pynput import keyboard

        self._value = 0.0
        self._decay = decay
        self._step = step
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key) -> None:
        from pynput.keyboard import Key

        if key == Key.up:
            self._value = float(np.clip(self._value + self._step, -1.0, 1.0))
        elif key == Key.down:
            self._value = float(np.clip(self._value - self._step, -1.0, 1.0))

    def read_reward(self) -> RewardMessage:
        reading = self._value
        self._value *= self._decay
        return RewardMessage(r=reading, source="fake")

    def close(self) -> None:
        self._listener.stop()


@dataclass
class ScriptedRewardSource(RewardSource):
    """Deterministic reward = similarity between a hidden target and whatever
    z `get_current_z` returns, plus noise. Used to prove the search loop
    converges without any human (or EEG) in the loop, e.g. in CI.

    `get_current_z` must return the *candidate currently being evaluated*
    (e.g. `OptimizerService.pending_candidate`), not the last accepted point -
    the latter is frozen for the whole reward window and would carry no
    signal about the thing actually on screen.
    """

    target: np.ndarray
    get_current_z: Callable[[], np.ndarray]
    noise_std: float = 0.05
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def read_reward(self) -> RewardMessage:
        z = np.asarray(self.get_current_z())
        dist = np.linalg.norm(z - self.target[: len(z)])
        # Reward falls off with distance; scale chosen so it spans ~[-1, 1]
        # over the optimizer's bounds.
        raw = 1.0 - dist
        noisy = raw + self._rng.normal(0, self.noise_std)
        r = float(np.clip(noisy, -1.0, 1.0))
        return RewardMessage(r=r, raw_faa=raw, source="fake")
