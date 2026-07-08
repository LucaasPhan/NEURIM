"""Upgrades over the plain hill-climb, for when there's time: a (1+1)
evolution strategy with the 1/5-success-rule step adaptation, and a
Gaussian-process Bayesian optimizer whose posterior uncertainty gives
backtracking/global-search behavior for free.

Both implement the same propose()/update() interface as MomentumHillClimb so
the Optimizer service can swap algorithms via config without touching the
state machine.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class OnePlusOneES:
    """(1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma."""

    def __init__(
        self,
        dims: int,
        bounds: float = 1.0,
        sigma: float = 0.3,
        noise_threshold: float = 0.08,
        window: int = 10,
        rng: np.random.Generator | None = None,
    ):
        self.dims = dims
        self.bounds = bounds
        self.sigma = sigma
        self.noise_threshold = noise_threshold
        self.window = window
        self.rng = rng or np.random.default_rng()

        self.z = np.zeros(dims)
        self.best_z = self.z.copy()
        self.best_reward = -np.inf
        self._history: deque[int] = deque(maxlen=window)

    def propose(self) -> np.ndarray:
        candidate = self.z + self.rng.normal(scale=self.sigma, size=self.dims)
        return np.clip(candidate, -self.bounds, self.bounds)

    def update(self, candidate: np.ndarray, reward_before: float, reward_after: float) -> bool:
        if reward_after > self.best_reward:
            self.best_reward = reward_after
            self.best_z = candidate.copy()

        success = (reward_after - reward_before) > self.noise_threshold
        self._history.append(1 if success else 0)
        if success:
            self.z = candidate

        if len(self._history) == self.window:
            rate = sum(self._history) / self.window
            if rate > 0.2:
                self.sigma *= 1.2
            elif rate < 0.2:
                self.sigma *= 0.85
        return success

    def set_step_size(self, step_size: float) -> None:
        self.sigma = step_size

    def revert_to_best(self) -> None:
        self.z = self.best_z.copy()
        self._history.clear()


class GPBanditOptimizer:
    """GP-BO with a UCB acquisition, maximized by random search over the box
    (cheap and good enough at 8-16 dims). The GP's own uncertainty naturally
    favors under-explored regions, which is where the spec's "backtracking
    behavior for free" comes from.
    """

    def __init__(
        self,
        dims: int,
        bounds: float = 1.0,
        kappa: float = 2.0,
        n_candidates: int = 512,
        rng: np.random.Generator | None = None,
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        self.dims = dims
        self.bounds = bounds
        self.kappa = kappa
        self.n_candidates = n_candidates
        self.rng = rng or np.random.default_rng()

        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
        self._zs: list[np.ndarray] = []
        self._rewards: list[float] = []

        self.z = np.zeros(dims)
        self.best_z = self.z.copy()
        self.best_reward = -np.inf

    def propose(self) -> np.ndarray:
        if len(self._zs) < 3:
            # Not enough data to fit a GP yet - explore randomly near current z.
            candidate = self.z + self.rng.normal(scale=0.3, size=self.dims)
            return np.clip(candidate, -self.bounds, self.bounds)

        self._gp.fit(np.array(self._zs), np.array(self._rewards))
        candidates = self.rng.uniform(-self.bounds, self.bounds, size=(self.n_candidates, self.dims))
        mu, sigma = self._gp.predict(candidates, return_std=True)
        ucb = mu + self.kappa * sigma
        return candidates[int(np.argmax(ucb))]

    def update(self, candidate: np.ndarray, reward_before: float, reward_after: float) -> bool:
        self._zs.append(candidate.copy())
        self._rewards.append(reward_after)
        if reward_after > self.best_reward:
            self.best_reward = reward_after
            self.best_z = candidate.copy()
        accepted = reward_after >= reward_before
        if accepted:
            self.z = candidate
        return accepted

    def set_step_size(self, step_size: float) -> None:
        pass  # GP-BO's exploration/exploitation trade-off is via kappa, not step size.

    def revert_to_best(self) -> None:
        self.z = self.best_z.copy()
