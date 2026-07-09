"""The risk register's central claim: if the loop doesn't converge on a
synthetic, noiseless-ish reward function driven by nothing more than
distance-to-target, no EEG signal will save it. These tests prove the
hill-climb + state machine actually finds a hidden target in a bounded
low-dim search space before reaching for (1+1)-ES or GP-BO.
"""

import numpy as np
import pytest

from src.common.config import Config
from src.optimizer.evolution import OnePlusOneES
from src.optimizer.hill_climb import MomentumHillClimb
from src.optimizer.service import OptimizerService
from src.optimizer.state_machine import StateMachine


def _run_hidden_target_search(optimizer_config, state_machine_config, target, max_steps=200, noise_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    config = Config(optimizer=optimizer_config, state_machine=state_machine_config)
    service = OptimizerService(config)
    service.notify_calibrated()

    for _ in range(max_steps * config.optimizer.reward_window_steps):
        z = service.pending_candidate()
        dist = np.linalg.norm(z - target[: len(z)])
        raw_reward = 1.0 - dist
        noisy_r = float(np.clip(raw_reward + rng.normal(0, noise_std), -1.0, 1.0))
        result = service.observe_reward(noisy_r)
        if result is not None and service.state_machine.should_stop():
            break
    return service


@pytest.mark.parametrize("algorithm", ["hill_climb", "es_1p1"])
def test_optimizer_converges_toward_hidden_target(algorithm):
    from src.common.config import OptimizerConfig, StateMachineConfig

    dims = 6
    target = np.array([0.5, -0.4, 0.3, 0.2, -0.3, 0.4])
    opt_config = OptimizerConfig(
        search_dims=dims,
        algorithm=algorithm,
        step_size_explore=0.3,
        step_size_refine_min=0.05,
        noise_threshold=0.05,
        reward_window_steps=2,
    )
    sm_config = StateMachineConfig(
        settle_reward_threshold=0.85,
        settle_motion_threshold=0.15,
        settle_patience_steps=3,
        max_steps=150,
    )

    service = _run_hidden_target_search(opt_config, sm_config, target, max_steps=150)

    final_z = service.current_z()
    dist = np.linalg.norm(final_z - target)
    # A random point in a [-1,1]^6 box is ~2.0 from any fixed target on
    # average; converging means landing meaningfully closer than that.
    assert dist < 0.6, f"{algorithm} failed to converge: dist={dist:.3f}, final_z={final_z}"


def test_hill_climb_rejects_within_noise_band():
    optimizer = MomentumHillClimb(dims=3, step_size=0.2, noise_threshold=0.1)
    candidate = optimizer.propose()
    z_before = optimizer.z.copy()
    accepted = optimizer.update(candidate, reward_before=0.5, reward_after=0.52)
    assert not accepted
    assert np.allclose(optimizer.z, z_before)


def test_hill_climb_accepts_clear_improvement():
    optimizer = MomentumHillClimb(dims=3, step_size=0.2, noise_threshold=0.05)
    candidate = optimizer.propose()
    accepted = optimizer.update(candidate, reward_before=0.0, reward_after=0.5)
    assert accepted
    assert np.allclose(optimizer.z, candidate)


def test_hill_climb_reverses_on_clear_drop():
    optimizer = MomentumHillClimb(dims=3, step_size=0.2, noise_threshold=0.05)
    original_velocity = optimizer.velocity.copy()
    candidate = optimizer.propose()
    optimizer.update(candidate, reward_before=0.5, reward_after=0.0)
    # Reversal is damped (currently x0.75), not a pure negation - check
    # direction flipped and magnitude shrank, rather than pinning the factor.
    assert np.dot(optimizer.velocity, original_velocity) < 0
    assert np.linalg.norm(optimizer.velocity) < np.linalg.norm(original_velocity)


def test_one_plus_one_es_adapts_sigma_on_success_streak():
    optimizer = OnePlusOneES(dims=3, sigma=0.2, noise_threshold=0.01, window=5)
    for _ in range(5):
        candidate = optimizer.propose()
        optimizer.update(candidate, reward_before=0.0, reward_after=1.0)  # always succeeds
    assert optimizer.sigma > 0.2  # success rate 1.0 > 1/5 -> sigma grows
