import warnings

import numpy as np
import pytest

from src.optimizer.latent_turbo import NoiseAwareLatentTuRBO
from src.optimizer.observation import Observation, effective_sample_size, window_statistics

# sklearn's GP hyperparameter fit warns when length scales hit their bounds on
# tiny/degenerate data - expected in these small tests, not a failure.
warnings.filterwarnings("ignore")


# ---- observation model ------------------------------------------------------

def test_effective_sample_size_below_n_for_autocorrelated():
    rng = np.random.default_rng(0)
    # Strongly autocorrelated series (random walk) - overlapping-window-like.
    walk = np.cumsum(rng.normal(size=200))
    ess = effective_sample_size(walk)
    assert 1.0 <= ess < 200
    assert ess < 50  # heavy autocorrelation -> far fewer effective samples than 200


def test_effective_sample_size_near_n_for_iid():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=400)
    ess = effective_sample_size(iid)
    assert ess > 200  # independent samples -> ESS close to n


def test_window_statistics_variance_uses_effective_n():
    # Autocorrelated window: variance-of-mean must exceed sample_var / raw_n,
    # because the effective N is smaller than the raw count.
    rng = np.random.default_rng(2)
    samples = np.cumsum(rng.normal(scale=0.1, size=40)) * 0.05
    obs = window_statistics(samples, clip=(-1, 1))
    raw_n = len(samples)
    sample_var = np.var(samples, ddof=1)
    assert obs.effective_sample_count < raw_n
    assert obs.reward_variance >= sample_var / raw_n
    assert 0.0 <= obs.artifact_fraction <= 1.0


def test_window_statistics_flags_saturation_as_artifact():
    clean = window_statistics([0.1, 0.12, 0.09, 0.11], clip=(-1, 1))
    saturated = window_statistics([1.0, 1.0, -1.0, 1.0], clip=(-1, 1))
    assert saturated.artifact_fraction > clean.artifact_fraction


# ---- optimizer: uncertainty-aware acceptance --------------------------------

def test_uncertain_observation_is_not_credibly_better():
    opt = NoiseAwareLatentTuRBO(dims=3, bounds=1.0, rng=np.random.default_rng(0))
    opt.observe(opt.propose(), Observation(0.5, 0.001, 5, 0.0, 1))  # incumbent = 0.5
    # Higher raw mean but huge uncertainty -> not a credible improvement.
    accepted_uncertain = opt.observe(opt.propose(), Observation(0.55, 1.0, 1, 0.9, 2))
    assert not accepted_uncertain
    # Clearly higher with low uncertainty -> accepted.
    accepted_confident = opt.observe(opt.propose(), Observation(0.9, 0.001, 5, 0.0, 3))
    assert accepted_confident
    assert opt.best_reward == pytest.approx(0.9)


# ---- optimizer: trust-region adaptation -------------------------------------

def test_trust_region_shrinks_on_repeated_failures():
    opt = NoiseAwareLatentTuRBO(dims=3, bounds=1.0, failure_tol=3, length_init=0.5,
                                rng=np.random.default_rng(0))
    opt.observe(opt.propose(), Observation(0.5, 0.001, 5, 0.0, 1))  # incumbent
    length_before = opt.length
    for i in range(4):  # clearly-worse observations -> failures
        opt.observe(opt.propose(), Observation(0.1, 0.001, 5, 0.0, 2 + i))
    assert opt.length < length_before


def test_trust_region_grows_on_repeated_credible_improvements():
    opt = NoiseAwareLatentTuRBO(dims=3, bounds=1.0, success_tol=3, length_init=0.3,
                                length_max=1.6, rng=np.random.default_rng(0))
    opt.observe(opt.propose(), Observation(0.1, 0.001, 5, 0.0, 1))  # incumbent
    length_before = opt.length
    r = 0.3
    for i in range(4):  # steadily improving, confident
        opt.observe(opt.propose(), Observation(r, 0.001, 5, 0.0, 2 + i))
        r += 0.2
    assert opt.length > length_before


# ---- optimizer: convergence -------------------------------------------------

def test_latent_turbo_converges_toward_target():
    dims, bounds = 6, 1.5
    rng = np.random.default_rng(3)
    target = rng.uniform(-0.7, 0.7, size=dims) * bounds
    opt = NoiseAwareLatentTuRBO(dims=dims, bounds=bounds, rng=np.random.default_rng(4))
    start = np.linalg.norm(opt.best_z - target)
    noise = np.random.default_rng(5)
    for t in range(200):
        z = opt.propose()
        dist = np.linalg.norm(z - target) / bounds
        r = 1.0 - dist + noise.normal(0, 0.03)
        opt.observe(z, Observation(float(r), 0.002, 3.0, 0.0, t))
    final = np.linalg.norm(opt.best_z - target)
    assert final < 0.5 * start  # got at least halfway to the target
