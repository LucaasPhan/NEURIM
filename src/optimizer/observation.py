"""Observation model for the optimizer: a presentation window is NOT reduced to
a single reward = mean(faa_samples). Each window yields an estimate *and* its
uncertainty, so the GP can weight stable-EEG observations more than
motion-artifact ones (heteroscedastic noise).

    y_i = f(z_i) + eps_i,   eps_i ~ N(0, sigma_i^2)

with a different sigma_i^2 per observation.

Because consecutive FAA samples use overlapping 2-second windows, they are NOT
independent - treating every 250ms reading as an independent sample massively
under-estimates the noise. `effective_sample_size` corrects for that via the
integrated autocorrelation time, and the variance of the *mean estimate* is
divided by the effective N, not the raw N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """One presentation window's reward estimate and its uncertainty."""

    reward_mean: float
    reward_variance: float          # variance of the MEAN estimate = sigma_i^2 for the GP
    effective_sample_count: float   # autocorrelation-corrected, <= raw sample count
    artifact_fraction: float        # 0 = clean, 1 = all samples suspect (motion/saturation)
    t: float = 0.0                  # window timestamp / step index, for recency weighting


def effective_sample_size(samples: np.ndarray) -> float:
    """ESS accounting for autocorrelation of overlapping-window FAA samples.

    Uses the initial-positive-sequence estimate of the integrated
    autocorrelation time tau = 1 + 2*sum_k rho_k (truncated at the first
    non-positive lag), then ESS = n / tau. Overlapping 2s windows sampled every
    250ms are highly correlated, so tau > 1 and ESS is well below n.
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 2:
        return float(n)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return float(n)  # constant signal: every sample is identical, no extra info but not zero
    acf = np.correlate(x, x, mode="full")[n - 1:] / denom  # rho_0..rho_{n-1}, rho_0 = 1
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0.0:
            break
        tau += 2.0 * acf[k]
    ess = n / max(tau, 1.0)
    return float(min(max(ess, 1.0), n))


def window_statistics(
    samples,
    clip: tuple[float, float] = (-1.0, 1.0),
    t: float = 0.0,
    min_variance: float = 1e-6,
) -> Observation:
    """Turn a window of FAA reward samples into an Observation.

    artifact_fraction is proxied here by clip-saturation (samples pinned at the
    reward bounds, a signature of motion/arousal spikes). A real deployment
    should replace this with EMOTIV per-channel contact-quality / motion flags;
    the Observation contract stays identical.
    """
    s = np.asarray(samples, dtype=float)
    n = s.size
    if n == 0:
        return Observation(0.0, 1.0, 0.0, 1.0, t)

    mean = float(s.mean())
    ess = effective_sample_size(s)
    sample_var = float(s.var(ddof=1)) if n > 1 else 1.0
    # Variance of the MEAN uses the effective N (overlapping windows), not raw n.
    var_of_mean = sample_var / max(ess, 1.0)

    lo, hi = clip
    saturated = np.mean((s <= lo + 1e-6) | (s >= hi - 1e-6)) if n else 1.0
    # High within-window spread is also artifact-like; fold it in gently.
    spread_flag = min(1.0, sample_var / 0.25)  # ~1 when std ~ 0.5 of the clip range
    artifact_fraction = float(min(1.0, 0.7 * saturated + 0.3 * spread_flag))

    return Observation(
        reward_mean=mean,
        reward_variance=max(var_of_mean, min_variance),
        effective_sample_count=ess,
        artifact_fraction=artifact_fraction,
        t=t,
    )
