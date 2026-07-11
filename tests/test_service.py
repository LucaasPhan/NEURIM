"""Regression tests for FAARewardSource (KNOWN_ISSUES #1).

The bug: once the FAA window filled during warm-up, later reads pushed zero new
samples, so raw_faa/reward() froze on the warm-up window and the optimizer
hill-climbed a single constant. After the fix the window keeps sliding by
~one update interval per read, so the reward tracks fresh EEG.
"""

import numpy as np

from src.signal_service.eeg_sources import MockEEGSource
from src.signal_service.faa import FAARewardComputer
from src.signal_service.service import FAARewardSource

FRONTAL_CHANNELS = ["F7", "F8", "AF3", "AF4", "F3", "F4", "FC5", "FC6"]


def _build_source(samples_per_read: int) -> FAARewardSource:
    fs = 128
    # Slowly-varying asymmetry so the true FAA genuinely moves over the run.
    eeg = MockEEGSource(FRONTAL_CHANNELS, sample_rate_hz=fs, bias_fn=lambda t: np.sin(0.5 * t))
    computer = FAARewardComputer(fs=fs, window_s=1.0)
    return FAARewardSource(eeg, computer, samples_per_read=samples_per_read)


def test_raw_faa_varies_across_reads_after_warmup():
    source = _build_source(samples_per_read=32)

    raws = []
    for _ in range(100):
        msg = source.read_reward()
        assert msg is not None
        raws.append(msg.raw_faa)

    # The exact regression KNOWN_ISSUES #1 reproduced: raw_faa constant across
    # every read. Sliding the window makes it vary read to read.
    assert len(set(raws)) > 50, "raw_faa froze after warm-up (window is not sliding)"


def test_reward_is_not_constant_across_reads():
    source = _build_source(samples_per_read=32)

    rewards = [source.read_reward().r for _ in range(100)]
    assert len(set(rewards)) > 1, "reward froze after warm-up"


def test_samples_per_read_is_floored_at_one():
    # A zero/negative count must never disable sliding entirely.
    source = _build_source(samples_per_read=0)
    assert source.samples_per_read == 1

    raws = [source.read_reward().raw_faa for _ in range(100)]
    assert len(set(raws)) > 1, "raw_faa froze with samples_per_read floored to 1"
