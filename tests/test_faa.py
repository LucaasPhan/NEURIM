import numpy as np

from src.signal_service.faa import FAARewardComputer, RunningBaseline, band_power, raw_faa


def _alpha_signal(fs, duration_s, amplitude=1.0, freq=10.0, seed=0):
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return amplitude * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.05, n)


def test_band_power_higher_for_larger_amplitude():
    fs = 128.0
    quiet = _alpha_signal(fs, 2.0, amplitude=0.2)
    loud = _alpha_signal(fs, 2.0, amplitude=2.0)
    assert band_power(loud, fs, (8, 13)) > band_power(quiet, fs, (8, 13))


def test_raw_faa_positive_when_right_stronger():
    fs = 128.0
    window = {
        "F3": _alpha_signal(fs, 2.0, amplitude=0.5, seed=1),
        "F4": _alpha_signal(fs, 2.0, amplitude=2.0, seed=2),
    }
    value = raw_faa(window, fs, "F3", "F4", (8, 13))
    assert value > 0


def test_raw_faa_negative_when_left_stronger():
    fs = 128.0
    window = {
        "F3": _alpha_signal(fs, 2.0, amplitude=2.0, seed=1),
        "F4": _alpha_signal(fs, 2.0, amplitude=0.5, seed=2),
    }
    value = raw_faa(window, fs, "F3", "F4", (8, 13))
    assert value < 0


def test_running_baseline_z_score():
    baseline = RunningBaseline()
    baseline.fit([0.0, 0.1, -0.1, 0.05, -0.05])
    assert abs(baseline.z_score(baseline.mean)) < 1e-9
    assert baseline.z_score(baseline.mean + baseline.std) == 1.0


def test_faa_reward_computer_ready_and_clipped():
    fs = 128.0
    computer = FAARewardComputer(fs=fs, window_s=1.0, clip=(-1.0, 1.0))
    computer.baseline.fit([0.0])
    computer.baseline.std = 0.01  # tiny std -> any real signal saturates the clip

    assert not computer.ready()
    n_samples = int(fs * 1.0) + 5
    for i in range(n_samples):
        computer.push_sample({"F3": 0.1 * np.sin(i), "F4": 2.0 * np.sin(i)})
    assert computer.ready()

    r = computer.reward()
    assert r is not None
    assert -1.0 <= r <= 1.0
