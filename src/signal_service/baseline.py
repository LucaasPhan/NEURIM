"""Per-session baseline calibration: 30s of rest before anything else runs."""

from __future__ import annotations

from .faa import FAARewardComputer, RunningBaseline


def calibrate_baseline(
    computer: FAARewardComputer,
    sample_iter,
    duration_s: float,
) -> RunningBaseline:
    """Consume samples from `sample_iter` for `duration_s`, fitting the baseline.

    `sample_iter` yields (timestamp, {channel: value}) tuples, same shape the
    live Signal service loop consumes. Returns the fitted RunningBaseline (also
    stored on `computer.baseline`).
    """
    readings: list[float] = []
    t0 = None
    for t, channel_values in sample_iter:
        if t0 is None:
            t0 = t
        computer.push_sample(channel_values)
        raw = computer.raw_value()
        if raw is not None:
            readings.append(raw)
        if t - t0 >= duration_s:
            break

    if not readings:
        raise RuntimeError("No FAA readings collected during baseline period")

    computer.baseline.fit(readings)
    return computer.baseline
