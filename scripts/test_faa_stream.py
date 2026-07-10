#!/usr/bin/env python3
"""Live FAA diagnostic: stream EEG and print, a few times a second, the F3/F4
alpha-band powers, the raw FAA index, and the mapped reward r(t).

Read-only sanity check on the Signal service - no optimizer, no generator.
Use it to confirm the headset gives a sane, responsive FAA before wiring the
signal into the loop, and to watch how leaning-in / pulling-away moves the
number.

    python scripts/test_faa_stream.py                # real EPOC X via Cortex
    python scripts/test_faa_stream.py --mock         # synthetic signal, no hardware
    python scripts/test_faa_stream.py --baseline 0   # skip baseline (prints raw FAA as reward)
    python scripts/test_faa_stream.py --baseline 15  # shorter 15s rest calibration

Column meanings:
    P(F3), P(F4)  alpha-band (8-13Hz) power in each frontal channel
    FAA           ln(P_F4) - ln(P_F3); >0 = more left-frontal activation
    reward        FAA z-scored against your resting baseline, clipped to [-1, 1]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource, wall_clock_pace
from src.signal_service.faa import FAARewardComputer


def _cue(reward: float) -> str:
    if reward > 0.15:
        return "lean-in  (approach)"
    if reward < -0.15:
        return "pull-away (avoid)"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mock", action="store_true", help="synthetic EEG instead of hardware")
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="rest-calibration seconds (default: config faa.baseline_duration_s; 0 to skip)",
    )
    args = parser.parse_args()

    config = Config.load()
    fs = config.eeg.sample_rate_hz
    baseline_s = config.faa.baseline_duration_s if args.baseline is None else args.baseline

    if args.mock:
        # Slow oscillating bias so F3/F4 asymmetry visibly swings, exercising
        # the whole path without a headset.
        eeg_source = MockEEGSource(
            config.eeg.channels, fs, bias_fn=lambda t: math.sin(2 * math.pi * t / 20.0)
        )
    else:
        client_id, client_secret = emotiv_credentials()
        eeg_source = EmotivCortexSource(client_id, client_secret)

    print(f"[faa] connecting ({'mock' if args.mock else 'EPOC X via Cortex'}) ...")
    eeg_source.connect()

    computer = FAARewardComputer(
        fs=fs,
        channel_left=config.faa.channel_left,
        channel_right=config.faa.channel_right,
        band=config.faa.band_hz,
        window_s=config.faa.window_s,
        clip=config.faa.clip,
    )
    print(
        f"[faa] channels: left={config.faa.channel_left} right={config.faa.channel_right}  "
        f"band={config.faa.band_hz[0]:.0f}-{config.faa.band_hz[1]:.0f}Hz  window={config.faa.window_s:.0f}s"
    )

    # A single generator instance drives both the baseline and live phases.
    sample_iter = (
        wall_clock_pace(eeg_source.stream(), fs) if args.mock else eeg_source.stream()
    )

    try:
        if baseline_s > 0:
            print(f"[faa] hold still and rest for {baseline_s:.0f}s to fit the baseline ...")
            baseline = calibrate_baseline(computer, sample_iter, duration_s=baseline_s)
            print(f"[faa] baseline fitted: mean={baseline.mean:+.4f} std={baseline.std:.4f} n={baseline.n}")
        else:
            print("[faa] --baseline 0: no calibration, reward = clipped raw FAA (uncentered)")

        emit_every = max(1, int(fs * config.faa.update_interval_s))
        print()
        print(f"{'t(s)':>7} {'P(F3)':>11} {'P(F4)':>11} {'FAA':>8} {'reward':>8}   cue")
        print("-" * 68)

        i = 0
        for t, sample in sample_iter:
            computer.push_sample(sample)
            i += 1
            if i % emit_every != 0:
                continue
            powers = computer.band_powers()
            raw = computer.raw_value()
            reward = computer.reward()
            if powers is None or raw is None or reward is None:
                continue  # window not full yet
            p_left, p_right = powers
            print(
                f"{t:>7.1f} {p_left:>11.4g} {p_right:>11.4g} "
                f"{raw:>+8.3f} {reward:>+8.2f}   {_cue(reward)}"
            )
    except KeyboardInterrupt:
        print("\n[faa] stopped.")
    finally:
        eeg_source.close()


if __name__ == "__main__":
    main()
