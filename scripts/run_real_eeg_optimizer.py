#!/usr/bin/env python3
"""Real-EEG optimizer client: drive run_streamdiffusion_server.py with a
z-stream produced by your ACTUAL brain signal, watching convergence morph on
the server. Identical to run_mock_optimizer.py except the reward comes from
real (or --mock) EEG via FAARewardComputer - see test_faa_stream.py - instead
of a scripted target. Same real OptimizerService, state machine, and
Interpolator either way; only the reward source differs.

    # real EPOC X headset, real StreamDiffusion server on a GPU box:
    python scripts/run_real_eeg_optimizer.py --server-url http://GPUHOST:8766

    # no headset needed, verify the wiring:
    python scripts/run_real_eeg_optimizer.py --mock --server-url http://localhost:8766

    # no server needed either, verify EEG -> FAA -> optimizer only:
    python scripts/run_real_eeg_optimizer.py --mock --dry-run

    # push the config's anchor prompts to the server first:
    python scripts/run_real_eeg_optimizer.py --server-url http://GPUHOST:8766 --set-anchors

NOTE: --set-anchors POSTs to /anchors, which only exists on run_diffusion_server.py
(SDXL-Turbo) - run_streamdiffusion_server.py has no such endpoint yet, so this
flag will fail against it. Edit config.yaml's anchor_prompts and restart the
StreamDiffusion server instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.generator.service import Interpolator
from src.optimizer.service import OptimizerService
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource
from src.signal_service.service import FAARewardSource, build_faa_service

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _save_frame(png_bytes: bytes, name: str = "live_frame.png") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "wb") as f:
        f.write(png_bytes)


class _SessionSnapshot:
    """Same start/end capture as run_demo.py's _SessionSnapshot, for the
    optional offline DiffMorpher showcase - see scripts/run_diffmorpher_showcase.py.
    Adapted for raw PNG bytes (this script's _post_render already returns the
    HTTP response body directly) rather than a base64-encoded FrameMessage.
    """

    def __init__(self):
        self._start_saved = False

    def on_frame(self, png_bytes: bytes) -> None:
        _save_frame(png_bytes)
        if not self._start_saved:
            _save_frame(png_bytes, "session_start.png")
            self._start_saved = True

    def save_end(self, png_bytes: bytes) -> None:
        _save_frame(png_bytes, "session_end.png")


def _post_render(base_url: str, z: np.ndarray, frame_size: int, timeout: float) -> bytes:
    import requests

    resp = requests.post(
        base_url.rstrip("/") + "/render",
        json={"z": [float(v) for v in z], "frame_size": frame_size},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"server returned {resp.status_code}: {resp.text[:200]}")
    return resp.content


def _post_anchors(base_url: str, prompts: list[str], timeout: float) -> None:
    import requests

    resp = requests.post(
        base_url.rstrip("/") + "/anchors",
        json={"anchor_prompts": prompts},
        timeout=timeout,
    )
    resp.raise_for_status()
    print(f"[real-eeg-opt] set {len(prompts)} anchor prompt(s) on the server")


def _cue(reward: float) -> str:
    if reward > 0.15:
        return "lean-in"
    if reward < -0.15:
        return "pull-away"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mock", action="store_true", help="synthetic EEG instead of real hardware")
    parser.add_argument("--baseline", type=float, default=None,
                        help="rest-calibration seconds (default: config faa.baseline_duration_s)")
    parser.add_argument("--server-url", default="http://localhost:8766",
                        help="base URL of run_streamdiffusion_server.py")
    parser.add_argument("--algorithm", choices=["hill_climb", "es_1p1", "gp_bo", "latent_turbo"], default=None)
    parser.add_argument("--frames-per-step", type=int, default=6,
                        help="interpolated frames rendered between optimizer steps")
    parser.add_argument("--set-anchors", action="store_true",
                        help="POST config.generator.anchor_prompts to the server before streaming")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't contact the server; just print the reward/state trace")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    args = parser.parse_args()

    config = Config.load()
    if args.algorithm:
        config.optimizer.algorithm = args.algorithm
    frame_size = config.generator.frame_size
    baseline_s = config.faa.baseline_duration_s if args.baseline is None else args.baseline

    if args.mock:
        eeg_source = MockEEGSource(config.eeg.channels, config.eeg.sample_rate_hz)
    else:
        client_id, client_secret = emotiv_credentials()
        eeg_source = EmotivCortexSource(client_id, client_secret)

    print(f"[real-eeg-opt] connecting ({'mock' if args.mock else 'EPOC X via Cortex'}) ...")
    eeg_source.connect()

    try:
        signal_service = build_faa_service(config, eeg_source)
        reward_source: FAARewardSource = signal_service.reward_source  # type: ignore[assignment]

        if baseline_s > 0:
            print(f"[real-eeg-opt] hold still and rest for {baseline_s:.0f}s to fit the baseline ...")
            baseline = calibrate_baseline(reward_source.computer, eeg_source.stream(), duration_s=baseline_s)
            print(f"[real-eeg-opt] baseline fitted: mean={baseline.mean:+.4f} std={baseline.std:.4f} n={baseline.n}")
        else:
            print("[real-eeg-opt] --baseline 0: no calibration, reward = clipped raw FAA (uncentered)")

        optimizer = OptimizerService(config)
        optimizer.notify_calibrated()
        interpolator = Interpolator()
        interpolator.set_target(np.asarray(optimizer.pending_candidate(), dtype=float))

        if not args.dry_run:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            if args.set_anchors:
                _post_anchors(args.server_url, list(config.generator.anchor_prompts), args.timeout)

        print(f"[real-eeg-opt] algorithm={config.optimizer.algorithm}  "
              f"{'DRY-RUN' if args.dry_run else args.server_url}")
        print(f"{'step':>4} {'state':>9} {'reward':>7} {'raw':>7}  cue        frames")
        print("-" * 56)

        frame_no = 0
        snapshot = _SessionSnapshot()

        def emit(z: np.ndarray) -> bytes | None:
            nonlocal frame_no
            if args.dry_run:
                frame_no += 1
                return None
            png = _post_render(args.server_url, z, frame_size, args.timeout)
            snapshot.on_frame(png)
            frame_no += 1
            return png

        try:
            while True:
                msg = reward_source.read_reward()
                if msg is None:
                    continue
                result = optimizer.observe_reward(msg.r)
                if result is None:
                    continue

                interpolator.set_target(np.asarray(result.z, dtype=float))
                for k in range(1, args.frames_per_step + 1):
                    emit(interpolator.sample(k / args.frames_per_step))

                raw = msg.raw_faa if msg.raw_faa is not None else float("nan")
                print(f"{result.step_index:>4} {result.state:>9} {result.reward_estimate:>+7.2f} "
                      f"{raw:>+7.2f}  {_cue(result.reward_estimate):<10} {frame_no}")
                if optimizer.state_machine.should_stop():
                    break
        except KeyboardInterrupt:
            print("\n[real-eeg-opt] interrupted")
        except Exception as exc:  # noqa: BLE001 - surface connection/render errors clearly
            sys.exit(f"[real-eeg-opt] server error: {exc}\n"
                     f"          is run_streamdiffusion_server.py listening at {args.server_url}?")

        final_state = optimizer.state_machine.state
        final_step = optimizer.state_machine.step_index

        # One clean render of the last *accepted* latent (not a mid-interpolation
        # blend) as the session's closing still - same logic as run_demo.py.
        if not args.dry_run:
            final_png = _post_render(args.server_url, optimizer.current_z(), frame_size, args.timeout)
            snapshot.save_end(final_png)

        print()
        print(f"[real-eeg-opt] final state={final_state}  steps={final_step}  frames emitted={frame_no}")
        if not args.dry_run:
            print(f"[real-eeg-opt] saved {OUT_DIR / 'live_frame.png'}, "
                  f"{OUT_DIR / 'session_start.png'}, and {OUT_DIR / 'session_end.png'}")
        if final_state == "settle":
            print("[real-eeg-opt] CONVERGED - the z-stream locked onto a candidate.")
            print("[real-eeg-opt] session settled - run scripts/run_diffmorpher_showcase.py "
                  "(in DiffMorpher's own venv) for a polished closing morph")
        else:
            print("[real-eeg-opt] did not settle within max_steps (try again, or check FAA signal quality "
                  "with scripts/test_faa_stream.py).")
    finally:
        eeg_source.close()


if __name__ == "__main__":
    main()
