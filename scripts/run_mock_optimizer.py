#!/usr/bin/env python3
"""Mock optimizer client: drive run_streamdiffusion_server.py with a z-stream
that naturally converges onto a fixed target image, so you can watch the real
convergence process morph on the server without EEG or a headset.

This is NOT a hand-rolled easing curve - it runs your ACTUAL OptimizerService,
state machine, and Interpolator against a scripted reward that scores closeness
to a fixed target latent (the "fixed image semantic"). The only thing mocked is
the reward: instead of FAA from a brain, reward = 1 - ||candidate - target||,
plus noise. So the z-trajectory that reaches the server is produced by the same
explore -> refine -> settle code the real system uses.

Each accepted step, it interpolates toward the new candidate (exactly like
LocalOrchestrator does) and POSTs each interpolated z to the server's /render,
saving the returned PNG - giving a smooth morph that settles on the target.

    # against a real StreamDiffusion server on a GPU box:
    python scripts/run_mock_optimizer.py --server-url http://GPUHOST:8766 --seed 3

    # verify the converging trajectory with no server (prints z + distance):
    python scripts/run_mock_optimizer.py --dry-run --seed 3

    # push the config's anchor prompts to the server first:
    python scripts/run_mock_optimizer.py --server-url http://GPUHOST:8766 --set-anchors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config
from src.generator.service import Interpolator
from src.optimizer.service import OptimizerService
from src.signal_service.fake_reward import ScriptedRewardSource

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "mock_stream"


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
    print(f"[mock-opt] set {len(prompts)} anchor prompt(s) on the server")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server-url", default="http://localhost:8766",
                        help="base URL of run_streamdiffusion_server.py")
    parser.add_argument("--seed", type=int, default=0, help="which fixed target image to converge to")
    parser.add_argument("--algorithm", choices=["hill_climb", "es_1p1", "gp_bo", "latent_turbo"], default=None)
    parser.add_argument("--noise", type=float, default=0.05, help="reward noise std (higher = harder)")
    parser.add_argument("--frames-per-step", type=int, default=6,
                        help="interpolated frames rendered between optimizer steps")
    parser.add_argument("--set-anchors", action="store_true",
                        help="POST config.generator.anchor_prompts to the server before streaming")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't contact the server; just print the converging z-trajectory")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    args = parser.parse_args()

    config = Config.load()
    if args.algorithm:
        config.optimizer.algorithm = args.algorithm
    dims = config.optimizer.search_dims
    frame_size = config.generator.frame_size

    rng = np.random.default_rng(args.seed)
    target = rng.uniform(-0.8, 0.8, size=dims)

    optimizer = OptimizerService(config)
    optimizer.notify_calibrated()
    reward_source = ScriptedRewardSource(
        target=target,
        get_current_z=optimizer.pending_candidate,
        noise_std=args.noise,
        seed=args.seed,
    )
    interpolator = Interpolator()
    interpolator.set_target(np.asarray(optimizer.pending_candidate(), dtype=float))

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.set_anchors:
            _post_anchors(args.server_url, list(config.generator.anchor_prompts), args.timeout)

    print(f"[mock-opt] target=seed{args.seed} ({dims}-D)  algorithm={config.optimizer.algorithm}  "
          f"noise={args.noise}  {'DRY-RUN' if args.dry_run else args.server_url}")
    print(f"{'step':>4} {'state':>9} {'reward':>7} {'dist':>6}  frames")
    print("-" * 46)

    frame_no = 0
    start_dist = float(np.linalg.norm(optimizer.current_z() - target))
    max_ticks = config.state_machine.max_steps * config.optimizer.reward_window_steps + 50

    def emit(z: np.ndarray) -> None:
        nonlocal frame_no
        if args.dry_run:
            frame_no += 1
            return
        png = _post_render(args.server_url, z, frame_size, args.timeout)
        (OUT_DIR / f"frame_{frame_no:04d}.png").write_bytes(png)
        (OUT_DIR / "mock_live.png").write_bytes(png)  # overwrite target for live viewers
        frame_no += 1

    try:
        for _ in range(max_ticks):
            r = reward_source.read_reward().r
            result = optimizer.observe_reward(r)
            if result is None:
                continue

            interpolator.set_target(np.asarray(result.z, dtype=float))
            for k in range(1, args.frames_per_step + 1):
                emit(interpolator.sample(k / args.frames_per_step))

            dist = float(np.linalg.norm(optimizer.current_z() - target))
            print(f"{result.step_index:>4} {result.state:>9} {result.reward_estimate:>+7.2f} "
                  f"{dist:>6.2f}  {frame_no}")
            if optimizer.state_machine.should_stop():
                break
    except (KeyboardInterrupt,) as exc:  # noqa: PERF203
        print(f"\n[mock-opt] interrupted ({exc})")
    except Exception as exc:  # noqa: BLE001 - surface connection/render errors clearly
        sys.exit(f"[mock-opt] server error: {exc}\n"
                 f"          is run_streamdiffusion_server.py listening at {args.server_url}?")

    final_state = optimizer.state_machine.state
    final_dist = float(np.linalg.norm(optimizer.current_z() - target))
    closer = 1.0 - final_dist / start_dist if start_dist > 0 else 0.0
    print()
    print(f"[mock-opt] final state={final_state}  started {start_dist:.2f} -> ended {final_dist:.2f} "
          f"({closer:.0%} closer)  frames emitted={frame_no}")
    if not args.dry_run:
        print(f"[mock-opt] frames in {OUT_DIR}/  (frame_0000.png ... , live at mock_live.png)")
    if final_state == "settle":
        print("[mock-opt] CONVERGED - the z-stream locked onto the target image.")
    else:
        print("[mock-opt] did not fully settle (try --seed, lower --noise, or --frames-per-step).")


if __name__ == "__main__":
    main()
