#!/usr/bin/env python3
"""Build-order step 1: prove the optimizer/generator loop converges before
EEG ever touches it.

Two modes:
  --mode scripted   Fully automated. A hidden target latent stands in for
                     "what the person wants"; reward = similarity to it. No
                     human, no keyboard - this is what tests/CI runs to prove
                     convergence (see tests/test_optimizer.py for the
                     assertions; this script is for eyeballing it directly).
  --mode keyboard    Interactive. Up/down arrows nudge reward like a
                     human's "warmer/colder" gut feeling; watch
                     data/processed/live_frame.png update as the shape morphs.

If the loop doesn't converge to something a person points at and says "yes"
here, no EEG signal will save it - this is the whole point of shipping the
fake-reward path first.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config
from src.generator.service import GeneratorService
from src.optimizer.service import OptimizerService
from src.orchestrator.orchestrator import LocalOrchestrator
from src.signal_service.fake_reward import ScriptedRewardSource
from src.signal_service.service import SignalService

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def run_scripted(config: Config, seed: int) -> None:
    rng = np.random.default_rng(seed)
    target = rng.uniform(-0.8, 0.8, size=config.optimizer.search_dims)
    print(f"[fake-loop] hidden target z = {np.round(target, 2)}")

    optimizer = OptimizerService(config)
    # Score whatever candidate is currently being evaluated, not the last
    # accepted point - that's frozen for the whole window and would carry no
    # signal about the thing actually on screen.
    reward_source = ScriptedRewardSource(
        target=target, get_current_z=lambda: optimizer.pending_candidate(), seed=seed
    )
    signal_service = SignalService(reward_source, update_interval_s=config.faa.update_interval_s)
    generator = GeneratorService(config)
    orchestrator = LocalOrchestrator(config, signal_service, generator, optimizer=optimizer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    def on_frame(frame_msg):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 15 == 0:  # ~ every half second at 30fps
            _save_frame(frame_msg)

    orchestrator.on_frame = on_frame

    wall_budget = config.state_machine.max_steps * config.loop.optimizer_step_interval_s * 1.5

    async def main():
        await orchestrator.calibrate()
        await orchestrator.run(max_wall_seconds=wall_budget)

    asyncio.run(main())

    final_z = orchestrator.optimizer.current_z()
    dist = float(np.linalg.norm(final_z - target))
    baseline_dist = float(np.linalg.norm(target))  # optimizer starts at z=0
    improvement = 1.0 - dist / baseline_dist if baseline_dist > 0 else 0.0
    state = orchestrator.optimizer.state_machine.state
    steps = orchestrator.optimizer.state_machine.step_index
    print(f"[fake-loop] final state={state} steps={steps} dist_to_target={dist:.3f} "
          f"(started {baseline_dist:.3f}, {improvement:.0%} closer)")
    print(f"[fake-loop] final z      = {np.round(final_z, 2)}")
    if state == "settle":
        print("[fake-loop] CONVERGED - loop locked onto a candidate.")
    elif improvement > 0.5:
        print("[fake-loop] converging, did not fully settle within the step budget "
              "(this is expected some fraction of the time with plain hill-climb - "
              "see README.md; try --algorithm gp_bo or more steps).")
    else:
        print("[fake-loop] did not make meaningful progress this run.")


def run_keyboard(config: Config) -> None:
    from src.signal_service.fake_reward import KeyboardRewardSource

    print("[fake-loop] up/down arrows nudge reward. Ctrl+C to stop.")
    print(f"[fake-loop] watching {OUT_DIR / 'live_frame.png'} - open it in an image viewer that auto-refreshes.")

    generator = GeneratorService(config)
    reward_source = KeyboardRewardSource()
    signal_service = SignalService(reward_source, update_interval_s=config.faa.update_interval_s)
    orchestrator = LocalOrchestrator(config, signal_service, generator)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orchestrator.on_frame = _save_frame

    async def main():
        await orchestrator.calibrate()
        await orchestrator.run()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        reward_source.close()


def _save_frame(frame_msg) -> None:
    import base64

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "live_frame.png", "wb") as f:
        f.write(base64.b64decode(frame_msg.frame_b64))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["scripted", "keyboard"], default="scripted")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", choices=["hill_climb", "es_1p1", "gp_bo"], default=None)
    args = parser.parse_args()

    cfg = Config.load()
    cfg.generator.backend = "procedural"  # no GPU needed to prove the loop converges
    if args.algorithm:
        cfg.optimizer.algorithm = args.algorithm
    cfg.state_machine.max_steps = args.max_steps

    if args.mode == "scripted":
        run_scripted(cfg, seed=args.seed)
    else:
        run_keyboard(cfg)
