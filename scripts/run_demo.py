#!/usr/bin/env python3
"""Phase 3: the real thing. Ties EEG (or --mock), the optimizer, and the
generator (procedural, --backend diffusion, --backend remote_diffusion, or --backend openai)
together via the Orchestrator,
optionally serving frames to a frontend over websockets (--serve).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import errno
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.generator.service import GeneratorService
from src.orchestrator.orchestrator import LocalOrchestrator, WebSocketOrchestrator
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource
from src.signal_service.service import build_faa_service

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _save_frame(frame_msg, name: str = "live_frame.png") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "wb") as f:
        f.write(base64.b64decode(frame_msg.frame_b64))


class _SessionSnapshot:
    """Saves the session's first and final frame to data/processed/, for the
    OFFLINE DiffMorpher showcase (scripts/run_diffmorpher_showcase.py) - a
    polished start->end morph video generated after the live loop finishes.
    This is deliberately separate from the real-time crossfade/keyframe loop:
    DiffMorpher's per-pair LoRA fine-tune + 50-step SD2.1 sampling is far too
    slow to run inside the ~1s-per-step online loop, so it only ever runs once,
    after the session settles, against these two saved stills.
    """

    def __init__(self):
        self._start_saved = False

    def on_frame(self, frame_msg) -> None:
        _save_frame(frame_msg)
        if not self._start_saved:
            _save_frame(frame_msg, "session_start.png")
            self._start_saved = True

    def save_end(self, frame_msg) -> None:
        _save_frame(frame_msg, "session_end.png")


async def run_local(config: Config, eeg_source) -> None:
    signal_service = build_faa_service(config, eeg_source)
    generator = GeneratorService(config)
    snapshot = _SessionSnapshot()
    orchestrator = LocalOrchestrator(config, signal_service, generator, on_frame=snapshot.on_frame)
    print("[demo] calibrating baseline...")
    await orchestrator.calibrate()
    print("[demo] running - writing frames to", OUT_DIR / "live_frame.png")
    await orchestrator.run()
    final_state = orchestrator.optimizer.state_machine.state
    final_step = orchestrator.optimizer.state_machine.step_index
    print(f"[demo] state={final_state} steps={final_step}")

    # One clean render of the last *accepted* latent (not a mid-interpolation
    # blend) as the session's closing still.
    final_frame = generator.render(
        orchestrator.optimizer.current_z(),
        final_step,
        state=final_state,
        reward_estimate=orchestrator._last_reward_estimate,
    )
    snapshot.save_end(final_frame)
    print(f"[demo] saved {OUT_DIR / 'session_start.png'} and {OUT_DIR / 'session_end.png'}")
    if final_state == "settle":
        print("[demo] session settled - run scripts/run_diffmorpher_showcase.py "
              "(in DiffMorpher's own venv) for a polished closing morph")


async def run_served(config: Config, host: str, port: int) -> None:
    generator = GeneratorService(config)
    hub = WebSocketOrchestrator(config, generator, host=host, port=port)
    browser_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"[demo] starting websocket hub on ws://{browser_host}:{hub.port} "
          "(Signal service connects with {\"role\": \"signal\"}, frontend with {\"role\": \"display\"})")
    await hub.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="use synthetic EEG instead of real hardware")
    parser.add_argument("--backend", choices=["procedural", "diffusion", "remote_diffusion", "openai"], default=None)
    parser.add_argument("--remote-url", default=None, help="remote diffusion server URL when using remote_diffusion")
    parser.add_argument("--algorithm", choices=["hill_climb", "es_1p1", "gp_bo", "latent_turbo"], default=None)
    parser.add_argument("--serve", action="store_true", help="run the websocket hub instead of local mode")
    parser.add_argument("--host", default="0.0.0.0", help="websocket host when using --serve")
    parser.add_argument("--port", type=int, default=8765, help="websocket port when using --serve")
    args = parser.parse_args()

    config = Config.load()
    if args.backend:
        config.generator.backend = args.backend
    if args.remote_url:
        config.generator.remote_diffusion_url = args.remote_url
    if args.algorithm:
        config.optimizer.algorithm = args.algorithm

    if args.serve:
        try:
            asyncio.run(run_served(config, args.host, args.port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                browser_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
                print(
                    f"[demo] port {args.port} is already in use; a websocket hub may already be running at "
                    f"ws://{browser_host}:{args.port}\n"
                    f"[demo] stop the old process or start this one on another port, e.g. "
                    f"--port {args.port + 1}"
                )
            else:
                raise
        except KeyboardInterrupt:
            print("\n[demo] stopped")
        return

    if args.mock:
        eeg_source = MockEEGSource(config.eeg.channels, config.eeg.sample_rate_hz)
    else:
        client_id, client_secret = emotiv_credentials()
        eeg_source = EmotivCortexSource(client_id, client_secret)
    eeg_source.connect()

    try:
        asyncio.run(run_local(config, eeg_source))
    finally:
        eeg_source.close()


if __name__ == "__main__":
    main()
