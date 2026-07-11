"""Shared optimizer-to-diffusion rendering loop primitives."""

from __future__ import annotations

import numpy as np

from src.common.config import Config
from src.generator.service import Interpolator
from src.optimizer.service import OptimizerService

from .diffusion_client import DiffusionClient
from .frame_store import FrameStore


class OptimizerRenderLoop:
    def __init__(
        self,
        config: Config,
        frames_per_step: int,
        client: DiffusionClient | None,
        frame_store: FrameStore,
        capture_snapshots: bool = False,
        finalizer=None,
        finalize_prompt: str = "",
    ) -> None:
        self.config = config
        self.frames_per_step = frames_per_step
        self.client = client
        self.frame_store = frame_store
        self.capture_snapshots = capture_snapshots
        self.finalizer = finalizer
        self.finalize_prompt = finalize_prompt
        self.optimizer = OptimizerService(config)
        self.optimizer.notify_calibrated()
        self.interpolator = Interpolator()
        self.interpolator.set_target(np.asarray(self.optimizer.pending_candidate(), dtype=float))
        self.frame_count = 0

    def render_candidate(self, z: np.ndarray) -> bytes | None:
        self.interpolator.set_target(np.asarray(z, dtype=float))
        last_png = None
        for index in range(1, self.frames_per_step + 1):
            last_png = self.emit(self.interpolator.sample(index / self.frames_per_step))
        return last_png

    def emit(self, z: np.ndarray) -> bytes | None:
        self.frame_count += 1
        if self.client is None:
            return None
        png = self.client.render(z, self.config.generator.frame_size)
        self.frame_store.save_live(png, capture_start=self.capture_snapshots)
        return png

    def save_final_frame(self) -> tuple[bool, str | None]:
        if self.client is None:
            return False, "render client is unavailable"
        png = self.client.render(self.optimizer.current_z(), self.config.generator.frame_size)
        self.frame_store.save_end(png)
        # Finalize the last morphed frame through OpenAI, then hand it to the
        # frontend. On any failure the raw frame is still delivered so the
        # session always ends with something to show.
        finalized, refined, error = self._finalize(png)
        self.frame_store.save_target(finalized)
        return refined, error

    def _finalize(self, png: bytes) -> tuple[bytes, bool, str | None]:
        if self.finalizer is None:
            return png, False, "OpenAI image refinement is unavailable"
        try:
            return self.finalizer.finalize(png, self.finalize_prompt), True, None
        except Exception as exc:  # noqa: BLE001
            print(f"[optimizer-loop] image finalize failed, using raw final frame: {exc}")
            return png, False, str(exc)
