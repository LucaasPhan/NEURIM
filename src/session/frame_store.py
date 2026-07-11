"""Live-frame and session snapshot storage."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class FrameStore:
    def __init__(self, directory: Path = DEFAULT_PROCESSED_DIR) -> None:
        self.directory = directory
        self._start_saved = False

    def save_live(self, png_bytes: bytes, capture_start: bool = False) -> Path:
        live_path = self.save(png_bytes, "live_frame.png")
        if capture_start and not self._start_saved:
            self.save(png_bytes, "session_start.png")
            self._start_saved = True
        return live_path

    def save_end(self, png_bytes: bytes) -> Path:
        return self.save(png_bytes, "session_end.png")

    def save_target(self, png_bytes: bytes) -> Path:
        """The finalized image the frontend displays (GET /api/target-frame)."""
        return self.save(png_bytes, "target_frame.png")

    def clear_target(self) -> None:
        """Drop a previous session's finalized image so the frontend only sees
        this session's result once it lands."""
        (self.directory / "target_frame.png").unlink(missing_ok=True)

    def clear_live(self) -> None:
        """Drop the previous session's live image before accepting a new prompt."""
        (self.directory / "live_frame.png").unlink(missing_ok=True)

    def save(self, png_bytes: bytes, name: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / name
        # Readers poll these files while renders are being written. Replace a
        # complete temporary file atomically so they never observe a partial PNG.
        with NamedTemporaryFile(dir=self.directory, prefix=f".{name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(png_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
