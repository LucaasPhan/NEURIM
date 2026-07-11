"""Client for a remote GPU diffusion supervisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.generator.prompt_curation import PromptCurationManifest


class RemoteDiffusionSupervisorClient:
    """Remote equivalent of DiffusionProcessManager.

    The local API owns EEG and optimization. The GPU supervisor owns restarting
    the manifest-driven diffusion process on the private GPU machine.
    """

    def __init__(self, supervisor_url: str, timeout_s: float = 300.0) -> None:
        self.supervisor_url = supervisor_url.rstrip("/")
        self.timeout_s = timeout_s
        self.base_url = ""

    def restart(self, manifest_path: Path, manifest: PromptCurationManifest | None = None) -> dict[str, Any]:
        if manifest is None:
            raise RuntimeError("remote diffusion supervisor restart requires the curated manifest payload")

        import requests

        response = requests.post(
            f"{self.supervisor_url}/diffusion/restart",
            json={
                "filename": manifest_path.name,
                "manifest": manifest.to_dict(),
            },
            timeout=self.timeout_s,
        )
        if response.status_code != 200:
            detail = response.text[:300]
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("detail"):
                    detail = str(payload["detail"])
            except Exception:
                pass
            raise RuntimeError(
                f"diffusion supervisor returned {response.status_code}: {detail}"
            )
        payload = response.json()
        render_url = str(payload.get("render_url", "")).strip()
        if not render_url:
            raise RuntimeError("diffusion supervisor response did not include render_url")
        self.base_url = render_url.rstrip("/")
        return payload

    def stop(self) -> None:
        return None
