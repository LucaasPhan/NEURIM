"""GPU-side supervisor for manifest-driven diffusion processes."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generator.anchor_session import load_prompt_session_manifest, manifest_metadata
from src.server.api.diffusion_process import DiffusionProcessManager
from src.server.api.settings import REPO_ROOT


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:48] or "prompt"


class RestartRequest(BaseModel):
    filename: str | None = None
    manifest: dict[str, Any]


def create_app(
    repo_root: Path = REPO_ROOT,
    process_manager: DiffusionProcessManager | None = None,
) -> FastAPI:
    render_host = os.environ.get("NEURIM_DIFFUSION_HOST", "127.0.0.1")
    render_port = _env_int("NEURIM_DIFFUSION_PORT", 8766)
    public_render_url = os.environ.get(
        "NEURIM_DIFFUSION_PUBLIC_URL",
        f"http://{render_host}:{render_port}",
    ).rstrip("/")
    manager = process_manager or DiffusionProcessManager(
        repo_root=repo_root,
        host=render_host,
        port=render_port,
        python_executable=os.environ.get("NEURIM_DIFFUSION_PYTHON"),
        cuda_visible_devices=os.environ.get("NEURIM_DIFFUSION_CUDA_VISIBLE_DEVICES"),
        model=os.environ.get("NEURIM_DIFFUSION_MODEL", "stabilityai/sd-turbo"),
        startup_timeout_s=_env_float("NEURIM_DIFFUSION_STARTUP_TIMEOUT_S", 300.0),
    )
    output_dir = repo_root / "data" / "processed" / "prompt_sessions"
    app = FastAPI(title="NEURIM Diffusion Supervisor", version="0.1.0")
    app.state.process_manager = manager

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "diffusion": manager.status(), "render_url": public_render_url}

    @app.post("/diffusion/restart")
    def restart(request: RestartRequest) -> dict[str, Any]:
        manifest_payload = request.manifest
        user_prompt = str(manifest_payload.get("user_prompt", "")).strip()
        if not user_prompt:
            raise HTTPException(status_code=400, detail="manifest.user_prompt is required")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = request.filename or f"{stamp}-{_slug(user_prompt)}.json"
        filename = Path(filename).name
        manifest_path = output_dir / filename
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
            manifest = load_prompt_session_manifest(manifest_path)
            print(
                f"[diffusion-supervisor] restarting diffusion for prompt={manifest.user_prompt!r} "
                f"manifest={manifest_path}",
                flush=True,
            )
            remote_manifest = manager.restart(manifest_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[diffusion-supervisor] restart failed: {exc}", flush=True)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        print(
            f"[diffusion-supervisor] diffusion ready render_url={public_render_url}",
            flush=True,
        )
        return {
            "ok": True,
            "render_url": public_render_url,
            "manifest_path": str(manifest_path),
            "manifest": manifest_metadata(manifest),
            "remote_manifest": remote_manifest,
            "diffusion": manager.status(),
        }

    @app.post("/diffusion/stop")
    def stop() -> dict[str, Any]:
        manager.stop()
        return {"ok": True, "diffusion": manager.status()}

    @app.on_event("shutdown")
    def shutdown() -> None:
        if _env_bool("NEURIM_DIFFUSION_STOP_ON_SUPERVISOR_EXIT", True):
            manager.stop()

    return app


app = create_app()
