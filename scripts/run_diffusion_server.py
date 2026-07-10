#!/usr/bin/env python3
"""HTTP server that runs the local diffusion model for a remote NEURIM client.

Run this on the GPU machine, then set the client config:

  generator.backend: "remote_diffusion"
  generator.remote_diffusion_url: "http://GPU_HOST:8766"

At startup it fits a projector over the config's anchor_prompts (encoded through
the diffusion model's own text encoders) so the client can send a low-dim search
vector `z` and get back a genuine latent-morph keyframe: z -> to_embedding(z) ->
prompt_embeds render, at a pinned seed. POST /render accepts either {"z": [...]}
(the morph path) or {"prompt": "..."} (a plain text render, e.g. for curl tests).

NOTE: the morph only *varies* if there are >= 2 anchor_prompts in config.yaml -
with a single anchor the projector's subspace is degenerate (to_embedding returns
the same mean embedding for every z) and every frame is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config
from src.generator.diffusion_pipeline import DiffusionGenerator
from src.optimizer.projection import PCAProjector


class DiffusionRenderServer:
    def __init__(self, generator: DiffusionGenerator, projector: PCAProjector | None, seed: int, dims: int):
        self.generator = generator
        self.projector = projector
        self.seed = seed
        self.dims = dims
        self.anchor_prompts: list[str] = []
        self.lock = threading.Lock()

    def render_png(self, payload: dict) -> bytes:
        frame_size = int(payload.get("frame_size", 512))
        seed = int(payload.get("seed", self.seed))
        z = payload.get("z")
        with self.lock:
            if z is not None:
                if self.projector is None:
                    raise RuntimeError(
                        "server received a z vector but no projector was fit - "
                        "check that config.generator.anchor_prompts is non-empty"
                    )
                embedding = self.projector.to_embedding(np.asarray(z, dtype=float))
                image = self.generator.render_from_embedding(embedding, seed=seed)
            else:
                prompt = payload.get("prompt") or "a little brown puppy"
                image = self.generator.render_from_prompt(prompt, seed=seed)
        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def set_anchor_prompts(self, prompts: list[str]) -> None:
        cleaned = [prompt.strip() for prompt in prompts if prompt.strip()]
        if len(cleaned) < 2:
            raise ValueError("remote diffusion requires at least two anchor prompts")
        with self.lock:
            self.projector = _fit_projector(self.generator, cleaned, self.dims)
            self.anchor_prompts = cleaned


def make_handler(render_server: DiffusionRenderServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path not in {"/render", "/anchors"}:
                self.send_error(404, "expected POST /render or POST /anchors")
                return

            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/anchors":
                    prompts = payload.get("anchor_prompts", [])
                    if not isinstance(prompts, list):
                        raise ValueError("anchor_prompts must be a list")
                    render_server.set_anchor_prompts([str(prompt) for prompt in prompts])
                    self._send_json({"ok": True, "count": len(prompts)})
                    return
                png = render_server.render_png(payload)
            except Exception as exc:  # noqa: BLE001 - report server-side failures to the client.
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            self.send_response(200)
            self.send_header("content-type", "image/png")
            self.send_header("content-length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[diffusion-server] {self.address_string()} - {fmt % args}")

    return Handler


def _fit_projector(generator: DiffusionGenerator, anchor_prompts: list[str], dims: int) -> PCAProjector | None:
    if not anchor_prompts:
        print("[diffusion-server] WARNING: no anchor_prompts in config - z-morph disabled, "
              "only {\"prompt\": ...} renders will work")
        return None
    if len(anchor_prompts) < 2:
        print(f"[diffusion-server] WARNING: only {len(anchor_prompts)} anchor prompt(s) - the morph "
              "will be STATIC (projector subspace is degenerate). Add more anchor_prompts for variation.")
    print(f"[diffusion-server] encoding {len(anchor_prompts)} anchor prompt(s) and fitting projector...")
    embeddings = generator.encode_prompts(anchor_prompts)
    projector = PCAProjector(dims=dims).fit(embeddings)
    print(f"[diffusion-server] projector fit: {dims}-dim search space over "
          f"{embeddings.shape[0]} prompts (embed dim {embeddings.shape[1]})")
    return projector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model-id", default=None, help="defaults to config.generator.diffusion_model_id")
    parser.add_argument("--steps", type=int, default=None, help="defaults to config.generator.diffusion_steps")
    parser.add_argument("--seed", type=int, default=None, help="defaults to config.generator.remote_diffusion_seed")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config.load()
    model_id = args.model_id or config.generator.diffusion_model_id
    steps = args.steps if args.steps is not None else config.generator.diffusion_steps
    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed

    generator = DiffusionGenerator(
        model_id=model_id,
        num_inference_steps=steps,
        device=args.device,
    )
    projector = _fit_projector(generator, config.generator.anchor_prompts, config.optimizer.search_dims)

    render_server = DiffusionRenderServer(generator, projector, seed, config.optimizer.search_dims)
    render_server.anchor_prompts = list(config.generator.anchor_prompts)
    handler = make_handler(render_server)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[diffusion-server] listening on http://{args.host}:{args.port} (seed={seed}, steps={steps})")
    server.serve_forever()


if __name__ == "__main__":
    main()
