#!/usr/bin/env python3
"""Diagnostic for the StreamDiffusion backend - run it on the GPU server in
streamdiffusion-env to localize WHERE the pipeline is failing, since the three
failure modes look similar in the live demo but have different fixes:

  Test A (official API, no z-injection): does SD-Turbo + our wrapper config
     generate a coherent image AT ALL from a plain text prompt? If this is
     garbage/grey, the problem is model/config (t_index_list, acceleration,
     model download) - NOT our z-injection.

  Test B (our full path): bootstrap + inject z -> embedding -> img2img stream.
     Saves a z-sweep so you can flip through frames and see whether it (a) looks
     like the anchor prompts and (b) morphs smoothly. If Test A is fine but this
     is bad, the problem is our injection / projector / img2img strength.

Writes PNGs to data/processed/streamdiffusion_test/. Eyeball them.

    python scripts/test_streamdiffusion.py --streamdiffusion-repo ~/chun/StreamDiffusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "streamdiffusion_test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--streamdiffusion-repo", required=True)
    parser.add_argument("--model-id", default="stabilityai/sd-turbo")
    parser.add_argument("--t-index-list", default="32,45")
    parser.add_argument("--acceleration", default="xformers", choices=["none", "xformers", "tensorrt"])
    parser.add_argument("--sweep-frames", type=int, default=12, help="frames in the Test B z-sweep")
    parser.add_argument("--sweep-dim", type=int, default=0, help="which z dimension to sweep in Test B")
    args = parser.parse_args()

    # Imported here so build_wrapper's sys.path insertion happens first.
    from run_streamdiffusion_server import StreamDiffusionRenderServer, _fit_projector, build_wrapper

    config = Config.load()
    seed = config.generator.remote_diffusion_seed
    frame_size = config.generator.frame_size
    t_index_list = [int(x) for x in args.t_index_list.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wrapper = build_wrapper(config, args.streamdiffusion_repo, args.model_id, t_index_list,
                            args.acceleration, seed, frame_size)

    # --- Test A: raw generation via the official API, no injection ------------
    anchor = config.generator.anchor_prompts[0] if config.generator.anchor_prompts else "a golden retriever puppy"
    print(f"[test] A: official txt2img for prompt: {anchor!r}")
    wrapper.update_prompt(anchor)
    img = wrapper.txt2img()
    if isinstance(img, list):
        img = img[0]
    img.save(OUT_DIR / "A_official_txt2img.png")
    print(f"[test] A: saved {OUT_DIR / 'A_official_txt2img.png'} - if this isn't a coherent image, "
          "the problem is model/config, not our injection")

    # --- Test B: our injected z -> embedding -> bootstrap + img2img morph -----
    projector, embed_shape = _fit_projector(wrapper, config.generator.anchor_prompts, config.optimizer.search_dims)
    server = StreamDiffusionRenderServer(wrapper, projector, embed_shape, frame_size)
    server._bootstrap_frame().save(OUT_DIR / "B_bootstrap.png")

    dim = min(args.sweep_dim, projector.dims - 1)
    print(f"[test] B: sweeping z[{dim}] from -1 to +1 over {args.sweep_frames} frames via our render path")
    for i in range(args.sweep_frames):
        z = np.zeros(projector.dims)
        z[dim] = -1.0 + 2.0 * i / max(args.sweep_frames - 1, 1)
        png = server.render_png({"z": z.tolist()})
        (OUT_DIR / f"B_sweep_{i:02d}.png").write_bytes(png)
    print(f"[test] B: saved B_sweep_00..{args.sweep_frames - 1:02d}.png in {OUT_DIR} - "
          "flip through them: should look like the anchor prompts and change smoothly")


if __name__ == "__main__":
    main()
