#!/usr/bin/env python3
"""Offline closing showcase: a polished DiffMorpher morph between a session's
first and final frame (https://github.com/Kevin-thu/DiffMorpher).

This is deliberately NOT part of the live loop. DiffMorpher fine-tunes a LoRA
per image (~200 steps each) and samples SD2.1 at 50 steps x up to 50 frames -
far too slow for the ~1s-per-step online optimizer, and built on a different
base model than the SDXL-Turbo path (src/generator/diffusion_pipeline.py).
Run this once, after a session settles, against the two stills scripts/run_demo.py
already saves to data/processed/{session_start,session_end}.png.

CRITICAL: DiffMorpher pins diffusers==0.17.1 / transformers==4.34.1 (pre-SDXL).
Installing those into this project's torch-env would break run_diffusion_server.py.
DiffMorpher must live in its OWN venv - see the --diffmorpher-python argument.

One-time setup (on whichever machine has the GPU for this step - can be the
same GPU server, just a separate environment):

    git clone https://github.com/Kevin-thu/DiffMorpher.git
    cd DiffMorpher
    python3 -m venv .venv-diffmorpher
    source .venv-diffmorpher/bin/activate
    pip install -r requirements.txt

Then, from the NEURIM repo (any environment - this script only shells out):

    python scripts/run_diffmorpher_showcase.py \\
        --diffmorpher-repo /path/to/DiffMorpher \\
        --diffmorpher-python /path/to/DiffMorpher/.venv-diffmorpher/bin/python
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Our anchor prompts all share this scaffold (see config.yaml) - a generic
# caption in the same style is enough for DiffMorpher's inversion/LoRA step,
# since it doesn't need to describe the image precisely, just plausibly.
_DEFAULT_PROMPT = "a close portrait photo of a puppy sitting on soft white bedding"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diffmorpher-repo", required=True, help="path to a cloned Kevin-thu/DiffMorpher checkout")
    parser.add_argument("--diffmorpher-python", required=True,
                         help="python executable inside DiffMorpher's OWN venv (not this project's)")
    parser.add_argument("--start", default=str(DATA_DIR / "session_start.png"),
                         help="first image (default: the session's saved start frame)")
    parser.add_argument("--end", default=str(DATA_DIR / "session_end.png"),
                         help="second image (default: the session's saved settled frame)")
    parser.add_argument("--prompt-0", default=_DEFAULT_PROMPT)
    parser.add_argument("--prompt-1", default=_DEFAULT_PROMPT)
    parser.add_argument("--output", default=str(DATA_DIR / "showcase.gif"))
    parser.add_argument("--num-frames", type=int, default=16, help="DiffMorpher default; 50 for a longer morph")
    parser.add_argument("--duration-ms", type=int, default=100, help="ms per frame in the output GIF")
    parser.add_argument("--no-adain", action="store_true", help="disable AdaIN normalization (on by default)")
    parser.add_argument("--no-reschedule", action="store_true", help="disable reschedule sampling (on by default)")
    parser.add_argument("--no-lora", action="store_true",
                         help="skip per-image LoRA fine-tuning entirely - much faster, lower identity fidelity")
    args = parser.parse_args()

    diffmorpher_repo = Path(args.diffmorpher_repo).resolve()
    main_py = diffmorpher_repo / "main.py"
    if not main_py.exists():
        sys.exit(f"[showcase] {main_py} not found - is --diffmorpher-repo a real DiffMorpher checkout?")

    start_path = Path(args.start).resolve()
    end_path = Path(args.end).resolve()
    for p in (start_path, end_path):
        if not p.exists():
            sys.exit(f"[showcase] {p} not found - run scripts/run_demo.py to completion first "
                      "(it saves session_start.png / session_end.png)")

    with tempfile.TemporaryDirectory(prefix="diffmorpher_out_") as tmp_out:
        cmd = [
            args.diffmorpher_python,
            str(main_py),
            "--image_path_0", str(start_path),
            "--prompt_0", args.prompt_0,
            "--image_path_1", str(end_path),
            "--prompt_1", args.prompt_1,
            "--output_path", tmp_out,
            "--num_frames", str(args.num_frames),
            "--duration", str(args.duration_ms),
        ]
        if not args.no_adain:
            cmd.append("--use_adain")
        if not args.no_reschedule:
            cmd.append("--use_reschedule")
        if args.no_lora:
            cmd.append("--no_lora")

        print(f"[showcase] running DiffMorpher in its own venv "
              f"({'no LoRA - fast' if args.no_lora else '~200 LoRA steps per image - slow'})...")
        print(f"[showcase] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(diffmorpher_repo))
        if result.returncode != 0:
            sys.exit(f"[showcase] DiffMorpher exited with code {result.returncode} - see its output above")

        produced = Path(tmp_out) / "output.gif"
        if not produced.exists():
            sys.exit(f"[showcase] expected {produced} but DiffMorpher didn't create it - check its console output")

        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(produced, output_path)

    print(f"[showcase] saved {output_path}")


if __name__ == "__main__":
    main()
