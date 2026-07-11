#!/usr/bin/env python3
"""
Continuous Stable Diffusion dog-breed morphing.

The animation interpolates:
1. The diffusion noise latent z with spherical interpolation (SLERP).
2. CLIP text-conditioning embeddings between consecutive dog breeds.

Controls
--------
q or Esc : quit
Space    : pause/resume
s        : save the current frame as a PNG
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Helps unsupported MPS operations fall back to CPU on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline


DEFAULT_BREEDS = [
    "Golden Retriever",
    "German Shepherd",
    "Siberian Husky",
    "Pembroke Welsh Corgi",
    "Shiba Inu",
    "Dalmatian",
    "Standard Poodle",
]

PROMPT_TEMPLATE = (
    "centered studio portrait photograph of a {breed} dog, "
    "head and shoulders, looking directly at the camera, "
    "same neutral gray background, soft even lighting, "
    "symmetrical composition, highly detailed realistic fur"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously morph Stable Diffusion latents across dog breeds."
    )
    parser.add_argument(
        "--model",
        default="stabilityai/sd-turbo",
        help="Hugging Face Diffusers model ID.",
    )
    parser.add_argument(
        "--breeds",
        nargs="+",
        default=DEFAULT_BREEDS,
        help='Breed sequence. Quote multiword breeds, e.g. "Golden Retriever".',
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Square output size in pixels; must be divisible by 8.",
    )
    parser.add_argument(
        "--frames-per-transition",
        type=int,
        default=32,
        help="Number of generated frames between consecutive breeds.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Denoising steps per frame. SD-Turbo is designed for 1-4 steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2306,
        help="Seed for the continuous latent stream.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of full breed cycles; 0 means run until q/Esc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional MP4 path. Frames are encoded at --video-fps.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=24.0,
        help="Playback frame rate for the optional MP4.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.breeds) < 2:
        raise ValueError("Provide at least two dog breeds.")
    if args.size < 256 or args.size % 8 != 0:
        raise ValueError("--size must be at least 256 and divisible by 8.")
    if args.frames_per_transition < 2:
        raise ValueError("--frames-per-transition must be at least 2.")
    if not 1 <= args.steps <= 4:
        raise ValueError("--steps must be between 1 and 4 for SD-Turbo.")
    if args.cycles < 0:
        raise ValueError("--cycles cannot be negative.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")


def select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_pipeline(
    model_id: str, device: torch.device, dtype: torch.dtype
) -> StableDiffusionPipeline:
    kwargs: dict[str, object] = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except Exception:
        # Some compatible repositories do not provide an fp16 variant.
        kwargs.pop("variant", None)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


@torch.inference_mode()
def encode_breed_prompts(
    pipe: StableDiffusionPipeline,
    breeds: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [PROMPT_TEMPLATE.format(breed=breed) for breed in breeds]
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds


def random_latent(
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
    channels: int,
    vae_scale_factor: int,
) -> torch.Tensor:
    shape = (
        1,
        channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    # Generate on CPU for deterministic behavior on CUDA and MPS alike.
    return torch.randn(
        shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)


def slerp(z0: torch.Tensor, z1: torch.Tensor, amount: float) -> torch.Tensor:
    """Spherical interpolation of two complete latent tensors."""
    if amount <= 0.0:
        return z0
    if amount >= 1.0:
        return z1

    a = z0.float().reshape(-1)
    b = z1.float().reshape(-1)
    a_unit = a / a.norm().clamp_min(1e-8)
    b_unit = b / b.norm().clamp_min(1e-8)
    dot = torch.dot(a_unit, b_unit).clamp(-1.0, 1.0)

    # Nearly parallel vectors are numerically safer with linear interpolation.
    if torch.abs(dot) > 0.9995:
        result = torch.lerp(z0.float(), z1.float(), amount)
    else:
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        result = (
            torch.sin((1.0 - amount) * omega) / sin_omega * z0.float()
            + torch.sin(amount * omega) / sin_omega * z1.float()
        )

    return result.to(dtype=z0.dtype)


def cosine_ease(t: float) -> float:
    """Smooth interpolation with zero velocity at both anchor points."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


@torch.inference_mode()
def render_frame(
    pipe: StableDiffusionPipeline,
    prompt_embeds: torch.Tensor,
    latent: torch.Tensor,
    size: int,
    steps: int,
) -> np.ndarray:
    image = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        latents=latent,
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=0.0,
        output_type="pil",
    ).images[0]

    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_hud(
    frame: np.ndarray,
    source_breed: str,
    target_breed: str,
    amount: float,
    seconds_per_frame: float,
    paused: bool,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, 74), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0.0, output)

    label = f"{source_breed}  ->  {target_breed}   morph={amount:0.2f}"
    performance = (
        "paused"
        if paused
        else f"{seconds_per_frame:0.2f} sec/frame | q quit | space pause | s save"
    )
    cv2.putText(
        output,
        label,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        performance,
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return output


def open_video_writer(
    output_path: Path | None,
    size: int,
    fps: float,
) -> cv2.VideoWriter | None:
    if output_path is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (size, size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video file: {output_path}")
    return writer


def save_still(frame: np.ndarray) -> Path:
    path = Path(f"dog_morph_{datetime.now():%Y%m%d_%H%M%S}.png")
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save frame to {path}")
    return path


def read_key(delay_ms: int = 1) -> int:
    return cv2.waitKey(delay_ms) & 0xFF


def main() -> int:
    args = parse_args()
    validate_args(args)

    device, dtype = select_device()
    print(f"Device: {device}; dtype: {dtype}")
    if device.type == "cpu":
        print(
            "Warning: no CUDA or Apple MPS accelerator was detected. "
            "Generation will be slow; try --size 384."
        )

    print(f"Loading {args.model} ...")
    pipe = load_pipeline(args.model, device, dtype)
    print("Encoding breed prompts ...")
    breed_embeds = encode_breed_prompts(pipe, args.breeds, device)

    latent_channels = int(pipe.unet.config.in_channels)
    vae_scale_factor = int(pipe.vae_scale_factor)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    current_latent = random_latent(
        generator,
        device,
        dtype,
        args.size,
        args.size,
        latent_channels,
        vae_scale_factor,
    )

    writer = open_video_writer(args.output, args.size, args.video_fps)
    window_name = "Continuous Stable Diffusion Dog Morph"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.size, args.size)

    paused = False
    should_quit = False
    last_display: np.ndarray | None = None
    ema_seconds: float | None = None
    rendered_any_frame = False
    cycle_index = 0

    try:
        while args.cycles == 0 or cycle_index < args.cycles:
            for breed_index, source_breed in enumerate(args.breeds):
                target_index = (breed_index + 1) % len(args.breeds)
                target_breed = args.breeds[target_index]
                next_latent = random_latent(
                    generator,
                    device,
                    dtype,
                    args.size,
                    args.size,
                    latent_channels,
                    vae_scale_factor,
                )

                # Skip duplicated t=0 anchors after the very first transition.
                first_frame = 0 if not rendered_any_frame else 1

                for frame_index in range(
                    first_frame, args.frames_per_transition
                ):
                    while paused and not should_quit:
                        if last_display is not None:
                            cv2.imshow(window_name, last_display)
                        key = read_key(30)
                        if key in (ord("q"), 27):
                            should_quit = True
                        elif key == ord(" "):
                            paused = False
                        elif key == ord("s") and last_display is not None:
                            saved = save_still(last_display)
                            print(f"Saved {saved}")

                    if should_quit:
                        break

                    raw_t = frame_index / (args.frames_per_transition - 1)
                    amount = cosine_ease(raw_t)

                    latent = slerp(current_latent, next_latent, amount)
                    prompt_embeds = torch.lerp(
                        breed_embeds[breed_index : breed_index + 1].float(),
                        breed_embeds[target_index : target_index + 1].float(),
                        amount,
                    ).to(dtype=breed_embeds.dtype)

                    started = time.perf_counter()
                    frame = render_frame(
                        pipe,
                        prompt_embeds,
                        latent,
                        args.size,
                        args.steps,
                    )
                    elapsed = time.perf_counter() - started
                    ema_seconds = (
                        elapsed
                        if ema_seconds is None
                        else 0.85 * ema_seconds + 0.15 * elapsed
                    )

                    display = add_hud(
                        frame,
                        source_breed,
                        target_breed,
                        amount,
                        ema_seconds,
                        paused=False,
                    )
                    cv2.imshow(window_name, display)
                    last_display = display
                    rendered_any_frame = True

                    if writer is not None:
                        writer.write(frame)

                    key = read_key(1)
                    if key in (ord("q"), 27):
                        should_quit = True
                        break
                    if key == ord(" "):
                        paused = True
                    elif key == ord("s"):
                        saved = save_still(display)
                        print(f"Saved {saved}")

                current_latent = next_latent
                if should_quit:
                    break

            cycle_index += 1
            if should_quit:
                break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved video to {args.output}")
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)