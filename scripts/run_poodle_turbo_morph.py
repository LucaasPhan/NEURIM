#!/usr/bin/env python3
"""Single-process demo: Latent TuRBO searches a dog-breed morph for poodle.

This starts from the same idea as scripts/continuous_dog_latent_morph.py:
render SD-Turbo frames from both prompt embeddings and diffusion noise latents.

The difference is that the path is no longer a fixed breed list. A
NoiseAwareLatentTuRBO optimizer proposes a continuous breed-weight vector z.
A fake EEG/FAA reward source returns noisy reward samples that are highest when
the rendered mixture is poodle-like. The optimizer keeps exploring different
breed mixtures, then settles once the poodle weight stays high.

Controls
--------
q or Esc : quit
Space    : pause/resume
s        : save the current frame as a PNG

Example
-------
    python scripts/run_poodle_turbo_morph.py --size 384 --steps 1
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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.latent_turbo import NoiseAwareLatentTuRBO
from src.optimizer.observation import Observation, window_statistics


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
        description="Optimize a continuous dog morph toward poodle with fake EEG reward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--breeds", nargs="+", default=DEFAULT_BREEDS)
    parser.add_argument("--target-breed", default="Standard Poodle")
    parser.add_argument("--size", type=int, default=512, help="square output size; divisible by 8")
    parser.add_argument("--steps", type=int, default=1, help="SD-Turbo denoising steps, usually 1-4")
    parser.add_argument("--seed", type=int, default=2306)
    parser.add_argument("--max-optimizer-steps", type=int, default=80)
    parser.add_argument("--frames-per-candidate", type=int, default=8)
    parser.add_argument("--reward-samples", type=int, default=12)
    parser.add_argument("--reward-noise", type=float, default=0.08)
    parser.add_argument("--artifact-rate", type=float, default=0.05)
    parser.add_argument("--settle-weight", type=float, default=0.82)
    parser.add_argument("--settle-patience", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="optional MP4 output path")
    parser.add_argument("--video-fps", type=float, default=12.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.breeds) < 2:
        raise ValueError("Provide at least two breeds.")
    if args.target_breed not in args.breeds:
        raise ValueError(f"--target-breed must be one of --breeds; got {args.target_breed!r}")
    if args.size < 256 or args.size % 8 != 0:
        raise ValueError("--size must be at least 256 and divisible by 8.")
    if not 1 <= args.steps <= 4:
        raise ValueError("--steps should be between 1 and 4 for SD-Turbo.")
    if args.frames_per_candidate < 2:
        raise ValueError("--frames-per-candidate must be at least 2.")
    if args.reward_samples < 2:
        raise ValueError("--reward-samples must be at least 2.")


def select_device() -> tuple[torch.device, torch.dtype]:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    import torch
    from diffusers import StableDiffusionPipeline

    kwargs: dict[str, object] = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("variant", None)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


def encode_breed_prompts(
    pipe: StableDiffusionPipeline,
    breeds: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    import torch

    prompts = [PROMPT_TEMPLATE.format(breed=breed) for breed in breeds]
    with torch.inference_mode():
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
    size: int,
    channels: int,
    vae_scale_factor: int,
) -> torch.Tensor:
    import torch

    shape = (1, channels, size // vae_scale_factor, size // vae_scale_factor)
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def make_breed_latents(
    n: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    size: int,
    channels: int,
    vae_scale_factor: int,
) -> torch.Tensor:
    import torch

    return torch.cat(
        [
            random_latent(generator, device, dtype, size, channels, vae_scale_factor)
            for _ in range(n)
        ],
        dim=0,
    )


def softmax_weights(z: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    x = np.asarray(z, dtype=float) * temperature
    x = x - x.max()
    w = np.exp(x)
    return w / max(float(w.sum()), 1e-12)


def blend_prompt_embeds(breed_embeds: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    import torch

    w = torch.tensor(weights, device=breed_embeds.device, dtype=torch.float32)
    blended = torch.sum(breed_embeds.float() * w[:, None, None], dim=0, keepdim=True)
    return blended.to(dtype=breed_embeds.dtype)


def blend_noise_latents(breed_latents: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    """Weighted latent blend, renormalized to preserve a plausible noise norm."""
    import torch

    w = torch.tensor(weights, device=breed_latents.device, dtype=torch.float32)
    blended = torch.sum(breed_latents.float() * w[:, None, None, None], dim=0, keepdim=True)
    target_norm = torch.sum(
        torch.linalg.vector_norm(breed_latents.float().reshape(breed_latents.shape[0], -1), dim=1)
        * w
    )
    current_norm = torch.linalg.vector_norm(blended.reshape(1, -1), dim=1).clamp_min(1e-8)
    blended = blended * (target_norm / current_norm).reshape(1, 1, 1, 1)
    return blended.to(dtype=breed_latents.dtype)


def cosine_ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def interpolate_z(z0: np.ndarray, z1: np.ndarray, amount: float) -> np.ndarray:
    return z0 + (z1 - z0) * cosine_ease(amount)


def render_frame(
    pipe: StableDiffusionPipeline,
    breed_embeds: torch.Tensor,
    breed_latents: torch.Tensor,
    z: np.ndarray,
    size: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    import torch

    weights = softmax_weights(z)
    prompt_embeds = blend_prompt_embeds(breed_embeds, weights)
    latent = blend_noise_latents(breed_latents, weights)
    with torch.inference_mode():
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
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), weights


class FakeFAAReward:
    """Noisy EEG-like reward: high only when the poodle anchor dominates."""

    def __init__(
        self,
        target_index: int,
        noise_std: float,
        artifact_rate: float,
        sample_count: int,
        seed: int,
    ):
        self.target_index = target_index
        self.noise_std = noise_std
        self.artifact_rate = artifact_rate
        self.sample_count = sample_count
        self.rng = np.random.default_rng(seed)
        self._ar_noise = 0.0

    def true_reward(self, z: np.ndarray) -> float:
        weights = softmax_weights(z)
        poodle_weight = float(weights[self.target_index])
        # Reward range is approximately [-1, 1]. Penalize diffuse mixtures so a
        # confident poodle anchor beats a vague "all breeds" blend.
        entropy = -float(np.sum(weights * np.log(weights + 1e-12))) / np.log(len(weights))
        reward = 2.0 * poodle_weight - 1.0 - 0.15 * entropy
        return float(np.clip(reward, -1.0, 1.0))

    def observe(self, z: np.ndarray, t: int) -> Observation:
        base = self.true_reward(z)
        samples = []
        for _ in range(self.sample_count):
            self._ar_noise = 0.65 * self._ar_noise + self.rng.normal(0.0, self.noise_std)
            sample = base + self._ar_noise
            if self.rng.random() < self.artifact_rate:
                sample += self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.3, 0.8)
            samples.append(float(np.clip(sample, -1.0, 1.0)))
        return window_statistics(samples, clip=(-1.0, 1.0), t=t, min_variance=1e-5)


def top_breeds(breeds: Sequence[str], weights: np.ndarray, n: int = 3) -> str:
    order = np.argsort(weights)[::-1][:n]
    return " | ".join(f"{breeds[i]} {weights[i]:.2f}" for i in order)


def add_hud(
    frame: np.ndarray,
    breeds: Sequence[str],
    weights: np.ndarray,
    step: int,
    reward: float,
    target_weight: float,
    state: str,
    seconds_per_frame: float,
    paused: bool,
) -> np.ndarray:
    import cv2

    output = frame.copy()
    height, width = output.shape[:2]
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, 96), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.58, output, 0.42, 0.0, output)

    lines = [
        f"Latent TuRBO fake EEG target: Standard Poodle | step={step:03d} | state={state}",
        f"reward={reward:+.3f} | poodle_weight={target_weight:.2f} | {top_breeds(breeds, weights)}",
        "paused" if paused else f"{seconds_per_frame:.2f} sec/frame | q quit | space pause | s save",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (14, 27 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255) if i < 2 else (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return output


def open_video_writer(output_path: Path | None, size: int, fps: float) -> cv2.VideoWriter | None:
    import cv2

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
    import cv2

    path = Path(f"poodle_turbo_morph_{datetime.now():%Y%m%d_%H%M%S}.png")
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save frame to {path}")
    return path


def read_key(delay_ms: int = 1) -> int:
    import cv2

    return cv2.waitKey(delay_ms) & 0xFF


def make_optimizer(dims: int, seed: int) -> NoiseAwareLatentTuRBO:
    return NoiseAwareLatentTuRBO(
        dims=dims,
        bounds=1.0,
        min_obs=min(6, max(3, dims - 1)),
        length_init=1.6,
        length_min=0.04,
        length_max=2.0,
        success_tol=2,
        failure_tol=4,
        n_candidates=384,
        improve_delta=0.02,
        success_prob_tau=0.55,
        motion_limit=0.35,
        recovery_reward_floor=0.0,
        rng=np.random.default_rng(seed),
    )


def main() -> int:
    import cv2
    import torch

    args = parse_args()
    validate_args(args)

    device, dtype = select_device()
    print(f"Device: {device}; dtype: {dtype}")
    if device.type == "cpu":
        print("Warning: no CUDA or MPS accelerator detected. Try --size 384 for CPU.")

    print(f"Loading {args.model} ...")
    pipe = load_pipeline(args.model, device, dtype)
    print("Encoding breed anchors ...")
    breed_embeds = encode_breed_prompts(pipe, args.breeds, device)

    latent_channels = int(pipe.unet.config.in_channels)
    vae_scale_factor = int(pipe.vae_scale_factor)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    breed_latents = make_breed_latents(
        len(args.breeds),
        generator,
        device,
        dtype,
        args.size,
        latent_channels,
        vae_scale_factor,
    )

    target_index = args.breeds.index(args.target_breed)
    optimizer = make_optimizer(len(args.breeds), args.seed + 17)
    reward_source = FakeFAAReward(
        target_index=target_index,
        noise_std=args.reward_noise,
        artifact_rate=args.artifact_rate,
        sample_count=args.reward_samples,
        seed=args.seed + 31,
    )

    writer = open_video_writer(args.output, args.size, args.video_fps)
    window_name = "NEURIM fake EEG Latent TuRBO dog morph"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.size, args.size)

    display_z = np.zeros(len(args.breeds), dtype=float)
    candidate = optimizer.propose()
    paused = False
    should_quit = False
    last_display: np.ndarray | None = None
    ema_seconds: float | None = None
    last_reward = float("-inf")
    settle_streak = 0
    state = "explore"

    print(f"Target breed: {args.target_breed}")
    print("Optimizer will explore breed mixtures until poodle reward dominates.")

    try:
        for step in range(1, args.max_optimizer_steps + 1):
            if should_quit:
                break

            for frame_index in range(args.frames_per_candidate):
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

                amount = frame_index / max(args.frames_per_candidate - 1, 1)
                render_z = interpolate_z(display_z, candidate, amount)

                started = time.perf_counter()
                frame, weights = render_frame(
                    pipe,
                    breed_embeds,
                    breed_latents,
                    render_z,
                    args.size,
                    args.steps,
                )
                elapsed = time.perf_counter() - started
                ema_seconds = elapsed if ema_seconds is None else 0.85 * ema_seconds + 0.15 * elapsed

                display = add_hud(
                    frame,
                    args.breeds,
                    weights,
                    step,
                    last_reward if np.isfinite(last_reward) else 0.0,
                    float(weights[target_index]),
                    state,
                    ema_seconds,
                    paused=False,
                )
                cv2.imshow(window_name, display)
                last_display = display
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

            if should_quit:
                break

            display_z = candidate.copy()
            observation = reward_source.observe(candidate, t=step)
            accepted = optimizer.observe(candidate, observation)
            weights = softmax_weights(candidate)
            poodle_weight = float(weights[target_index])
            last_reward = float(observation.reward_mean)
            state = "settle" if poodle_weight >= args.settle_weight else "explore"
            if poodle_weight >= args.settle_weight and last_reward > 0.55:
                settle_streak += 1
            else:
                settle_streak = 0

            print(
                f"step={step:03d} accepted={str(accepted):5s} "
                f"reward={last_reward:+.3f} var={observation.reward_variance:.4f} "
                f"poodle={poodle_weight:.2f} turbo_length={optimizer.length:.3f} "
                f"top={top_breeds(args.breeds, weights)}"
            )

            if settle_streak >= args.settle_patience:
                state = "settle"
                print(
                    f"Settled on {args.target_breed}: poodle weight stayed above "
                    f"{args.settle_weight:.2f} for {args.settle_patience} steps."
                )
                break

            candidate = optimizer.propose()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if last_display is not None:
            cv2.imshow(window_name, last_display)
            cv2.waitKey(400)
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
