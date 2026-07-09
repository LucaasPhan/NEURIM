"""SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in
~100-300ms. Everything torch/diffusers is lazy-imported so the rest of the
codebase runs fine on a machine with no GPU - see procedural.py for what
runs instead in that case.
"""

from __future__ import annotations

import numpy as np


class DiffusionGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        num_inference_steps: int = 2,
        device: str | None = None,
    ):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = device or self._detect_device(torch)
        self.num_inference_steps = num_inference_steps
        self._torch = torch

        if self.device == "cuda":
            # fp16 + the "fp16" weight variant is the fast path CUDA supports.
            self.dtype = torch.float16
            load_kwargs = {"torch_dtype": self.dtype, "variant": "fp16"}
        else:
            # fp16 has a known black-image bug on MPS (Apple Silicon), and
            # CPU has no native fp16 compute either - fp32 on both, slower
            # but actually produces correct images.
            self.dtype = torch.float32
            load_kwargs = {"torch_dtype": self.dtype}

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs).to(self.device)
        self._prev_image = None

    @staticmethod
    def _detect_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def render_from_prompt(self, prompt: str):
        """Straight text -> image, bypassing the embedding/projector path -
        for sanity-checking the model/hardware before wiring up the
        anchor-prompt projector that the live optimizer loop uses.
        """
        image = self.pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
        ).images[0]
        self._prev_image = image
        return image

    def encode_prompts(self, prompts: list[str]) -> np.ndarray:
        """Run each anchor prompt through the pipeline's text encoder(s) and
        return stacked embeddings, for PCAProjector.fit() / anchor projection.
        """
        embeddings = []
        for prompt in prompts:
            with self._torch.no_grad():
                embeds = self.pipe.encode_prompt(
                    prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                )
            prompt_embeds = embeds[0] if isinstance(embeds, tuple) else embeds
            embeddings.append(prompt_embeds.flatten().cpu().numpy())
        return np.stack(embeddings)

    def render(self, embedding: np.ndarray):
        """embedding: flattened prompt_embeds matching the pipeline's expected
        shape (reshape happens against the pipe's own text-encoder output
        shape, captured the first time encode_prompts() runs).
        """
        prompt_embeds = self._torch.tensor(embedding, dtype=self.dtype, device=self.device)
        prompt_embeds = prompt_embeds.reshape(1, -1, prompt_embeds.shape[-1])
        image = self.pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
        ).images[0]
        self._prev_image = image
        return image

    def render_smoothed(self, embedding: np.ndarray, strength: float = 0.2):
        """img2img pass against the previous frame, for smoother morphing
        between optimizer steps than independent from-scratch samples.
        """
        if self._prev_image is None:
            return self.render(embedding)
        from diffusers import AutoPipelineForImage2Image

        if not hasattr(self, "_img2img_pipe"):
            self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        prompt_embeds = self._torch.tensor(embedding, dtype=self.dtype, device=self.device)
        prompt_embeds = prompt_embeds.reshape(1, -1, prompt_embeds.shape[-1])
        image = self._img2img_pipe(
            image=self._prev_image,
            prompt_embeds=prompt_embeds,
            strength=strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
        ).images[0]
        self._prev_image = image
        return image
