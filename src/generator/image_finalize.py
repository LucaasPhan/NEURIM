"""OpenAI image finalize pass.

The anchor-morph diffusion server produces its last frame by blending several
anchor latents/prompt-embeds together, so the final image often carries morph
artifacts - ghosting, doubled features, half-blended subjects. This pass feeds
that frame back through OpenAI's image *edit* API so it gets resolved into a
single clean image while keeping the composition the session converged on.

Kept deliberately close in shape to ``openai_image.py`` (lazy client, dotenv
load, injectable ``client`` for tests).
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

# {subject} is the user's original prompt; the rest steers OpenAI to clean up
# the morph blend without reinventing the scene the optimizer settled on.
FINALIZE_INSTRUCTION = (
    "Refine this image into a single clean, coherent picture of {subject}. "
    "Resolve any morphing, ghosting, doubled edges, or half-blended features "
    "into one clear, well-formed subject. Keep the existing composition, "
    "framing, colors, and lighting. Sharp, natural, realistic."
)


class ImageFinalizer:
    def __init__(
        self,
        model: str = "gpt-image-2",
        size: str = "1024x1024",
        quality: str = "low",
        frame_size: int = 512,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            from openai import OpenAI

            client = OpenAI()

        self.client = client
        self.model = model
        self.size = size
        self.quality = quality
        self.frame_size = frame_size

    def finalize(self, png_bytes: bytes, subject: str) -> bytes:
        """Return a cleaned PNG for ``png_bytes`` (the last morphed frame).

        Raises on API/decoding failure - callers decide whether to fall back to
        the raw frame.
        """
        instruction = FINALIZE_INSTRUCTION.format(subject=subject.strip() or "the subject")

        image_file = io.BytesIO(png_bytes)
        image_file.name = "frame.png"
        result = self.client.images.edit(
            model=self.model,
            image=image_file,
            prompt=instruction,
            size=self.size,
            quality=self.quality,
        )

        image_b64 = result.data[0].b64_json
        if not image_b64:
            raise RuntimeError("OpenAI image edit response did not include b64_json")

        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        if image.size != (self.frame_size, self.frame_size):
            image = image.resize((self.frame_size, self.frame_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
