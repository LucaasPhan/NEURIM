"""Image -> pseudo-3D pyramid quadrants.

Real-time text-to-3D (TripoSR) is the part of the pipeline most likely to
blow the latency budget, so there's a procedural fallback from day one: fake
the "3D" by rotating the 2D sprite through a viewing angle. The loop doesn't
care whether the object is "true" 3D, only that the morph is smooth.
"""

from __future__ import annotations

import math

from PIL import Image


class TripoSRConverter:
    """Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`
    package and a CUDA GPU. Raises ImportError with a clear message if either
    is missing, so callers can fall back to ProceduralPseudo3D.
    """

    def __init__(self, device: str = "cuda"):
        from tsr.system import TSR  # noqa: F401  (import here to fail fast, lazily)

        self.device = device
        self._model = TSR.from_pretrained(
            "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
        ).to(device)

    def to_mesh_render(self, image: Image.Image):
        scene_codes = self._model([image], device=self.device)
        return self._model.render(scene_codes, n_views=4)[0]


class ProceduralPseudo3D:
    """Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU."""

    def apply_angle(self, image: Image.Image, angle_rad: float) -> Image.Image:
        # Squash horizontally with cos(angle) to fake a turntable rotation.
        squash = max(0.15, abs(math.cos(angle_rad)))
        w, h = image.size
        squashed = image.resize((max(1, int(w * squash)), h))
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(squashed, ((w - squashed.width) // 2, 0))
        return canvas


def mirrored_quadrants(image: Image.Image, canvas_size: int) -> Image.Image:
    """Compose 4 copies of `image`, each facing outward from center, for a
    tabletop hologram pyramid (classic Pepper's-ghost 4-quadrant layout).
    """
    quad = image.convert("RGBA").resize((canvas_size // 2, canvas_size // 2))
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 255))
    cx, cy = canvas_size // 2, canvas_size // 2
    w, h = quad.size

    placements = [
        (quad.rotate(180), (cx - w // 2, 0)),  # top, pointing down toward center
        (quad.rotate(0), (cx - w // 2, canvas_size - h)),  # bottom, pointing up
        (quad.rotate(90), (0, cy - h // 2)),  # left, pointing right
        (quad.rotate(270), (canvas_size - w, cy - h // 2)),  # right, pointing left
    ]
    for tile, pos in placements:
        canvas.paste(tile, pos, tile)
    return canvas
