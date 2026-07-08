"""CPU-only fallback renderer: a deterministic function of z, no GPU or model
weights required. This is what proves the fake-reward loop end-to-end (build
order step 1), and it's the safety net if real-time text-to-3D blows the
latency budget on demo day - the loop only cares that the morph is smooth, not
that the object is "true" 3D.

z's first few dimensions are mapped onto a star/polygon's hue, spike count,
spikiness, rotation and scale, so a human watching the morph can visually
confirm the search is converging on *something*.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw


def _dim(z: np.ndarray, i: int, default: float = 0.0) -> float:
    return float(z[i]) if i < len(z) else default


class ProceduralRenderer:
    def render(self, z: np.ndarray, size: int = 512) -> Image.Image:
        z = np.clip(z, -1.0, 1.0)
        hue = (_dim(z, 0) + 1) / 2
        spikes = int(round(3 + (_dim(z, 1) + 1) / 2 * 9))  # 3..12
        spike_ratio = 0.3 + (_dim(z, 2) + 1) / 2 * 0.6  # 0.3..0.9
        rotation = _dim(z, 3) * math.pi
        scale = 0.35 + (_dim(z, 4) + 1) / 2 * 0.35  # 0.35..0.70 of canvas

        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        color = (int(r * 255), int(g * 255), int(b * 255))

        img = Image.new("RGB", (size, size), (10, 10, 16))
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2
        outer_r = size / 2 * scale
        inner_r = outer_r * spike_ratio

        points = []
        n_points = spikes * 2
        for i in range(n_points):
            angle = rotation + i * math.pi / spikes
            radius = outer_r if i % 2 == 0 else inner_r
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

        draw.polygon(points, fill=color)
        return img
