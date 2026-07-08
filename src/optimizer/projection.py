"""Reduce the search space from the raw latent/embedding dim down to 8-16
dims, per the spec: never hill-climb directly in a 1000-dim latent, you'll
never get enough samples. Two ways to build the low-dim basis:

  - PCAProjector: PCA over a bank of CLIP prompt embeddings.
  - AnchorInterpolationProjector: interpolation weights across a handful of
    hand-picked anchor prompt embeddings (simpler, more interpretable, no
    fitting required).

Both map a low-dim search vector z -> a full embedding the Generator can
condition on, and both bound z so the optimizer's [-bounds, bounds] box makes
sense in the projected space.
"""

from __future__ import annotations

import numpy as np


class PCAProjector:
    """Low-dim search vector <-> full embedding, via PCA fit on a prompt bank."""

    def __init__(self, dims: int):
        self.dims = dims
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "PCAProjector":
        """embeddings: [n_prompts, embed_dim]"""
        self.mean_ = embeddings.mean(axis=0)
        centered = embeddings - self.mean_
        # SVD-based PCA; robust for n_prompts << embed_dim (typical here).
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self.dims, vt.shape[0])
        self.components_ = vt[:k]
        # Scale so a unit step in z covers roughly one prompt-bank standard
        # deviation along that principal axis - keeps step sizes meaningful.
        self.scale_ = s[:k] / max(len(embeddings) - 1, 1) ** 0.5
        self.scale_ = np.where(self.scale_ > 1e-9, self.scale_, 1.0)
        if k < self.dims:
            pad = self.dims - k
            self.components_ = np.vstack([self.components_, np.zeros((pad, embeddings.shape[1]))])
            self.scale_ = np.concatenate([self.scale_, np.ones(pad)])
        return self

    def to_embedding(self, z: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "call fit() first"
        return self.mean_ + (z * self.scale_) @ self.components_

    def to_z(self, embedding: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "call fit() first"
        centered = embedding - self.mean_
        return (self.components_ @ centered) / self.scale_


class AnchorInterpolationProjector:
    """z is a weight vector over `anchor_embeddings`; softmax-normalized so
    the projected embedding always stays inside the anchors' convex hull.
    len(anchor_embeddings) becomes the search dimensionality.
    """

    def __init__(self, anchor_embeddings: np.ndarray):
        self.anchors = anchor_embeddings
        self.dims = anchor_embeddings.shape[0]

    def to_embedding(self, z: np.ndarray) -> np.ndarray:
        weights = np.exp(z - z.max())
        weights /= weights.sum()
        return weights @ self.anchors

    def to_z(self, embedding: np.ndarray) -> np.ndarray:
        # Least-squares fit of weights, then log so to_embedding's softmax
        # round-trips approximately - only used for seeding/debugging.
        weights, *_ = np.linalg.lstsq(self.anchors.T, embedding, rcond=None)
        weights = np.clip(weights, 1e-6, None)
        return np.log(weights)
