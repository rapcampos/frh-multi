"""Ordered subspaces with basis-invariant, FRH-aligned window scoring.

Positions label separate Grassmann factors; basis columns within a factor have
no prescribed orientation. This is a positional adaptation of product-subspace
representations, not tensor-mode unfolding or word-boundary alignment.
"""

from __future__ import annotations

import numpy as np
import torch
from pydantic import Field, model_validator

from frames.abstract import BaseModel

from .concept import blend_bases, subspace_from_features


def positional_token_ids(stacked_arrays, max_token_count: int, pad_id: int | None):
    """Extract columns without compacting missing positions; exclude long words."""
    if max_token_count < 1:
        raise ValueError("max_token_count must be positive")
    columns = [[] for _ in range(max_token_count)]
    for stacked in stacked_arrays:
        for row in np.atleast_2d(np.asarray(stacked)):
            present = (
                np.ones(row.shape, dtype=bool) if pad_id is None else row != pad_id
            )
            # A non-padding token outside the requested width must not be truncated.
            if present.sum() > max_token_count or present[max_token_count:].any():
                continue
            for j in range(min(len(row), max_token_count)):
                if present[j]:
                    columns[j].append(int(row[j]))
    return [np.asarray(ids, dtype=np.int64) for ids in columns]


def _window_projections(hidden: torch.Tensor, bases: torch.Tensor, normalize: bool):
    if hidden.ndim < 2 or hidden.shape[-1] != bases.shape[-2]:
        raise ValueError("hidden states must have shape (..., sequence, d)")
    hs = hidden.float()
    width = bases.shape[0]
    if hs.shape[-2] == 0:
        return hs.new_empty(*hs.shape[:-1], width)
    windows = (
        torch.nn.functional.pad(hs, (0, 0, 0, width - 1))
        .unfold(-2, width, 1)
        .movedim(-1, -2)
    )  # (..., L, T, d)
    coordinates = torch.einsum(
        "...tjd,jdr->...tjr", windows, bases.to(hs.device).float()
    )
    magnitudes = coordinates.norm(dim=-1)
    if normalize:
        magnitudes = magnitudes / windows.norm(dim=-1).clamp_min(
            torch.finfo(hs.dtype).tiny
        )
    return magnitudes


class PositionalConceptSubspace(BaseModel, arbitrary_types_allowed=True):
    """Ordered bases of shape (T, d, rank_cap), with zero rank padding.

    Metadata records source construction; blended concepts retain each input's
    complete metadata. ``rank`` is the storage cap, not an effective rank.
    """

    synset: str | list[str]
    bases: torch.Tensor
    sample_counts: list[int]
    effective_ranks: list[int] | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self):
        if self.bases.ndim != 3 or min(self.bases.shape) < 1:
            raise ValueError("bases must have positive shape (T, d, rank_cap)")
        if not torch.isfinite(self.bases).all():
            raise ValueError("bases must be finite")
        ranks = (self.bases.norm(dim=-2) > 0).sum(-1).tolist()
        if not any(ranks):
            raise ValueError("a positional concept must have an active position")
        if self.effective_ranks is not None and self.effective_ranks != ranks:
            raise ValueError("effective_ranks must match nonzero basis columns")
        if len(self.sample_counts) != len(self) or any(
            n < 0 for n in self.sample_counts
        ):
            raise ValueError(
                "sample_counts must contain one nonnegative count per position"
            )
        self.effective_ranks = ranks
        return self

    @classmethod
    def from_features(cls, synset, features, rank: int = 8, rtol: float = 1e-3):
        """Build from one (N_j, d) feature matrix per token position."""
        if rank < 1 or not features:
            raise ValueError("positive rank and at least one position are required")
        if rtol < 0:
            raise ValueError("rtol must be nonnegative")
        bases = torch.stack([subspace_from_features(x, rank, rtol) for x in features])
        return cls(
            synset=synset,
            bases=bases,
            sample_counts=[len(x) for x in features],
            metadata={"method": "positional", "rtol": rtol},
        )

    def __len__(self):
        return self.bases.shape[0]

    @property
    def rank(self):
        return self.bases.shape[-1]

    @property
    def device(self):
        return self.bases.device

    @property
    def name(self):
        return self.synset

    def construction_metadata(self):
        return {
            "synset": self.synset,
            "max_token_count": len(self),
            "rank_cap": self.rank,
            "effective_ranks": list(self.effective_ranks),
            "sample_counts": list(self.sample_counts),
            "metadata": self.metadata,
        }

    def _active(self, device):
        return torch.tensor(self.effective_ranks, device=device) > 0

    def project(self, hidden: torch.Tensor, normalize: bool = False):
        """Mean projection magnitude per forward window, including partial tails.

        The active-position denominator is fixed even at the right boundary.
        Normalized magnitude is the square root of captured energy fraction.
        """
        values = _window_projections(hidden, self.bases, normalize)
        return values[..., self._active(values.device)].mean(-1)

    def expression(self, hidden: torch.Tensor, valid_mask: torch.Tensor | None = None):
        """Mean captured energy over real, active hidden-state/window pairs.

        A mask excludes tokenizer padding, including padded window starts. Zero
        real hidden states still count as observations with zero energy.
        """
        values = _window_projections(hidden, self.bases, True).square()
        if valid_mask is None:
            valid_mask = torch.ones(
                hidden.shape[:-1], dtype=torch.bool, device=hidden.device
            )
        if valid_mask.shape != hidden.shape[:-1]:
            raise ValueError("valid_mask must match hidden sequence dimensions")
        if hidden.shape[-2] == 0:
            return values.new_zeros(hidden.shape[:-2])
        mask = valid_mask.to(device=values.device, dtype=torch.bool)
        pairs = torch.nn.functional.pad(mask, (0, len(self) - 1), value=False).unfold(
            -1, len(self), 1
        )
        pairs = pairs & mask.unsqueeze(-1) & self._active(values.device)
        return (values * pairs).sum((-2, -1)) / pairs.sum((-2, -1)).clamp_min(1)

    @classmethod
    def blend(cls, concepts, rank: int | None = None):
        if not concepts or not all(isinstance(c, cls) for c in concepts):
            raise ValueError("blend requires positional concepts only")
        shape = concepts[0].bases.shape[:2]
        if any(c.bases.shape[:2] != shape for c in concepts):
            raise ValueError(
                "positional concepts must share width and feature dimension"
            )
        rank = max(c.rank for c in concepts) if rank is None else rank
        if rank < 1:
            raise ValueError("rank must be positive")
        bases = torch.stack(
            [blend_bases([c.bases[j] for c in concepts], rank) for j in range(shape[0])]
        )
        names = [
            n
            for c in concepts
            for n in (c.synset if isinstance(c.synset, list) else [c.synset])
        ]
        return cls(
            synset=names,
            bases=bases,
            sample_counts=[
                sum(c.sample_counts[j] for c in concepts) for j in range(shape[0])
            ],
            metadata={
                "method": "positional_blend",
                "constituents": [c.construction_metadata() for c in concepts],
            },
        )


def positional_alignment(hidden, guide, repel_guide=None, normalize=False):
    """Signed positional window score, with separate active-position averages."""
    if not isinstance(guide, PositionalConceptSubspace):
        raise ValueError("guide must be positional")
    if repel_guide is not None:
        if not isinstance(repel_guide, PositionalConceptSubspace):
            raise ValueError("cannot mix pooled and positional guides")
        if guide.bases.shape[:2] != repel_guide.bases.shape[:2]:
            raise ValueError(
                "attraction and repulsion must share width and feature dimension"
            )
    scores = guide.project(hidden, normalize=normalize)
    if repel_guide is not None:
        scores = scores - repel_guide.project(hidden, normalize=normalize)
    return scores
