"""Subspace concept representation.

A concept is represented as a linear subspace of the unembedding feature space:
the top-``r`` principal directions (SVD) of the pooled whitened+centered token
features of every token across every lemma of the concept. This is the subspace
analogue of the frame pipeline's :class:`frames.representations.concept.Concept`,
but with no Stiefel/Procrustes machinery — just an orthonormal basis and plain
projection.

This module is deliberately torch-only (no model, no cuDF) so the math is unit
testable on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from frames.abstract import BaseModel


def lemma_token_rows(stacked, pad_id: int | None) -> list[np.ndarray]:
    """Split a synset's stacked token array into one id row per lemma.

    ``get_all_synsets`` returns one row per synset whose ``tokens`` cell is a
    single pad-filled ``(n_lemmas, max_tokens)`` array (every lemma stacked
    together). The hierarchical builder needs **one block per lemma**, so iterate
    the *rows* of that array here — dropping rows that are entirely padding —
    rather than treating the whole array as a single block (which silently
    collapses the concept to rank-1).

    Args:
        stacked: a ``(n_lemmas, max_tokens)`` (or ``(max_tokens,)``) token-id array.
        pad_id: the tokenizer pad id to strip; ``None`` keeps every id.

    Returns:
        One 1-D id array per non-empty lemma row.
    """
    rows = []
    for row in np.atleast_2d(np.asarray(stacked)):
        ids = row[row != pad_id] if pad_id is not None else row
        if ids.size:
            rows.append(ids)
    return rows


def subspace_from_features(
    features: torch.Tensor, rank: int, rtol: float = 1e-3
) -> torch.Tensor:
    """Fixed-rank orthonormal basis spanning a cloud of feature vectors.

    Args:
        features: ``(N, d)`` matrix whose rows are the token feature vectors of a
            concept (whitened + centered unembedding features).
        rank: the fixed subspace rank ``r``. The returned basis always has ``r``
            columns; if the token cloud spans fewer than ``r`` significant
            directions, the extra columns are zero (the fixed-rank policy, mirror
            of the frame pipeline's zero-padding to a common ``k``).
        rtol: directions whose singular value is ``<= rtol * s_max`` are treated
            as rank-deficient and zeroed, matching
            :meth:`Concept.joint_subspace`.

    Returns:
        ``(d, r)`` tensor with orthonormal columns (up to the kept rank), on the
        same device as ``features`` and in float32.
    """
    features = features.float()
    d = features.shape[-1]
    basis = torch.zeros(d, rank, dtype=features.dtype, device=features.device)

    if features.numel() == 0 or features.shape[0] == 0:
        return basis

    # Right singular vectors span the row space (= span of the token vectors).
    _, singular_values, vh = torch.linalg.svd(features, full_matrices=False)
    available = min(rank, vh.shape[0])
    top = vh[:available].mT  # (d, available)

    keep = singular_values[:available] > rtol * singular_values.amax()
    basis[:, :available] = top * keep.unsqueeze(0)
    return basis


def _principal_directions(features: torch.Tensor, k: int, rtol: float) -> torch.Tensor:
    """Top-``k`` right singular vectors of ``features`` (rows are vectors).

    Like :func:`subspace_from_features` but returns only the kept columns with
    **no** zero-padding: ``(d, kept)`` where ``kept <= k`` and directions whose
    singular value is ``<= rtol * s_max`` are dropped. Used as the per-lemma
    step of the hierarchical builder.
    """
    features = features.float()
    if features.numel() == 0 or features.shape[0] == 0:
        return features.new_zeros(features.shape[-1], 0)

    _, singular_values, vh = torch.linalg.svd(features, full_matrices=False)
    available = min(k, vh.shape[0])
    s = singular_values[:available]
    keep = s > rtol * s.amax()
    return vh[:available].mT[:, keep]  # (d, kept)


def subspace_from_lemma_features(
    lemma_features: list[torch.Tensor], rank: int, r_lemma: int = 1, rtol: float = 1e-3
) -> torch.Tensor:
    """Fixed-rank concept basis from a *set of per-lemma* feature blocks.

    The hierarchical ("subspace-of-subspaces") construction: each lemma is first
    reduced to its own ``r_lemma``-dimensional subspace, and the concept is the
    principal subspace of that *set* of lemma-subspaces. Equivalently, the
    top-``rank`` eigenvectors of ``Σᵢ BᵢBᵢᵀ`` (the sum of the lemma projectors) —
    which, since ``Σᵢ BᵢBᵢᵀ = B_all B_allᵀ`` for the stacked bases
    ``B_all = [B₁ | … | Bₘ]``, are exactly the top-``rank`` **left** singular
    vectors of ``B_all``.

    Contrast with :func:`subspace_from_features`, which pools *all* tokens and
    weights the concept one-vote-per-token; here each lemma casts an equal vote
    regardless of its token count. (Uniform lemma weighting for now; a weighted
    ``Σᵢ wᵢ BᵢBᵢᵀ`` variant — scale each ``Bᵢ`` by ``√wᵢ`` before stacking — is a
    natural future knob.)

    Args:
        lemma_features: one ``(nᵢ, d)`` block of whitened+centered token features
            per lemma. Empty/all-zero blocks are skipped. Must be non-empty and
            share the feature dimension ``d``.
        rank: fixed concept rank ``r``; the returned basis always has ``r``
            columns, zero-padded if the lemma set spans fewer than ``r``
            significant directions (mirrors :func:`subspace_from_features`).
        r_lemma: directions kept per lemma (capped at each lemma's token count).
        rtol: relative singular-value floor, applied at both the per-lemma and
            the combining SVD, matching :func:`subspace_from_features`.

    Returns:
        ``(d, rank)`` tensor with orthonormal columns (up to the kept rank), in
        float32 on the same device as the input.
    """
    if not lemma_features:
        raise ValueError("lemma_features must be non-empty to infer the dimension")

    d = lemma_features[0].shape[-1]
    device = lemma_features[0].device
    basis = torch.zeros(d, rank, dtype=torch.float32, device=device)

    per_lemma = [_principal_directions(x, r_lemma, rtol) for x in lemma_features]
    per_lemma = [b for b in per_lemma if b.shape[-1] > 0]
    if not per_lemma:
        return basis

    b_all = torch.cat(per_lemma, dim=-1)  # (d, Σ kept_i)
    # Left singular vectors of the stacked bases = eigenvectors of Σᵢ BᵢBᵢᵀ.
    u, singular_values, _ = torch.linalg.svd(b_all, full_matrices=False)
    available = min(rank, u.shape[-1])
    keep = singular_values[:available] > rtol * singular_values.amax()
    basis[:, :available] = u[:, :available] * keep.unsqueeze(0)
    return basis


def blend_bases(
    bases: list[torch.Tensor], rank: int, rtol: float = 1e-3
) -> torch.Tensor:
    """Fixed-rank basis blending several concept subspaces into one.

    Union-then-reduce (the subspace analogue of the frame pipeline's Family-1
    blend, ``Concept.average``): drop each basis's zero-padded columns, stack the
    orthonormal bases ``[B₁ | … | B_N]``, and take the top-``rank`` **left**
    singular vectors — i.e. the principal subspace of ``Σ_c B_c B_cᵀ`` (the sum
    of the per-concept projectors). Concepts vote one direction per kept basis
    column, so overlapping directions reinforce and disjoint ones coexist up to
    the rank budget. Uses the same combining SVD as
    :func:`subspace_from_lemma_features`.

    Args:
        bases: one ``(d, rᵢ)`` orthonormal basis per concept (trailing zero
            columns from the fixed-rank policy are ignored). Must be non-empty
            and share the feature dimension ``d``.
        rank: fixed blend rank ``r``; the returned basis always has ``r``
            columns, zero-padded if the union spans fewer than ``r`` significant
            directions.
        rtol: directions whose singular value is ``<= rtol * s_max`` are zeroed,
            matching :func:`subspace_from_features`.

    Returns:
        ``(d, rank)`` tensor with orthonormal columns (up to the kept rank), in
        float32 on the same device as the input.
    """
    if not bases:
        raise ValueError("bases must be non-empty to infer the dimension")

    d = bases[0].shape[-2]
    device = bases[0].device
    basis = torch.zeros(d, rank, dtype=torch.float32, device=device)

    # Move every basis to a common device before stacking: under a sharded
    # (``device_map="auto"``) model, concepts built at different times can live
    # on different shards, and ``torch.cat`` across devices raises.
    kept = [b[:, b.norm(dim=0) > 0].float().to(device) for b in bases]
    kept = [b for b in kept if b.shape[-1] > 0]
    if not kept:
        return basis

    b_all = torch.cat(kept, dim=-1)  # (d, Σ rᵢ)
    u, singular_values, _ = torch.linalg.svd(b_all, full_matrices=False)
    available = min(rank, u.shape[-1])
    keep = singular_values[:available] > rtol * singular_values.amax()
    basis[:, :available] = u[:, :available] * keep.unsqueeze(0)
    return basis


class ConceptSubspace(BaseModel, arbitrary_types_allowed=True):
    """A concept represented as an orthonormal subspace basis.

    Attributes:
        synset: WordNet synset name (or list of names).
        basis: orthonormal columns, shape ``(d, r)`` for a single concept.
    """

    synset: str | list[str]
    basis: torch.Tensor

    @classmethod
    def blend(
        cls, concepts: list["ConceptSubspace"], rank: int | None = None
    ) -> "ConceptSubspace":
        """Blend several concepts into one via :func:`blend_bases` (Family 1).

        The single guide that steers toward *all* of ``concepts`` at once; feed
        it to :meth:`SubspaceUnembeddingRepresentation.generate_with_topk_guide`
        unchanged. ``rank`` defaults to the largest input rank.
        """
        if not concepts or not all(isinstance(c, cls) for c in concepts):
            raise ValueError("blend requires non-empty pooled/hierarchical concepts")
        rank = rank if rank is not None else max(c.rank for c in concepts)
        basis = blend_bases([c.basis for c in concepts], rank)
        names: list[str] = []
        for c in concepts:
            names.extend(c.synset if isinstance(c.synset, list) else [c.synset])
        return cls(synset=names, basis=basis)

    @property
    def rank(self) -> int:
        """Number of subspace directions ``r`` (including zero-padded ones)."""
        return self.basis.shape[-1]

    def __len__(self) -> int:
        return self.rank

    @property
    def device(self) -> torch.device:
        return self.basis.device

    @property
    def name(self) -> str:
        return self.synset

    def coordinates(self, h: torch.Tensor) -> torch.Tensor:
        """Coordinates of ``h`` in the subspace: ``h @ basis`` → ``(..., r)``."""
        return h.float() @ self.basis.to(h.device).float()

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Projection magnitude ``‖Bᵀh‖`` onto the subspace → ``(...)``.

        The per-position alignment score. The constant ``√r`` normalisation of
        the frame correlation is omitted: rank is fixed, so it never changes
        which candidate wins.
        """
        return self.coordinates(h).norm(dim=-1)

    def energy_fraction(self, h: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Fraction of ``h``'s energy captured by the subspace ``∈ [0, 1]``.

        The interpretable expression metric.
        """
        captured = self.coordinates(h).pow(2).sum(-1)
        total = h.float().pow(2).sum(-1).clamp_min(eps)
        return captured / total

    def __str__(self) -> str:
        d, r = self.basis.shape[-2], self.basis.shape[-1]
        return f"{self.__class__.__name__}(synset={self.synset!r}, d={d}, rank={r})"
