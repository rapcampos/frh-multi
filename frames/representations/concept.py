from __future__ import annotations

import torch

from ..linalg import Frame
from ..linalg.orthogonalization import solve_procrustes
from ..linalg.stiefel import aligned_mean, frechet_mean


class Concept(Frame):
    """A single or set of concept frames."""

    synset: str | list[str]

    def __getitem__(self, synset: str) -> Concept:
        """Get a single concept by its synset."""
        idx = self._find_synset_index(synset)
        return Concept(synset=synset, tensor=self.tensor[[idx]])

    def __sub__(self, other: Concept) -> Concept:
        """Subtract one concept from another."""
        return Concept(
            synset=" - ".join([self.synset, other.synset]),
            tensor=super().__sub__(other),
        )

    def __str__(self) -> str:
        """Get a string representation of the concept."""
        count, dimension, rank = self.tensor.shape
        if count == 1:
            name = self.synset
            return f"{self.__class__.__name__}({name=}, {dimension=}, {rank=})"
        return f"{self.__class__.__name__}({count=}, {dimension=}, {rank=})"

    @property
    def name(self) -> str:
        """Get the name of the concept."""
        return self.synset

    @classmethod
    def _pad_to_common_rank(cls, concepts: list[Concept]) -> list[torch.Tensor]:
        """Zero-pad frame tensors to a common k (the standard unequal-rank
        policy: padding is rank-neutral, since rank counts nonzero vectors)."""
        k = max(len(c) for c in concepts)
        return [torch.nn.functional.pad(c.tensor, (0, k - len(c))) for c in concepts]

    @classmethod
    def average(
        cls,
        concepts: list[Concept],
        weights: list[float] | None = None,
        method: str = "extrinsic",
        **method_kwargs,
    ) -> Concept:
        """F1.a — weighted mean of concepts. `method` selects the geometry:

        - "extrinsic" (default, the E0 baseline): solve_procrustes(sum_i
          w_i * C_i) — the orthonormal frame closest in chordal distance to
          the weighted sum. NOT the geodesic Fréchet mean (RQ3 compares).
        - "aligned": generalized-Procrustes mean — each frame is rotated
          within its span to best match the mean before chordal averaging;
          invariant to right-rotations of the inputs.
        - "frechet": geodesic Karcher mean on the Stiefel manifold
          (canonical metric), via exp/log fixed-point iteration.

        Frames of different ranks are zero-padded to a common k. For the
        extrinsic mean, trailing all-zero columns of the weighted sum are
        excluded from the polar decomposition — their factor would be
        arbitrary — and restored as zeros. The intrinsic methods operate on
        Stiefel points, where zero columns are not allowed: they require all
        concepts to share the same effective rank (computed, padded back).

        Weights default to uniform; for the extrinsic mean only ratios matter
        (the polar factor is scale-invariant) and negative weights repel
        (frame-space analogue of score-space negative steering). The
        intrinsic means are Fréchet functionals — weights must be
        nonnegative.
        """
        weights = weights if weights is not None else [1.0] * len(concepts)
        padded = cls._pad_to_common_rank(concepts)
        synset = " | ".join(c.synset for c in concepts)

        if method != "extrinsic":
            mean = cls._intrinsic_average(padded, weights, method, **method_kwargs)
            return cls(synset=synset, tensor=mean)

        summed = sum(w * t for w, t in zip(weights, padded))

        k = summed.size(-1)
        nonzero_cols = summed.abs().sum(dim=(0, -2)) > 0
        k_eff = int(torch.nonzero(nonzero_cols).max()) + 1 if nonzero_cols.any() else k
        mean = solve_procrustes(summed[..., :k_eff])
        mean = torch.nn.functional.pad(mean, (0, k - k_eff))

        return cls(synset=synset, tensor=mean)

    @classmethod
    def _intrinsic_average(
        cls,
        padded: list[torch.Tensor],
        weights: list[float],
        method: str,
        **method_kwargs,
    ) -> torch.Tensor:
        """Riemannian alternatives to the extrinsic mean (E2 / thesis RQ3).

        Zero-padded columns are not on the Stiefel manifold, so all concepts
        must share the same effective rank; the mean is computed on those
        columns and zero-padded back to the common k.
        """
        means = {"aligned": aligned_mean, "frechet": frechet_mean}
        if method not in means:
            raise ValueError(f"unknown average method {method!r}")

        k = padded[0].size(-1)
        ranks = []
        for tensor in padded:
            nonzero_cols = tensor.abs().sum(dim=(0, -2)) > 0
            ranks.append(
                int(torch.nonzero(nonzero_cols).max()) + 1 if nonzero_cols.any() else 0
            )
        k_eff = ranks[0]
        if k_eff == 0 or any(rank != k_eff for rank in ranks):
            raise ValueError(
                f"method={method!r} requires equal effective ranks, got {ranks}"
            )

        points = torch.stack([tensor[..., :k_eff] for tensor in padded])
        mean = means[method](points, weights, **method_kwargs)
        return torch.nn.functional.pad(mean, (0, k - k_eff))

    @classmethod
    def joint_subspace(cls, concepts: list[Concept], rtol: float = 1e-3) -> Concept:
        """F1.b — orthonormal basis of the union of the concepts' spans.

        Concatenates all frames' vectors and orthonormalizes via SVD,
        truncating directions with singular values below rtol * s_max
        (rank-deficient directions would otherwise be arbitrary).

        OR-like semantics: unlike `average`, a candidate aligned with ANY
        constituent scores well against the result — at the cost of a higher
        rank denominator in the correlation.
        """
        stacked = torch.cat([c.tensor for c in concepts], dim=-1)
        basis_full, singular_values, _ = torch.linalg.svd(
            stacked.float(), full_matrices=False
        )
        keep = singular_values > rtol * singular_values.amax(dim=-1, keepdim=True)
        basis = basis_full * keep.unsqueeze(-2)
        k_eff = int(keep.sum(-1).max())

        return cls(
            synset=" + ".join(c.synset for c in concepts),
            tensor=basis[..., :k_eff].to(stacked.dtype),
        )

    def _find_synset_index(self, synset: str) -> int:
        """Find the index of a synset in the dataframe."""
        if isinstance(self.synset, str):
            raise ValueError("Concept has only one synset.")
        return self.synset.index(synset)
