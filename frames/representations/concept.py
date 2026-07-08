from __future__ import annotations

import torch

from ..linalg import Frame
from ..linalg.orthogonalization import solve_procrustes


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
        cls, concepts: list[Concept], weights: list[float] | None = None
    ) -> Concept:
        """F1.a — weighted extrinsic (chordal/Procrustes) mean of concepts.

        Computes solve_procrustes(sum_i w_i * C_i): the orthonormal frame
        closest in chordal distance to the weighted sum. Note this is the
        EXTRINSIC mean, not the geodesic Fréchet mean on the Stiefel manifold
        (thesis RQ3 compares the two; Riemannian variants drop in here).

        Frames of different ranks are zero-padded to a common k. Trailing
        all-zero columns of the weighted sum are excluded from the polar
        decomposition — their factor would be arbitrary — and restored as
        zeros, keeping rank bookkeeping consistent.

        Weights default to uniform; the overall scale is irrelevant (the polar
        factor is scale-invariant), only ratios matter. Negative weights repel
        (frame-space analogue of score-space negative steering).
        """
        weights = weights if weights is not None else [1.0] * len(concepts)
        padded = cls._pad_to_common_rank(concepts)
        summed = sum(w * t for w, t in zip(weights, padded))

        k = summed.size(-1)
        nonzero_cols = summed.abs().sum(dim=(0, -2)) > 0
        k_eff = int(torch.nonzero(nonzero_cols).max()) + 1 if nonzero_cols.any() else k
        mean = solve_procrustes(summed[..., :k_eff])
        mean = torch.nn.functional.pad(mean, (0, k - k_eff))

        return cls(
            synset=" | ".join(c.synset for c in concepts),
            tensor=mean,
        )

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
