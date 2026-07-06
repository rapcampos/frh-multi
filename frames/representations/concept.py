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
    def average(cls, concepts: list[Concept]) -> Concept:
        """Compute the Fréchet mean of a list of concepts on the Stiefel manifold.

        Equivalent to solve_procrustes(F1 + F2 + ... + Fn): finds the single
        frame geometrically closest to all inputs. All tensors must share the
        same shape (n, d, k).
        """
        summed = torch.stack([c.tensor for c in concepts]).sum(0)
        return cls(
            synset=" | ".join(c.synset for c in concepts),
            tensor=solve_procrustes(summed),
        )

    def _find_synset_index(self, synset: str) -> int:
        """Find the index of a synset in the dataframe."""
        if isinstance(self.synset, str):
            raise ValueError("Concept has only one synset.")
        return self.synset.index(synset)
