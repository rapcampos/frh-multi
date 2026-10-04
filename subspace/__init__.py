"""Subspace concept pipeline.

An alternative to the frame representation: a WordNet concept is a linear
subspace (top-``r`` SVD) of the pooled whitened+centered unembedding features of
every token across every lemma of the concept. Build → project → generate,
reusing the FRH model, unembedding, and WordNet loaders.

``SubspaceUnembeddingRepresentation`` is imported lazily so that the torch-only
math (``ConceptSubspace``, ``subspace_from_features``) can be used without
pulling in the heavier model/cuDF stack.
"""

from .concept import (
    ConceptSubspace,
    blend_bases,
    subspace_from_features,
    subspace_from_lemma_features,
)
from .positional import (
    PositionalConceptSubspace,
    positional_alignment,
    positional_token_ids,
)

__all__ = [
    "PositionalConceptSubspace",
    "positional_alignment",
    "positional_token_ids",
    "ConceptSubspace",
    "blend_bases",
    "subspace_from_features",
    "subspace_from_lemma_features",
    "SubspaceUnembeddingRepresentation",
]


def __getattr__(name: str):
    if name == "SubspaceUnembeddingRepresentation":
        from .representation import SubspaceUnembeddingRepresentation

        return SubspaceUnembeddingRepresentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
