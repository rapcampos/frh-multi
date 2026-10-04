# Subspace concept pipeline

An alternative to the frame representation. It keeps the FRH front end — a
WordNet concept's lemmas are tokenized and each token is mapped to a
whitened+centered unembedding feature (`W @ Σ^{-1/2}` minus the meaningless
vector) — but replaces the Stiefel-frame machinery with a plain **linear
subspace**.

## How a concept is built

1. Pool the whitened+centered feature vectors of **every token across every
   lemma** of the concept into a matrix `M ∈ R^{N×d}`.
2. Take the top-`r` principal directions of that token cloud via SVD (the right
   singular vectors span the token cloud).
3. The concept is the orthonormal basis `B ∈ R^{d×r}`.

Rank `r` is fixed for every concept (like the frame pipeline's `k`); if a
concept spans fewer than `r` significant directions, the extra basis columns are
zero.

### Difference from the frame pipeline

| | Frame | Subspace |
|---|---|---|
| Per-lemma step | orthonormalize tokens into a frame | — (tokens pooled directly) |
| Aggregation | Procrustes-average lemma frames | one SVD of the pooled token cloud |
| Concept | Stiefel frame `(d, k)` | orthonormal basis `(d, r)` |
| Alignment | frame `*` correlation (sliding window) | `‖Bᵀh‖` (single vector, no window) |

## Usage

```python
from subspace import SubspaceUnembeddingRepresentation

sur = SubspaceUnembeddingRepresentation.from_model_id(
    "google/gemma-2-2b-it", rank=8
)

# build a concept subspace (disk-cached under cache/subspaces/)
dog = sur.get_concept_cached("dog.n.01", min_lemmas_per_synset=3, max_token_count=4)

# project text onto the concept (per-position alignment ‖Bᵀh‖)
alignment = sur.project("The animal ran across the yard", dog)

# interpretable expression metric, mean energy fraction in [0, 1]
score = sur.expression("The animal ran across the yard", dog)

# subspace-guided generation
texts, probe = sur.generate_with_topk_guide(
    "Tell me a story about", guide=dog, k=4, steps=24
)

# blend several concepts into one guide (Family-1 composition) and steer toward
# all of them at once — a blend is a single subspace, so generation is unchanged
cat = sur.get_concept_cached("cat.n.01", min_lemmas_per_synset=3, max_token_count=4)
animals = ConceptSubspace.blend([dog, cat])
texts, probe = sur.generate_with_topk_guide("Tell me a story about", guide=animals)

# fluency dial: lambda >= 0 trades steering strength against the model's own
# fluency (cumulative alignment + lambda * cumulative token log-prob)
texts, probe = sur.generate_with_topk_guide(
    "Tell me a story about", guide=animals, k=4, steps=24, fluency_weight=0.4
)
```

## Layout

- `concept.py` — `ConceptSubspace` (incl. `ConceptSubspace.blend`),
  `subspace_from_features`, and `blend_bases` (torch-only math).
- `generation.py` — `subspace_alignment`, `select_best_candidate`, and
  `select_best_candidate_fluent` (torch-only, the guided-generation
  scoring/selection; greedy top-k lookahead, not beam search; the `_fluent`
  variant adds the `fluency_weight` dial).
- `representation.py` — `SubspaceUnembeddingRepresentation`: build / project /
  generate (single or blended guide, with optional `fluency_weight`), reusing
  the FRH model, unembedding, and WordNet loaders.

Math is unit-tested in `tests/test_subspace.py` (CPU-only). The model-in-the-loop
demo lives in `playground/subspace_pipeline.ipynb`.

## Positional concepts (opt-in)

`method="positional"` builds one subspace for each word-token position. For
`max_token_count=3`, the three bases describe first, second, and third tokens.
Each uses an independent float32 SVD of the existing whitened features, without
additional centering. Words longer than the limit are excluded; missing tokens
are omitted without shifting later positions. Repeated tokens retain their votes.

```python
from subspace import PositionalConceptSubspace

pos = sur.get_concept_cached("dog.n.01", 11, 3, method="positional")
# pos.bases: (3, d, sur.rank); rank is a per-position cap, not window width.
print(pos.sample_counts, pos.effective_ranks)
texts, probe = sur.generate_with_topk_guide(
    "Tell me a story about", pos, k=4, steps=24,
    normalize_score=False, fluency_weight=0.4,
)
```

Direct DataFrame construction requires the width explicitly:
`sur.compute_concept(df, method="positional", max_token_count=3)`.
`PositionalConceptSubspace.from_features(name, features, rank=8)` accepts a list
of `(N_j, d)` feature matrices for CPU use. Empty positions have zero bases;
zero columns do not count toward effective rank. A wholly inactive concept is
rejected. Positional caches use a separate `_positional_v1` name and versioned
payload; pooled and hierarchical caches retain their existing formats.

For each forward window starting at `t`, scoring pairs `h[t+j]` with basis
`B[j]`. Append `T-1` zero states, score every original start, and average projection
magnitudes over the concept's active positions. The denominator stays fixed at
partial final windows. `normalize_score=True` divides each magnitude by its
hidden-state norm: this is **normalized magnitude**, not squared energy. No word
boundary detection is performed. Candidate proposals, greedy one-step lookahead,
fluency weighting, and cumulative selection remain unchanged.

`sur.project(text, pos)` returns raw window scores. `sur.expression(text, pos)`
returns mean **squared** normalized projection over active, real window/state
pairs, excluding tokenizer padding and synthetic tail padding. Direct
`pos.expression(hidden, valid_mask)` accepts an explicit padding mask; an empty
set of valid pairs returns zero. This expression metric differs from the steering
score and from the pilot's shared pooled-reference evaluation.

Use `PositionalConceptSubspace.blend([a, b])` for positionwise projector blending.
The default cap is the largest input cap. Input widths and feature dimensions
must agree; constituent metadata is retained, and blended sample counts sum
source observations (they are not deduplicated word counts). Repulsion subtracts
separately averaged scores via `repel_guide=b`; negating a basis cannot repel.
Mixing positional and pooled guides is rejected.

### Geometry and order limitation

Independent orthogonal rotations within each retained basis leave projection
scores unchanged. Exchanging positional bases can change individual window
scores. However, summing additive scores across all overlapping windows compares
each interior hidden state with every active subspace. For sequence length `L`
and fixed active count `a`, the cumulative raw score is

```text
sum_j sum_{q=j}^{L-1} ||B[j].T h[q]|| / a.
```

Thus permutation differences in the cumulative score arise from the first
`T-1` states; right zero padding preserves all available contributions. The same
identity holds for normalized magnitudes. This baseline tests position-specific
construction, not a claim of order-sensitive generation. In particular, later
candidate states can receive an order-invariant aggregate score. The pilot
includes a reversed-position control and an explicit cancellation diagnostic.

The ordered tuple is inspired by product-Grassmann representations in
[Gatto et al., Tensor Analysis with n-Mode Generalized Difference Subspace](https://arxiv.org/abs/1909.01954).
Token positions here are slices, not tensor-mode unfoldings. GDS, tensor coupling,
and word-boundary matching are outside this implementation.

### Pilot

`subspace/notebooks/positional_subspace_pilot.ipynb` compares positional rank-eight
bases, a reversed-position control, pooled rank-eight and rank-24 bases, FRH, and
unsteered generation on the same eligible English WordNet words. It uses bf16
Llama-3.1-8B-Instruct, `MAX_TOKENS=3`, `k=4`, 24 steps, and lambda 0/0.4. Raw and
normalized subspace scores are separate arms. `SMOKE=True` selects one prompt and
one single-concept case; the full run adds the second concept/prompt, blending,
and both repulsion directions. FRH repulsion uses its differential frame, while
subspace repulsion subtracts scores.

The notebook requires CUDA and cached model weights; model downloads are disabled
in its kernel. It writes new timestamped, manifest-hashed directories under
`resources/positional_subspace_pilot_unquantized/`, including exact token rows,
construction ranks/counts, source hashes, settings, continuations, shared pooled
projection cosine, continuation-only PPL, and separate concept trajectories.
No model run is needed for the CPU mathematical tests.
