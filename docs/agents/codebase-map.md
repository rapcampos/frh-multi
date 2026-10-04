# FRH multi-concept codebase map

## What this repository is

The repository begins with the paper implementation of the Frame Representation
Hypothesis (FRH) and extends its single-concept Top-k Concept-Guided Decoding (CGD)
into a master's-thesis research program on multi-concept composition. The invariant is
the language model: steering does not modify model weights, logits, or activations. It
reranks the model's own top-k next-token proposals using geometry in the whitened
unembedding/hidden-state spaces.

The research conclusion is deliberately narrower than the original roadmap: frame
composition is the strongest greedy-tier baseline; gauge alignment helps distant
concepts under weak search; true beam search largely removes family differences; and a
fluency term gives the practical expression/quality tradeoff. The earlier E-series found
true beam search over the extrinsic mean frame with a fluency weight around 0.4 to be
the strongest recorded recipe. The current thesis policy in `CONTEXT.md`, however,
keeps reported thesis results in the greedy tier and labels beam-tier results as future
work. Preserve that distinction when writing claims; neither result is a hard-coded API
default.

## Runtime data flow

1. `LanguageHuggingFaceModel` loads a causal LM and tokenizer, applying family-specific
   token fixes and the requested quantization.
2. `LinearUnembeddingRepresentation` takes `lm_head.weight`, computes its covariance,
   whitens it with the symmetric inverse square root, removes a mean or pad reference,
   and optionally normalizes token representations.
3. `MultiLingualWordNetSynsets` expands OMW lemmas across language and whitespace
   variants, tokenizes them, filters by token/lemma counts, and caches the table.
4. `FrameUnembeddingRepresentation` turns each lemma's ordered token sequence into a
   word frame and takes a Procrustes/polar mean over the padded lemma frames to obtain
   a `Concept`.
5. A CGD loop expands the current sequence(s) with the model's top-k proposals, forms
   sliding windows of last-layer hidden states, scores them against one or more
   concepts, and applies a survivor rule.
6. `EvaluationHarness` compares steered and unsteered continuations, calculates
   presence/expression/angle metrics and continuation perplexity, and appends JSONL.

## Core modules

### Linear algebra

- `frames/linalg/frame.py`: `Frame`, ordered trace correlation, diagonal similarity,
  unequal-rank `rho`, Procrustes subtraction, and relative rank.
- `frames/linalg/orthogonalization.py`: Gram-Schmidt and the SVD polar factor used as
  the project's Procrustes projection.
- `frames/linalg/stiefel.py`: canonical-metric exponential/logarithm, Karcher mean,
  and generalized-Procrustes aligned mean.
- `frames/linalg/angles.py`: gauge-invariant principal angles, Grassmann distance,
  mean canonical correlation, vector-to-subspace cosine, and text dominant subspace.
- `frames/linalg/gds.py`: Fukui-style common/difference spectral split, removal of
  common components, GDS/common blends, and the classical two-subspace difference.

### Representations and decoding

- `frames/representations/unembedding.py`: whitening and centering of the unembedding
  matrix.
- `frames/representations/concept.py`: synset-bearing `Concept`; extrinsic, aligned,
  and Fréchet averages; union-subspace construction.
- `frames/representations/aggregators.py`: z-score/rank normalization, weighted sum,
  soft minimum, hard constrained, and soft-constrained score aggregation.
- `frames/representations/schedulers.py`: round-robin, seeded stochastic, and
  sentence-boundary active-concept schedules.
- `frames/representations/frame.py`: concept construction/caching, probing, original
  greedy CGD, true beam CGD, fluency-aware selection, and convenience wrappers.

### Models, language data, and evaluation

- `frames/models/hf/`: model loading, AWQ/bitsandbytes configuration, tokenization,
  decoding, hidden states, and unembedding access.
- `frames/nlp/synsets.py`: OMW access and tokenized-lemma cache. This imports RAPIDS
  `cudf`, so model/data paths assume the CUDA-oriented environment.
- `frames/evaluation/harness.py`: experiment records and metric versions.
- `frames/experiments/notebook_helpers.py`: stable helpers extracted for new
  notebooks. Existing notebooks retain inline code as run provenance.

## Survivor rules and public entry points

`generate_with_topk_guide` is the paper-compatible single-concept API. Internally,
`_generate_guided` evaluates k candidates but chooses one best parent at each step;
the next pool is only that parent's k children. This is greedy one-step lookahead.

`generate_with_topk_multi_guide` uses the same survivor rule with multiple per-concept
scores and a pluggable scorer. Family 1 convenience methods first build an average or
joint-subspace guide. Family 3 supplies a stateful scorer that selects one active
concept at a time.

`generate_with_topk_beam_guide` changes the survivor rule: every surviving parent
expands, all children compete by cumulative score, and the top-k sequences survive
regardless of parent. After the first step its forward pool is k squared per input, so
GPU memory scales quickly with k and batch size.

Both loops can rank with

```text
cumulative concept score + lambda * cumulative token log-probability
```

while returning the selected beam's concept-score trajectory as the probe.

## Experiment lineage

- 01-14: paper-era foundations. Notebook 14 and `tests/golden/` pin the legacy
  single-concept path. Notebook 12 references a removed cascade API and is not
  rerunnable unchanged.
- 15-23: the E-series establishes pair selection, expression metric, aggregation
  geometry, rho crossover, family/beam comparisons, fluency, and negative steering.
- 24-33: broad showcases and the transition from legacy/AWQ pools to the validated
  bf16 concept pool in notebook 32.
- 34-38: parked common/hierarchy branches and gauge-invariant metric/pool controls.
- 39-52: the current bf16 greedy-core, scaling, lambda, cross-model, lemma-policy,
  qualitative, and presentation analyses. `NOTEBOOKS.md` is authoritative for active,
  future-work, and archived status.

Artifacts remain under repository `resources/` even when their producing notebook is
archived, because later notebooks may consume them. Executed copies go under
`outputs/` or `archive/executed/` according to the provenance rules in
`archive/README.md`.

## Tests and invariants

CPU tests cover aggregators, angles, GDS, composition, Stiefel operations, schedulers,
beam selection, evaluation records, helper functions, and rho. The GPU golden test
requires model weights and suitable hardware.

When changing the decoding loop, explicitly check:

- `k=1` degeneracy and the `lambda=0` path;
- whether scores are per-position or cumulative;
- the candidate-pool axis used for normalization;
- parent/child indexing after flattening and reshaping;
- that probes report concept score even when selection includes fluency;
- batch independence and deterministic seeded schedules.

When changing geometry, explicitly check:

- whether the operation should preserve ordered gauge or only span;
- orthonormality after SVD/projection;
- zero-padded columns and effective rank;
- dtype promotion for SVD/eigendecomposition (fp16/bf16 usually compute in float32 or
  float64 here);
- identical, orthogonal, rotated-basis, and unequal-rank cases.

## Known operational traps

- The top-level worktree is intentionally busy. Do not use the nested `frh-multi/`
  checkout as the active source tree and do not clean either tree incidentally.
- `frames/experiments/figstep_cdg_comparison.py` imports `itertools.batched`, which is
  unavailable on Python 3.11; notebook 10 therefore needs Python 3.12 or a scoped
  compatibility change.
- CPU tests can import cuDF and emit a no-GPU warning; that is not itself a failure.
- A globally installed `pytest-rerunfailures` plugin may try to create a socket in a
  sandbox. `-p no:rerunfailures` avoids the unrelated failure.
- `uv` may try to write its global cache outside the workspace. Set
  `UV_CACHE_DIR=/tmp/frh-uv-cache` in sandboxed runs.
- Several historical plans referenced by old summaries may be deleted or moved in the
  current dirty worktree. Use `NOTEBOOKS.md`, `archive/README.md`, the active notebooks,
  and on-disk artifacts to resolve provenance.
- Model loading may log in to Hugging Face and download gated multi-gigabyte weights.
  Confirm the run is needed and the intended quantization/model is selected first.
