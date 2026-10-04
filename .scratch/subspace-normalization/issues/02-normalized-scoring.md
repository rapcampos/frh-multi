# 02 — Normalized subspace scoring

Status: open
Type: task
Blocked by: 01

## Change

`subspace/generation.py::subspace_alignment` currently returns `‖Bᵀh‖`. Add a
normalized mode returning `‖Bᵀh‖ / ‖h‖` — the fraction of hidden-state energy in
the subspace, bounded in [0, 1] and directly comparable to the frame pipeline's
bounded correlation.

Three independent justifications, only the first of which depends on issue `01`:

1. Removes the reward for large-norm hidden states — the suspected truncation
   mechanism.
2. Makes the steering objective agree with the reported metric, which is cosine.
3. Puts the alignment term on a bounded scale so λ trades against `cum_logprob`
   comparably across methods.

## Requirements

- Opt-in parameter; the default must reproduce the existing unnormalized score
  bit-for-bit so `resources/68_comparison*` stays reproducible.
- Thread the option through `SubspaceUnembeddingRepresentation.generate_with_topk_guide`
  and `ModelBackend.generate`, recorded in the manifest settings so runs are
  self-describing.
- Do not touch the frame path. `_generate_with_topk_guide` is the original paper
  function and must not change behavior.
- Guard against `‖h‖ = 0` (degenerate padding positions).

## Tests

`subspace/generation.py` is deliberately model-free so selection logic is unit
testable on CPU — extend `tests/test_subspace.py` in that style:

- default path is numerically identical to the current implementation
- normalized score is bounded in [0, 1] and is scale-invariant in `h`
  (scaling `h` by any positive constant leaves it unchanged)
- a basis spanning `h` gives 1; a basis orthogonal to `h` gives 0
- repulsion composes with normalization without changing the default path

## Validation

New numbered notebook reusing the NB68 manifest, comparing normalized against
unnormalized on the truncation rate and length-controlled perplexity. Do not
edit NB68 and do not reuse its output prefix.

## Answer

**Implemented and tested. It does not fix truncation — it turns steering off.**

Shipped: `normalize` on `subspace_alignment`, `normalize_score` on
`SubspaceUnembeddingRepresentation.generate_with_topk_guide`, and
`make_experiment(..., normalize_score=True)`, which injects
`settings["subspace_normalize_score"]` only when enabled so manifests frozen
before the option existed still validate. 9 new tests in
`tests/test_subspace_normalization.py`; 217 pass; ruff and black clean.

### At lambda = 0 it is an exact no-op

Raw and normalized produce byte-identical output — same lengths, same lossy
steps, all 8 prompts. Two reasons compound:

- All `k` candidates in a set share a prefix, so the cumulative sum differs only
  at the last position and selection reduces to that position's score.
- `‖h‖` is nearly constant across a candidate set: CV 1.5% against the raw
  score's 28.5%, roughly 20x less spread. Dividing by a near-constant preserves
  the argmax in 97.5% of 1,536 candidate sets.

Llama's final hidden states carry massive-activation dimensions that make norms
uniform across tokens, so there is almost no variance for normalization to remove.

### At lambda > 0 it only rescales the trade-off

Raw alignment runs ~7.9 per position, normalized ~0.055 — a ~144x rescale of the
steering term against an unchanged fluency term. The same lambda therefore buys
~144x more fluency pressure.

| scoring | lambda | mean words | short-run rate | words shared with baseline | identical to baseline |
|---|---|---|---|---|---|
| raw | 0.0 | 7.8 | 0.62 | 0.2 | 0.12 |
| normalized | 0.0 | 7.8 | 0.62 | 0.2 | 0.12 |
| raw | 0.4 | 7.4 | 0.75 | 0.4 | 0.12 |
| normalized | 0.4 | **17.6** | **0.12** | **12.1** | **0.50** |

The truncation improvement is real but hollow: at lambda=0.4 the normalized score
shares 12.1 leading words with the unsteered greedy baseline and **half its
outputs are byte-identical to it**. Truncation disappears because steering
stopped happening, not because the defect was addressed.

Restoring the original steering strength needs lambda ~ 0.4/144 ~ 0.003, at which
truncation should return. The round-trip mechanism in issue `04` is untouched by
this change.

### What the change is still worth

Two of the three original justifications hold: the objective now matches the
cosine metric the battery reports, and the alignment term is bounded so lambda is
comparable across representations. The truncation rationale is withdrawn —
empirically now, not just theoretically. Keep it as an option, re-calibrate
lambda before using it, and do not treat it as the fix.

Promotes issue `04` from backlog to the actual fix.
