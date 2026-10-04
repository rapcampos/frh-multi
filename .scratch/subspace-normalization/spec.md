# Subspace steering: score normalization and rank budget

Status: in progress

## Motivation

NB68 (full battery, 4,736 continuations, 2026-09-17) compared matched FRH frame
blending against subspace steering. Subspace moves concept cosine 2-3x more than
frame blending but looked catastrophically less fluent: mean steered perplexity
184.5 vs 32.6 at λ=0.4, with a p90 of 660 against a median of 43.6.

That fluency gap is mostly an artifact. Splitting the λ=0.4 subspace runs by
length gives median PPL 659.7 for continuations of <=8 words (n=216) and 37.3 for
longer ones (n=820); `corr(words, log PPL) = -0.729`. On comparable-length output
subspace costs ~37 against frame blending's ~26 — 1.4x, not 5.7x.

The real defect is that subspace steering **stops generating early**:

| | frh_blend | subspace |
|---|---|---|
| continuation tokens (mean ± sd) | 24.9 ± 1.6 | 20.8 ± 7.3 |
| short-run rate (<=8 words) | 0.4-2.8% | 23-31% |

Truncation worsens monotonically with unit count (0.231 → 0.277 → 0.297 → 0.312)
while frame blending improves (0.028 → 0.004).

## Hypothesis

`subspace/generation.py::subspace_alignment` scores a candidate by `‖Bᵀh‖`, an
unnormalized projection magnitude of the raw hidden state. The frame pipeline's
score is a frame correlation bounded by 1. Three consequences:

1. The objective is unbounded and rewards large-norm hidden states regardless of
   direction. If EOT-adjacent hidden states carry distinctive norms, the score
   prefers them — the suspected truncation mechanism.
2. The steering objective does not match the reported metric, which is cosine.
3. λ cannot mean the same thing across methods, since it trades against scores on
   incomparable scales. NB68 acknowledges this in prose but does not correct it.

Proposed fix: score `‖Bᵀh‖ / ‖h‖` — the fraction of hidden-state energy in the
subspace, bounded in [0, 1]. Addresses all three at once.

**Point 1 is a hypothesis and is not yet tested.** Issue 01 tests it before any
code changes; if EOT candidates show no raw-score advantage, the truncation
mechanism is something else and the normalization rationale narrows to points 2
and 3.

## Scope

- `01` — EOT probe: validate or refute the mechanism. Gates `02`.
- `02` — normalized scoring in `subspace_alignment`, behind an explicit option so
  the existing unnormalized path stays reproducible.
- `03` — rank budget: `subspace_rank` is fixed at 8 (`steering_comparison.py:42`)
  whether blending 1 or 4 concepts, so at 4 units each concept gets ~2
  directions. Sweep it.

Deferred, not in scope here: the repulsion term `‖B_aᵀh‖ - ‖B_rᵀh‖` subtracts
unnormalized magnitudes from possibly overlapping subspaces, and counterfactuals
sit at chance for both methods (0.066 subspace, 0.079 frame). Tracked separately.

## Constraints

- The canonical frame function `_generate_with_topk_guide` must not change
  behavior. `subspace/` mirrors it but is a separate module; leave the frame path
  untouched.
- Experiments extend via a new numbered notebook reusing prior JSONL rows, never
  by editing NB68.
- `resources/68_comparison*` artifacts are final and must not be overwritten;
  `freeze_json` will refuse a changed manifest under the same prefix anyway.

## Known data caveat

NB68 prompt `p07` ("Continue this thought." / prefix `Here is what happened
next:`) has an empty greedy baseline, invalidating 592 of 4,736 runs (12.5%).
All NB68 numbers above are computed on the 4,144 valid runs. Any new battery
should fix or drop that prompt.
