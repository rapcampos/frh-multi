# Multi-Concept Steering — Brief Report

One-page interim summary. Source: `24_showcase_multiconcept.ipynb` — 30
WordNet concept pairs (27 unique concepts, ρ 0.19–0.57, all semantic regimes),
Llama-3.1-8B-Instruct, k=4, 24 generated tokens, 10 shared prompts. Full
evaluation lives in E0–E3 (`plans/multiconcept-cgd-report.md`); this is the
broad-coverage demonstration.

## The methods

FRH's Top-k Concept-Guided Decoding steers generation toward ONE WordNet
concept frame. We extended it to several concepts at the three possible
intervention points:

1. **F1.a — frame space**: average the concept frames into one composite
   frame, steer as usual.
2. **F2.a — score space**: score each candidate word against each concept,
   normalize per step, sum.
3. **F3.a — time**: one concept at a time, alternating per step.

## How they perform (pooled over 30 pairs)

| method | Δexpr concept A | Δexpr concept B | joint = min(A, B) | both improved | PPL cost |
|---|---|---|---|---|---|
| FRH single concept (reference) | **0.75** | — | — | — | 6.7× |
| **F1.a mean frame** | 0.56 | 0.59 | **0.35** | **55%** | 6.7× |
| F3.a round-robin | 0.52 | 0.54 | 0.32 | 51% | 6.6× |
| F2.a score sum | 0.43 | 0.47 | 0.21 | 51% | 7.1× |

(Δexpr = how much more the generated text expresses the concept than the
unsteered model's output, measured by projection onto the concept frame.)

Three takeaways:

- **Multi-concept steering works.** All three methods raise expression of BOTH
  concepts simultaneously on average, and on ~half the pairs both concepts
  improve at once — across unrelated, antonym, and near-synonym pairs alike.
- **The second concept is nearly free.** Steering two concepts costs the same
  fluency as steering one (6.6–7.1× PPL in every row), and each concept still
  retains 60–80% of the expression it would get if steered alone (F1.a: 78%,
  F3.a: 71%, F2.a: 61%).
- **Frame-space composition is the best default** — the simplest method (just
  average the frames) wins on joint expression, replicating our full
  experiments (E0) on 10× more pairs. Score-space aggregation is consistently
  last. Scheduling is a niche player: it leads on the antonym pairs
  (0.33 vs 0.28, n=6; also on the single "related" pair) — consistent with
  alternating topics achieving at passage level what averaging cannot in one
  word, though the per-regime samples are small.

Example (steer toward *sadness* + *music*, prompt "Tell me a short story."):

> *F1.a:* "Once, a long-finger pianoforté was left behind by its family in an
> antique shop. For years, the shop …"

## Paths forward (each already piloted)

1. **Beam search** — replacing the greedy candidate loop with true beam search
   was the single largest improvement in our full sweep (E1): it flips joint
   expression from negative to positive in every ρ band at no extra fluency
   cost. First candidate for the default configuration.
2. **ρ-adaptive frame averaging** — for dissimilar concept pairs (ρ < 0.30),
   rotation-aligning frames before averaging roughly doubles expression (E2/
   E2b); the selection rule is computable before generating anything.
3. **Negative steering** — the same machinery suppresses an unwanted concept
   robustly (E3); balancing suppression against expression of the wanted
   concept needs a weight sweep.
4. **Fluency control** — a fluency-aware candidate-selection dial (E1b) to
   bring the 6–7× perplexity cost down to a chosen budget.
5. **Model generality** — replicate on Gemma 2 9B (all machinery is
   model-agnostic).

Artifacts: `resources/24_showcase_summary.csv`,
`resources/24_showcase_joint_vs_rho.png`, raw `resources/24_showcase.jsonl`.
