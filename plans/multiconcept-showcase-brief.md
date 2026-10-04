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

1. **Frame blending**: average the concept frames into one composite frame,
   steer as usual (logged as `F1.a-mean`).
2. **Score voting**: score each candidate word against each concept,
   normalize per step, sum the votes (logged as `F2.a-wsum`).
3. **Concept scheduling**: one concept at a time, alternating per step
   (logged as `F3.a-roundrobin`).

## How they perform (pooled over 30 pairs)

| method | Δexpr concept A | Δexpr concept B | joint = min(A, B) | both improved | PPL cost |
|---|---|---|---|---|---|
| single-concept guidance (reference) | **0.75** | — | — | — | 6.7× |
| **frame blending** | 0.56 | 0.59 | **0.35** | **55%** | 6.7× |
| concept scheduling | 0.52 | 0.54 | 0.32 | 51% | 6.6× |
| score voting | 0.43 | 0.47 | 0.21 | 51% | 7.1× |

(Δexpr = how much more the generated text expresses the concept than the
unsteered model's output, measured by projection onto the concept frame.)

Three takeaways:

- **Multi-concept steering works.** All three methods raise expression of BOTH
  concepts simultaneously on average, and on ~half the pairs both concepts
  improve at once — across unrelated, antonym, and near-synonym pairs alike.
- **The second concept is nearly free.** Steering two concepts costs the same
  fluency as steering one (6.6–7.1× PPL in every row), and each concept still
  retains 60–80% of the expression it would get if steered alone (blending:
  78%, scheduling: 71%, voting: 61%).
- **Frame blending is the best default** — the simplest method (just
  average the frames) wins on joint expression, replicating our full
  experiments (E0) on 10× more pairs. Score voting is consistently
  last. Scheduling is a niche player: it leads on the antonym pairs
  (0.33 vs 0.28, n=6; also on the single "related" pair) — consistent with
  alternating topics achieving at passage level what averaging cannot in one
  word, though the per-regime samples are small.

**Cross-model check (notebook 25).** The same 30-pair sweep on
`gemma-2-2b-it` (concepts rebuilt for its tokenizer, English-only lemmas,
`min_lemmas=4` per the paper's model-comparison recipe) reproduces the method
ranking exactly — **blending > scheduling > voting on both model families** —
with both-concepts-improved rates of ~99% on Gemma (vs ~50% on Llama; raw
Δexpression magnitudes are not comparable across models, so rankings and
rates are the cross-model currency). Multi-concept steering is not a
Llama-specific phenomenon, and it works at 2B scale.

**Measurement caveat.** Δexpression is computed against a per-run unsteered
baseline; one baseline continuation ("What women can be?") flipped between two
otherwise byte-identical runs on different GPUs, shifting all pooled absolute
levels by ~0.3 while leaving every between-method gap unchanged (e.g.
blend−vote: 0.134 vs 0.137). Method comparisons are robust; absolute joint
values should be quoted per run, or baselines pinned to disk (open item).

Example (steer toward *sadness* + *music*, prompt "Tell me a short story."):

> *frame blending:* "Once, a long-finger pianoforté was left behind by its
> family in an antique shop. For years, the shop …"

## Paths forward (each already piloted)

1. **Beam search** — replacing the greedy candidate loop with true beam search
   was the single largest improvement in our full sweep (E1): it flips joint
   expression from negative to positive in every ρ band at no extra fluency
   cost. First candidate for the default configuration.
2. **ρ-adaptive blending** — for dissimilar concept pairs (ρ < 0.30),
   rotation-aligning frames before blending ("aligned blend") roughly doubles
   expression (E2/E2b); the selection rule is computable before generating
   anything.
3. **Negative steering** — the same machinery suppresses an unwanted concept
   robustly (E3); balancing suppression against expression of the wanted
   concept needs a weight sweep.
4. **Fluency control** — a fluency-aware candidate-selection dial (E1b) to
   bring the 6–7× perplexity cost down to a chosen budget.
5. **Model generality** — replicate on Gemma 2 9B (all machinery is
   model-agnostic).

Artifacts: `resources/24_showcase_summary.csv`,
`resources/24_showcase_joint_vs_rho.png`, raw `resources/24_showcase.jsonl`.
