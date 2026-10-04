# Experiment 53: Long-Generation Composition Results

- Run date: 2026-08-10
- Trajectory-analysis rerun: 2026-08-11 (corrected PPL v2)
- Status: complete (450/450 guided continuations)

## Executive conclusion

The short-horizon family ranking does not transfer unchanged to 200-step
generation. The result depends on the fluency weight:

1. **Scheduling is strongest under pure steering (`lambda = 0`).** It has the
   highest mean joint expression and joint token-to-subspace cosine and wins
   slightly more prompt-pair cells than blending on the gauge-invariant metric.
2. **Blending is strongest when the fluency term is enabled (`lambda = 0.4`).**
   It retains more steering than scheduling and wins 39 of 75 cells on joint
   token cosine.
3. **Normalized score voting is not competitive as a default long-generation
   method.** Its average joint expression is much lower, its perplexity is much
   higher, and its visible continuations are markedly shorter. It does,
   however, often move both concepts in the correct token-cosine direction, so
   its weakness is magnitude and generation quality rather than universal
   failure to steer.
4. **The fluency term is a real tradeoff, not a free improvement.** `lambda =
   0.4` lowers mean perplexity and repetition for every family, but also lowers
   the average steering signal, especially for scheduling.
5. **There is no evidence of aggregate semantic collapse with position among
   the cells that remain comparable, but the full 200-token persistence
   question is not resolved for every prompt.** Two unsteered baselines end
   before 60 tokens, and late-bin estimates therefore cover a selected subset
   of prompts and cells.

The practical recommendation is **schedule at `lambda = 0` when steering
strength is the priority, and blend at `lambda = 0.4` when a better
steering-fluency compromise is required**. Vote should remain a diagnostic
family until its short visible outputs and high perplexity are understood.

## Experimental identity

- Model: `meta-llama/Llama-3.1-8B-Instruct`, bf16.
- Decoding: deterministic Top-k Concept-Guided Decoding (CGD), `k = 4`.
- Survivor rule: greedy one-step lookahead, unchanged across methods.
- Horizon: `steps = 200`; the historical loop proposes a seed before its step
  iterations, so nominal steps and re-tokenized visible length are logged
  separately.
- Fluency weights: `lambda in {0, 0.4}`.
- Concepts: 15 selected pairs, `min_lemmas = 11`, `max_tokens = 3`.
- Prompts: `short_story`, `favorite_place`, `weekends`, `good_life`, and
  `weather`.
- Matrix: 15 pairs x 5 prompts x 3 methods x 2 fluency weights = 450 records.
- Baselines: one deterministic, prompt-matched unsteered continuation reused
  across all pairs and methods.
- Generation time recorded inside the 450 cells: 10,966.6 seconds (3.05 hours),
  excluding model loading and the subsequent trajectory analysis.

The three methods modify one CGD ingredient each:

- `blend`: use the polar/Procrustes mean of both frames as one guide.
- `vote`: retain both guides, z-score each concept across the candidate pool,
  then sum their scores.
- `schedule`: retain both guides and alternate the active concept with a reset
  round-robin scheduler.

## Metrics

For concept `i`, an expression delta is the mean FRH projection of the steered
continuation minus that of its prompt-matched baseline. Token cosine is a
gauge-invariant measure of hidden-state proximity to the concept span and is
subtracted from the same baseline.

- `joint expression = min(delta_a, delta_b)`.
- `joint token cosine = min(token_cos_delta_a, token_cos_delta_b)`.
- `both positive` is the fraction of cells for which the corresponding joint
  value is greater than zero.
- Whole-continuation PPL ratio is continuation-only perplexity divided by
  baseline perplexity. Values below 1 improve on the baseline; the project
  fluency flag is `> 2.5`.
- Trajectory PPL uses the corrected `continuation-prefix-masked-v2` metric. A
  continuation is scored once, each token loss remains conditioned on all
  earlier continuation tokens, and a steered/baseline bin is restricted to
  the same shared tokens. Bins with fewer than 10 shared tokens are omitted.
  Group curves exponentiate the mean log ratio (a geometric mean) and require
  valid PPL for at least 50% of the trajectory cells.
- Repeated 4-gram rate is the fraction of continuation 4-grams that repeat.

Expression uses the gauge-sensitive FRH quantity and token cosine uses a
gauge-invariant subspace quantity. Agreement is helpful evidence, but the two
measure different objects and should not be substituted for one another.

## Aggregate results

Each row averages 75 pair-prompt cells.

| Method | lambda | Joint expression | Both expression positive | Joint token cosine | Both token cosine positive | PPL ratio mean / median | PPL > 2.5 | Visible tokens | Repeated 4-gram rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blend | 0.0 | 1.9812 | 85.3% | 0.0151 | 85.3% | 3.95 / 2.55 | 50.7% | 197.3 | 0.3677 |
| schedule | 0.0 | **2.1888** | **88.0%** | **0.0179** | 89.3% | 3.72 / 2.13 | 45.3% | 203.5 | 0.3458 |
| vote | 0.0 | 1.2394 | 66.7% | 0.0153 | **94.7%** | 10.59 / 4.27 | 68.0% | 123.5 | 0.2541 |
| blend | 0.4 | **1.8853** | 81.3% | **0.0142** | 82.7% | 2.69 / **1.44** | 38.7% | 189.7 | 0.2954 |
| schedule | 0.4 | 1.7963 | **82.7%** | 0.0128 | 82.7% | **2.33** / 1.55 | **36.0%** | 194.2 | 0.2844 |
| vote | 0.4 | 1.0844 | 65.3% | 0.0106 | **84.0%** | 5.45 / 3.91 | 61.3% | 127.3 | **0.1657** |

### Method rankings are fluency-dependent

At `lambda = 0`, scheduling leads both mean steering measures. On joint token
cosine it wins 31/75 cells, versus 29 for blend and 15 for vote. Joint
expression produces a 32/32 split between schedule and blend, with vote
winning 11 cells.

At `lambda = 0.4`, blend becomes the clear cell-level leader: it wins 39/75
cells on joint token cosine and 35/75 on joint expression. Schedule wins 21 and
28 cells respectively; vote wins 15 and 12.

This interaction matters more than a single family-wide ranking. Round-robin
scheduling benefits most from pure concept scoring, while the composite blend
retains more of its signal after cumulative log-probability is added.

### `lambda = 0.4` improves fluency but weakens steering

Relative to the matched `lambda = 0` cell, the mean changes are:

| Method | Change in joint expression | Change in joint token cosine | Change in PPL ratio | Change in repeated 4-gram rate |
|---|---:|---:|---:|---:|
| blend | -0.0958 | -0.0010 | -1.2638 | -0.0723 |
| schedule | -0.3925 | -0.0052 | -1.3839 | -0.0614 |
| vote | -0.1550 | -0.0047 | -5.1329 | -0.0884 |

The PPL improvement is broad but not universal: it occurs in 61.3% of blend,
66.7% of schedule, and 64.0% of vote cells. Steering improves after enabling
the fluency term in fewer than half the cells for blend and vote and in roughly
30% for schedule. The correct interpretation is therefore a Pareto movement,
not a uniform model-quality improvement.

### Vote shows a distinct long-horizon failure mode

Vote has a high fraction of positive joint token-cosine cells, especially at
`lambda = 0`, yet its mean expression is only 57% of schedule's and its mean
PPL ratio is almost three times schedule's. Its re-tokenized visible
continuations average only 124--127 tokens, compared with roughly 190--204 for
blend and schedule.

All 450 records are labeled `step_cap`, so this is not logged as ordinary EOS
termination. The gap between loop steps and visible re-tokenized output should
be audited for special-token selection or decoding cleanup before treating
vote's shorter text as benign early stopping. Its lower repetition rate is
also partly confounded by this shorter visible horizon.

## Position and prompt effects

### Longitudinal evidence

All six method/lambda conditions have positive mean joint expression and
positive mean joint token cosine in the first 20-token bin. Among cells with a
valid comparison near tokens 180--199, the means also remain positive:

| Method | lambda | Joint expression at 0--19 | Joint expression at 180--199 | Joint token cosine at 180--199 | Cells at 180--199 |
|---|---:|---:|---:|---:|---:|
| blend | 0.0 | 1.1326 | 1.9464 | 0.0208 | 43 |
| blend | 0.4 | 1.1605 | 1.4777 | 0.0180 | 37 |
| schedule | 0.0 | 1.3199 | 1.8569 | 0.0216 | 45 |
| schedule | 0.4 | 1.0217 | 1.3659 | 0.0162 | 42 |
| vote | 0.0 | 1.1510 | 2.8981 | 0.0310 | 22 |
| vote | 0.4 | 1.0804 | 2.1398 | 0.0255 | 22 |

These late-bin values are descriptive, not a clean estimate of persistence for
the full matrix. Baseline subtraction is defined only when both continuations
have tokens. The `weather` and `weekends` baselines provide no bins after
tokens 40--59, and short vote outputs reduce coverage further. Consequently,
late bins overrepresent `short_story`, `favorite_place`, and `good_life` and
may select unusually persistent generations. The final partial bin has only
about 13--17% mean coverage and remains unsuitable for semantic method
ranking. It has fewer than 10 shared tokens in every cell, so corrected PPL v2
intentionally reports no value there.

### Corrected trajectory PPL has no spikes

The original trajectory plot decoded each 20-token slice as a standalone text
and evaluated it from BOS. That discarded the continuation prefix, made short
tails especially unstable, and then amplified them by taking an arithmetic
mean of raw ratios. Those spikes were measurement artifacts, not sudden
generation failures.

After the corrected prefix-conditioned rerun, all supported group-level PPL
ratios are finite and the extreme spikes disappear. For example, the
geometric ratios for tokens 0--19 / 20--39 / 180--199 are:

| Method | lambda | 0--19 | 20--39 | 180--199 |
|---|---:|---:|---:|---:|
| blend | 0.0 | 5.03 | 6.17 | 2.25 |
| schedule | 0.0 | 5.02 | 6.54 | 2.36 |
| vote | 0.0 | 4.78 | 6.29 | 1.75 |
| blend | 0.4 | 2.79 | 3.21 | 1.91 |
| schedule | 0.4 | 2.39 | 2.63 | 2.18 |
| vote | 0.4 | 3.21 | 3.57 | 2.38 |

Late ratios are conditional on the cells with shared coverage and should not
be read as a balanced comparison over all 75 pair-prompt cells. The
whole-continuation PPL table above is unchanged by this correction.

### Prompt dependence is large

`good_life` produces the strongest apparent joint steering for every method,
but also extreme repetition: mean repeated 4-gram rates range from 0.61 to
0.86. List continuation makes the concept signal easy to sustain while
creating a degeneration confound.

`favorite_place` is the hardest prompt for vote: its mean joint expression is
negative at both lambda settings and its PPL ratios are 31.5 and 11.8.
`short_story` also yields near-zero or negative expression for several
conditions despite generally positive token cosine. `weather` and `weekends`
are more fluent but their baselines end too early for a full late-horizon
comparison.

Prompt composition should therefore be treated as part of the experimental
identity. A single aggregate over list, narrative, and assistant-self-report
prompts hides qualitatively different failure modes.

## Pair effects

`respite_universe` is the most consistently strong pair: it is among the top
three pairs for both steering metrics in nearly every method/lambda condition.
`end_country` is also strong, particularly for schedule and at `lambda = 0.4`.
At the other end, `town_flood` and `passing_dawn` are frequently weak after the
fluency term is enabled.

Pair correlation `rho` is not a sufficient predictor of performance. Across
the 15 pairs, Spearman correlations between rho and joint token cosine range
from -0.02 to 0.45 depending on method and lambda, while expression
correlations change sign. The modest sample and curated pool do not support a
general rho threshold for choosing a long-generation method.

## Limitations and interpretation boundaries

- This is one bf16 model, one deterministic decode per cell, five prompts, and
  15 curated pairs. There are no sampling replicates or uncertainty intervals.
- Whole-continuation expression compares texts of unequal visible length.
  Trajectories correct this locally through shared-bin coverage, but late-bin
  coverage is selected and incomplete.
- Continuation-only PPL is unconditional on the prompt. It is a useful
  guardrail, not a complete conditional fluency evaluation.
- Corrected segment PPL is prefix-conditioned, shared-span matched, and
  geometrically aggregated. Short tails are masked rather than extrapolated;
  late-position fluency claims are still limited by selected coverage.
- Repetition and steering are prompt-dependent. In particular, the strong
  `good_life` signal is entangled with repetitive list structure.
- The experiment preserves the paper's greedy survivor rule. Results do not
  establish the same ranking for a true beam survivor rule.

## Recommended follow-ups

1. Audit vote's generated token IDs and special-token behavior; nominal steps
   should be reconciled with the much shorter visible continuation.
2. Regenerate fixed-length unsteered references, or use prefix-matched
   baselines, so every prompt supports the same 200-token trajectory horizon.
3. Repeat a compact matrix with prompt-conditioned PPL and a semantic-quality
   judge, retaining repetition diagnostics.
4. Use blend at `lambda = 0.4` and schedule at `lambda = 0` as the primary
   recipes in follow-up long-generation comparisons; retain vote as a control.
5. Separate list prompts from narrative prompts in tables instead of averaging
   them without a prompt-family label.

## Artifacts and provenance

- Source notebook: [`53_long_generation_composition.ipynb`](../53_long_generation_composition.ipynb)
- Executed notebook: [`outputs/53_long_generation_composition_full.ipynb`](../outputs/53_long_generation_composition_full.ipynb)
- Inputs: [`long-generation-inputs.yaml`](../long-generation-inputs.yaml)
- Raw records: [`resources/53_long_composition.jsonl`](../resources/53_long_composition.jsonl)
- Pair summary: [`resources/53_long_composition_summary.csv`](../resources/53_long_composition_summary.csv)
- Trajectories: [`resources/53_long_composition_trajectories.csv`](../resources/53_long_composition_trajectories.csv)
- Main figure: [`resources/53_long_composition_metrics.png`](../resources/53_long_composition_metrics.png)
- Trajectory figure: [`resources/53_long_composition_trajectories.png`](../resources/53_long_composition_trajectories.png)
- Timing preflight: [`resources/53_long_composition_timing.csv`](../resources/53_long_composition_timing.csv)
- Superseded PPL-v1 analysis artifacts (kept for provenance):
  [`.scratch/long-generation-ppl-v1-20260811/`](../.scratch/long-generation-ppl-v1-20260811/)
