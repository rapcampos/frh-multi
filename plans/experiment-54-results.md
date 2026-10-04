# Experiment 54: Woman/Man Differential Steering Results

- Run date: 2026-08-10
- Trajectory-analysis rerun: 2026-08-11 (corrected PPL v2)
- Status: complete (60/60 guided continuations)

## Executive conclusion

Negative steering produces a differential signal, but **frame-space
subtraction is the better balanced method**. Pool-normalized score subtraction
suppresses the repel concept more aggressively, yet often suppresses the
attract concept as well and can cause severe fluency failures.

Findings ranked by impact:

1. **Both negative methods suppress the repel concept beyond the positive-only
   control.** Frame difference does so on 9/10 `man_minus_woman` prompts and
   7/10 `woman_minus_man` prompts; score subtraction does so on 10/10 and 9/10.
2. **Frame difference gives the best steering-fluency balance.** It improves
   net expression over attract-only in both directions, preserves a slightly
   positive mean attract delta, and has mean PPL ratios of 2.06 and 1.88.
3. **Score subtraction often creates contrast by lowering both concepts.** Its
   mean attract delta is negative in both directions. The strict clean-success
   metric exposes this: only 4/10 and 1/10 prompts simultaneously increase the
   attract concept and decrease the repel concept.
4. **The behavior is direction-asymmetric.** `man_minus_woman` is more stable;
   `woman_minus_man` has weaker clean success, severe score-neg fluency
   outliers, and a late-bin contrast drop.
5. **The conclusion is metric-dependent.** Continuous FRH expression shows
   clean success in 10--40% of prompts, while the gauge-invariant token-cosine
   criterion shows it in only 0--10%. The experiment establishes contrast more
   clearly than clean, metric-robust differential steering.

The recommended recipe for follow-up work is **the projected frame difference
at `lambda = 0.4`**, with attract-only retained as a correlated-lift control.
Score subtraction is useful as an aggressive diagnostic control, not as the
default method.

## Experimental identity

- Model: `meta-llama/Llama-3.1-8B-Instruct`, bf16.
- Decoding: deterministic Top-k Concept-Guided Decoding (CGD), `k = 4`.
- Survivor rule: greedy one-step lookahead, unchanged across conditions.
- Horizon: `steps = 24`.
- Fluency weight: `lambda = 0.4`.
- Concept correlation measured by the model: `rho = 0.41927`.
- Directions:
  - `woman_minus_man`: attract `woman.n.01`, repel `man.n.01`;
  - `man_minus_woman`: attract `man.n.01`, repel `woman.n.01`.
- Prompts: all ten canonical prompts in `long-generation-inputs.yaml`.
- Matrix: 2 directions x 10 prompts x 3 guided methods = 60 records.
- Baselines: one deterministic, prompt-matched unsteered continuation reused
  across directions and methods.
- Generation time recorded inside the 60 cells: 91.4 seconds.
- Final run placement: approximately 13.48 GiB on physical GPU 2 and 1.83 GiB
  of model weights on physical GPU 1, with no CPU or disk offload. All final
  rows use this same placement. An earlier nine-row CPU-offloaded trial was
  preserved separately and excluded.

The evaluated guided conditions are:

- `attract-only`: positive guide only; controls for correlated concept lift.
- `frame-diff`: the FRH projected differential guide
  `polar(attract - repel)`.
- `score-neg`: separate guides, per-candidate-pool z-score normalization, and
  weights `[+1, -1]`.

The unsteered rows in the summary are synthetic zero-delta controls. The raw
JSONL contains only the 60 guided cells.

## Metrics

For each direction:

- `delta_attract` and `delta_repel` are steered-minus-baseline FRH expression.
- `net expression = delta_attract - delta_repel`; larger values indicate more
  contrast but do not require the attract concept to increase.
- `clean expression` requires `delta_attract > 0` and `delta_repel < 0`.
- Token-cosine versions use the gauge-invariant distance of hidden states to
  each concept span.
- Whole-continuation PPL ratio is continuation-only perplexity divided by
  baseline perplexity. Trajectory PPL uses the corrected
  `continuation-prefix-masked-v2` metric: token losses remain conditioned on
  their continuation prefix, bins use the same shared steered/baseline span,
  and bins shorter than 10 shared tokens are omitted. Group ratios are
  geometric means and require at least 50% valid cell coverage.

The distinction between net and clean outcomes is essential. A method can
obtain a positive net contrast by suppressing both concepts if the repel
concept falls further.

Surface-word presence was deliberately not evaluated: it was not a planned
endpoint for experiment 54, and prior experiments found it too sparse to
represent steering. Claims below use continuous expression, angle-based
metrics, and perplexity.

## Aggregate results

Each row averages ten prompts.

| Direction | Method | Attract delta | Repel delta | Net expression | Clean expression | Net token cosine | Clean token cosine | PPL mean / median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| man - woman | attract-only | 1.1065 | 0.6826 | 0.4238 | 0% | 0.0036 | 10% | 2.21 / 1.56 |
| man - woman | frame-diff | **0.0245** | -0.5188 | **0.5433** | 30% | **0.0037** | 10% | **2.06 / 1.82** |
| man - woman | score-neg | -0.3691 | **-0.8667** | 0.4976 | **40%** | 0.0024 | 0% | 3.49 / 2.64 |
| woman - man | attract-only | 0.3116 | 0.1881 | 0.1236 | 10% | -0.0001 | 0% | 2.31 / 1.92 |
| woman - man | frame-diff | **0.0813** | -0.2824 | 0.3636 | **20%** | 0.0012 | **10%** | **1.88 / 1.89** |
| woman - man | score-neg | -0.6486 | **-1.0454** | **0.3969** | 10% | **0.0022** | **10%** | 17.86 / 3.16 |

Bold entries identify the more favorable negative method within a direction;
they are not intended to compare the social desirability of the directions.

## Main findings

### Negative steering adds suppression beyond positive guidance

Attract-only raises both concepts on average. For `man_minus_woman`, the
attract and repel deltas are +1.1065 and +0.6826; for `woman_minus_man`, they
are +0.3116 and +0.1881. This correlated lift is expected for concept frames
with `rho = 0.419` and shows why a positive-only comparison is insufficient.

Relative to the matched attract-only continuation, the negative conditions
change the repel concept as follows:

| Direction | Method | Prompts with more repel suppression | Mean change in repel delta | Mean change in attract delta | Mean improvement in net expression |
|---|---|---:|---:|---:|---:|
| man - woman | frame-diff | 9/10 | -1.2014 | -1.0819 | +0.1194 |
| man - woman | score-neg | 10/10 | -1.5493 | -1.4755 | +0.0738 |
| woman - man | frame-diff | 7/10 | -0.4704 | -0.2304 | +0.2400 |
| woman - man | score-neg | 9/10 | -1.2335 | -0.9602 | +0.2733 |

Both methods therefore perform genuine additional suppression, but neither
isolates the repel concept perfectly. Score-neg obtains its larger suppression
by paying an almost equally large attract penalty. Frame-diff is less
aggressive and better preserves the desired side of the differential.

### Strict clean success is limited

For `man_minus_woman`, frame-diff is clean on `favorite_place`,
`short_story`, and `women`; score-neg is clean on `favorite_place`,
`most_important`, `weather`, and `weekends`.

For `woman_minus_man`, frame-diff is clean on `favorite_place` and `women`,
while score-neg is clean only on `women`. This prompt-level view explains why
positive net contrast alone overstates selectivity.

The gauge-invariant token-cosine test is stricter still. Every
direction-method condition has at most one clean token-cosine prompt. Net
token cosine is usually positive but small, indicating relative separation
without robust simultaneous movement toward one span and away from the other.

### Score subtraction has severe asymmetric fluency failures

The `woman_minus_man` score-neg mean PPL ratio of 17.86 is driven by two clear
failures:

- prompt `men`: PPL ratio 134.10 and only 5 visible re-tokenized tokens;
- prompt `weekends`: PPL ratio 19.04 and only 7 visible tokens.

Its median is 3.16, so even after discounting the outliers it remains worse
than frame-diff. Score-neg also has a PPL ratio above the 2.5 project threshold
on 60% of `man_minus_woman` prompts and 50% of `woman_minus_man` prompts.
Frame-diff crosses that threshold on 20% and 10% respectively.

Repetition is not the dominant failure at this 24-step horizon: frame-diff and
score-neg have near-zero average repeated 4-gram rates. The problematic cases
are short and low-probability rather than long repetitive loops. These
whole-continuation failures are unchanged by the trajectory-PPL correction.

### Corrected trajectory PPL supports only the full first bin

The corrected geometric PPL ratios for tokens 0--19 are:

| Direction | attract-only | frame-diff | score-neg |
|---|---:|---:|---:|
| man - woman | 2.14 | **1.78** | 2.53 |
| woman - man | **1.92** | 1.99 | 2.58 |

The short final bin has a valid PPL value for only one trajectory cell per
direction-method group (11--14% support), so the group-level plot masks every
final-bin PPL point. The old final-bin spikes came from scoring isolated text
fragments from BOS and averaging raw ratios; they are not retained as results.

### Direction asymmetry persists over position

The first 20-token bin favors both negative methods in both directions. In the
short final bin, `man_minus_woman` retains or increases its net expression,
while `woman_minus_man` weakens sharply:

| Direction | Method | Net expression, tokens 0--19 | Net expression, final shared bin | Clean rate, final shared bin |
|---|---|---:|---:|---:|
| man - woman | attract-only | 0.3233 | 0.8285 | 11.1% |
| man - woman | frame-diff | 0.4822 | 0.7117 | 33.3% |
| man - woman | score-neg | 0.4182 | 0.7174 | 33.3% |
| woman - man | attract-only | 0.2823 | -0.3210 | 0% |
| woman - man | frame-diff | 0.4474 | 0.0030 | 11.1% |
| woman - man | score-neg | 0.5665 | 0.1032 | 0% |

This is suggestive rather than definitive. The final bin averages only about
7--8 shared tokens (`coverage approximately 0.38` of a 20-token bin), covers
nine prompts for most conditions and seven for `woman_minus_man` score-neg,
and has no group-level PPL estimate under the minimum-token and 50%-support
rules. Its semantic measurements nevertheless agree with the
whole-continuation direction asymmetry and motivate a longer, fixed-coverage
replication.

## Qualitative and social-bias review

The executed notebook prints all 60 continuations with an empty
`stereotype_review` column. That column has not yet been manually annotated.
The quantitative differential metrics therefore must not be interpreted as a
finding about stereotype reduction, social fairness, or the normative quality
of either direction.

Several prompts explicitly ask about men or women, so a blinded manual review
should label occupational, familial, personality, agency, and competence
stereotypes before any qualitative conclusion is reported. This experiment is
a bounded dual-use evaluation of steering behavior, not a recommendation to
deploy gender steering.

## Limitations and interpretation boundaries

- Only one correlated concept pair, two directions, one bf16 model, ten
  prompts, and one deterministic continuation per cell were tested.
- The direction labels reverse attract and repel but do not make the two
  concept representations, prompts, or baseline projections symmetric.
- Strict clean success is a binary sign test with only ten observations per
  row. A one-prompt change moves the rate by ten percentage points.
- FRH expression and gauge-invariant token cosine disagree on several prompts.
  Neither should be presented as a definitive semantic or social-bias score.
- Continuation PPL is unconditional on the prompt. The extreme outliers are
  meaningful guardrails, but the metric is not a complete fluency judge.
- All records reached the loop's logged step cap, yet two score-neg outputs are
  much shorter after decoding and re-tokenization. Special-token behavior
  should be inspected before the next run.
- The final trajectory bin is short and incomplete. Its semantic values cannot
  establish long-horizon negative steering, and its group PPL is deliberately
  omitted.
- No manual stereotype review has been completed.

## Recommended follow-ups

1. Use frame-diff as the primary negative-steering method and retain
   attract-only plus score-neg as controls.
2. Run a lambda sweep for frame-diff in each direction; the current result
   cannot show whether a slightly larger fluency weight preserves contrast
   while lowering the remaining PPL failures.
3. Audit the token IDs for the short score-neg outputs and log special-token
   counts directly rather than inferring termination from cleaned text.
4. Extend to several human-interpretable, non-sensitive correlated pairs to
   test whether direction asymmetry is specific to the woman/man concepts.
5. Repeat the gender differential with fixed-length or prefix-matched
   baselines and more prompts before making a persistence claim.
6. Complete the qualitative stereotype annotation separately from the
   steering metrics and report inter-rater agreement.

## Artifacts and provenance

- Source notebook: [`54_long_generation_negative.ipynb`](../54_long_generation_negative.ipynb)
- Executed notebook: [`outputs/54_long_generation_negative_full.ipynb`](../outputs/54_long_generation_negative_full.ipynb)
- Inputs: [`long-generation-inputs.yaml`](../long-generation-inputs.yaml)
- Raw guided records: [`resources/54_long_negative.jsonl`](../resources/54_long_negative.jsonl)
- Aggregate summary: [`resources/54_long_negative_summary.csv`](../resources/54_long_negative_summary.csv)
- Trajectories: [`resources/54_long_negative_trajectories.csv`](../resources/54_long_negative_trajectories.csv)
- Quadrant figure: [`resources/54_long_negative_quadrants.png`](../resources/54_long_negative_quadrants.png)
- Trajectory figure: [`resources/54_long_negative_trajectories.png`](../resources/54_long_negative_trajectories.png)
- Superseded PPL-v1 analysis artifacts (kept for provenance):
  [`.scratch/long-generation-ppl-v1-20260811/`](../.scratch/long-generation-ppl-v1-20260811/)
