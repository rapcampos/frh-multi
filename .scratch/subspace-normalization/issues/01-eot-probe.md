# 01 — EOT probe: does `‖Bᵀh‖` prefer end-of-turn candidates?

Status: claimed
Type: research

## Question

Subspace steering truncates early on 23-31% of runs against frame blending's
0.4-2.8% (see `../spec.md`). Is the mechanism that the unnormalized score
`‖Bᵀh‖` rewards the large-norm hidden states that sit next to EOT?

This gates issue `02`. If EOT candidates show no raw-score advantage, the
truncation cause is elsewhere and normalization is justified only by metric
consistency and λ comparability, not by fixing truncation.

## Method

`probe_eot.py` replays the real selection loop from
`subspace/representation.py::generate_with_topk_guide` — same `_candidates`
call, same `subspace_alignment` scoring, same cumulative-max advance — and
records for every candidate at every step: the token, whether it is EOS, the raw
score `‖Bᵀh‖` at the newest position, the hidden-state norm `‖h‖`, and the
normalized score `‖Bᵀh‖/‖h‖`.

Run over both concept sources and four configs (two 1-unit, two 4-unit) across
all eight NB68 prompts, reusing `ModelBackend` so concept construction is
identical to the battery.

## What would confirm the hypothesis

EOT candidates carry a systematically higher raw score than non-EOT candidates,
and that gap shrinks or reverses under the normalized score.

## What would refute it

EOT candidates score at or below non-EOT under the raw score — truncation is
then driven by something else, most likely the candidate set itself (EOT already
being top-k by probability) rather than by the steering score.

## Answer

**Hypothesis refuted. The real cause is round-trip token loss in `_candidates`.**

`probe_eot.py` (v1) found **zero** EOT candidates in 6,144 observations, with
every run completing all 24 steps. EOT is never scored, so the score cannot be
preferring it. `eos_token_id` is correct (128009 = `<|eot_id|>`), so this is not
an ID mismatch.

v1 also failed to reproduce the battery — zero short runs for configs that
truncate 12.5-75% of the time in NB68 — because it instrumented the loop but
never decoded. `probe_truncation.py` (v2) called the real
`generate_with_topk_guide` and reproduced short runs with `eot_count == 0`
throughout, in two flavours: real token loss (p04: 3 generated ids after 24
steps) and subword garbling at full length (p06:
`'kaleidesopecopeptiomeof soundscrapessmelling'`, `'24 x   a day'`).

`probe_roundtrip.py` (v3) located it. Both pipelines rebuild their inputs every
step with `make_input(decode(tokens), padding=True)` — `subspace/representation.py:291`
and `frames/representations/frame.py:732` are the same code. Measuring the real
(non-pad) length of each surviving beam per step:

| | mean growth/step | min | lossy steps (of 24) |
|---|---|---|---|
| frh_blend | 1.005 | 1.0 | 0 on all 8 prompts |
| subspace | **0.526** | **-1.0** | up to 24 (p07), 21 (p04), 15 (p03) |

A step appends exactly one token, so growth below 1.0 is text that did not
survive re-tokenization; negative growth means the sequence shrank. Frame
blending never loses a token. Subspace loses one on roughly half its steps.

Final real lengths on the same prompts: frame 38-42 across the board, subspace
16-41 (p07: 16, p04: 19, p08: 24).

Note the first measurement of this was wrong: tracking `tokens.shape[-1]` showed
1.000 growth for both methods, because that is the padded batch width and is
guaranteed to grow by one per step. Per-row non-pad length is the correct
measurement.

### Consequences

The mechanism is the round trip; the subspace score is only the trigger, because
it selects tokens that do not round-trip cleanly. This explains every symptom at
once: short continuations, subword garbling, the worsening with unit count, and
why configs with a repel term truncate 6x *less* (6.5% vs 39%) — the subtraction
changes which tokens win.

It also means **normalization is not the fix for truncation**. Issue `02` still
stands on metric consistency and λ comparability, but its truncation rationale is
withdrawn. A new issue `04` should carry token ids through the loop instead of
round-tripping through text, and should precede `02` and `03`.
