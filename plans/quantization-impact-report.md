# Quantization Impact Report — AWQ-INT4 vs bf16 Llama-3.1-8B-Instruct

Every multiconcept experiment (E0–E3b, notebooks 16–23) and the broad
showcase (notebook 24 vs 29) rerun with the unquantized
`meta-llama/Llama-3.1-8B-Instruct` (bf16, sharded) in place of
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (fp16). Identical
design everywhere: the 30 vetted pairs of `15_e0_selected_pairs.json`, same
prompts, k, steps, seeds and metrics. Reruns are the generated
`<notebook>_unquantized.ipynb` copies; all artifacts carry the
`_unquantized` suffix. Runs executed 2026-07-16/17 (E-series chain: 1 h 52 m
total on 3× RTX A5000).

**Reading the numbers.** `joint_z` (z-scored joint expression) and rates
(`both_positive`, `clean_rate`, presence) are scale-free and comparable
across the two variants. Raw `delta` magnitudes live in each variant's own
whitened-unembedding scale; within-Llama the scales are near-identical, but
treat small raw-delta differences with care.

## Verdict in one paragraph

Dequantization strengthens steering across the board (singles nearly double;
every showcase method's joint gain turns positive) but **does not change the
paper's core recipe**: true beam search + simple frame blend remains the
best method at every tier, and the λ ≈ 0.4 fluency setting keeps the same
trade-off. Two documented findings do move: **the aligned-blend advantage at
low ρ (E2/E2b) largely dissolves** — it was substantially a quantization
artifact — and **score-space methods change the most in both directions**
(greedy voting and score-space negative steering improve; the beam-tier
score vote loses most of its advantage). The Llama-vs-Gemma family gap
survives dequantization.

## Per-experiment comparison

### E0 pilot (16) — lemma-presence metric

Identical: presence rates are 0.0 for both methods under both variants.
The methodological conclusion (surface-lemma presence is not a usable
success metric; expression delta is needed) is fully robust.

### E0 pilot v2 (17) — expression metric, greedy baselines

| method, k | joint_z AWQ | joint_z bf16 |
|---|---|---|
| F1.a blend, k=3 | −0.256 | −0.274 |
| F1.a blend, k=4 | −0.050 | −0.047 |
| F2.a vote, k=3 | −0.310 | −0.311 |
| F2.a vote, k=4 | −0.231 | **−0.070** |

Baselines replicate closely, with one exception: greedy score voting at k=4
improves markedly on bf16, closing most of its gap to the blend. First
sign of the pattern that score-space methods are the quantization-sensitive
ones.

### E2 (18) — blend geometries

| geometry | joint_z AWQ | joint_z bf16 |
|---|---|---|
| aligned | −0.170 | −0.171 |
| frechet | −0.179 | −0.204 |
| extrinsic | −0.181 | −0.229 |

Same ordering, gaps still small. The E2 conclusion (metric doesn't matter,
gauge does) stands, but see E2b for where the aligned story weakens.

### E2b (19) — aligned-vs-simple crossover in ρ ⚠️ finding changed

Mean aligned-minus-simple gap:

| regime | AWQ | bf16 |
|---|---|---|
| ρ < 0.30 | **+0.156** | +0.040 |
| ρ ≥ 0.30 | **−0.109** | +0.022 |

On AWQ, aligned blending clearly won below ρ ≈ 0.30 and lost above — the
documented crossover. On bf16 both effects collapse to noise level
(per-pair gaps mostly within ±0.03, largest +0.21 on woman/king vs +0.27 on
joy/music for AWQ). **The low-ρ aligned advantage was substantially a
quantization artifact**: AWQ noise appears to worsen the gauge mismatch
between dissimilar concept frames, which is exactly what the
generalized-Procrustes alignment repairs. On clean weights there is little
left to repair. `multiconcept-cgd-methods.md` §2 ("empirically wins for
ρ < 0.30 pairs") should be read with this caveat.

### E1 (20) — family sweep ⚠️ margin changed, winner unchanged

| method | joint_z AWQ | joint_z bf16 | Δ |
|---|---|---|---|
| beam + frame blend | 0.396 | **0.424** | +0.03 |
| beam + score vote | 0.383 | 0.172 | **−0.21** |
| aligned blend (greedy) | −0.212 | −0.192 | +0.02 |
| simple blend (greedy) | −0.217 | −0.258 | −0.04 |
| weighted vote (greedy) | −0.350 | −0.274 | +0.08 |
| round-robin schedule | −0.342 | −0.299 | +0.04 |
| softmin vote | −0.257 | −0.337 | −0.08 |
| sentence schedule | −0.387 | −0.394 | −0.01 |
| constrained vote | −0.636 | −0.489 | +0.15 |

Winner and top-tier structure unchanged: beam search dominates, frame
blending on top. But on AWQ the beam score-vote was statistically tied with
the beam frame-blend (0.383 vs 0.396); on bf16 the frame blend is clearly
ahead (0.424 vs 0.172). The greedy mid-field reshuffles slightly (weighted
vote overtakes softmin). Bottom (constrained/lexicographic) stays bottom.

### E1b (21) — fluency dial λ

| λ | joint_z AWQ / bf16 | ppl_ratio AWQ / bf16 |
|---|---|---|
| 0.0 | −0.017 / 0.057 | 6.43 / 5.82 |
| 0.1 | 0.018 / 0.010 | 5.55 / 5.37 |
| 0.2 | 0.012 / −0.024 | 4.82 / 4.47 |
| 0.4 | −0.100 / −0.099 | 2.83 / 2.97 |
| 1.0 | −0.373 / −0.397 | 1.38 / 1.31 |
| 2.0 | −0.687 / −0.698 | 1.13 / 1.05 |

The dial's shape is essentially identical; the cost of λ = 0.4 in joint
expression is the same to three decimals (−0.100 vs −0.099) at the same
ppl ≈ 3. **The recommended recipe (beam + simple blend + λ ≈ 0.4) survives
dequantization unchanged.** Unquantized is slightly more fluent at λ = 0.

### E3 (22) — negative steering, greedy

| method | net AWQ / bf16 | clean_rate AWQ / bf16 |
|---|---|---|
| frame-diff | **0.846** / **0.791** | 0.350 / 0.308 |
| score-neg | 0.674 / 0.739 | 0.233 / 0.258 |
| attract-only | 0.449 / 0.406 | 0.233 / 0.150 |

Ranking unchanged (frame-diff first), but score-space negative steering
improves on bf16 and narrows the gap from 0.17 to 0.05 net.

### E3b (23) — negative steering, beam + λ

| method, λ | net AWQ / bf16 | clean_rate AWQ / bf16 |
|---|---|---|
| frame-diff, 0.0 | **1.298** / **1.278** | 0.350 / 0.333 |
| score-neg, 0.0 | 1.051 / 1.160 | 0.342 / 0.300 |
| frame-diff, 0.4 | 1.046 / 0.996 | **0.417** / **0.408** |
| score-neg, 0.4 | 0.821 / 0.863 | 0.325 / 0.375 |
| attract-only, 0.0 | 0.602 / 0.569 | 0.167 / 0.133 |
| attract-only, 0.4 | 0.537 / 0.529 | 0.133 / 0.158 |

Same structure: frame-diff maximizes net expression at λ = 0; λ = 0.4
trades net for the best clean rate. Score-neg again gains from
dequantization (both net and, at λ = 0.4, clean rate).

### Showcase (24 vs 29) — 30 pairs, greedy tier

| method | joint Δ AWQ → bf16 | both-improved AWQ → bf16 |
|---|---|---|
| blend | +0.02 → +0.27 | 51% → 52% |
| vote | −0.12 → +0.30 | 45% → 59% |
| schedule | −0.01 → +0.27 | 47% → 52% |

Singles mean Δ 0.41 → 0.75 (~1.8×); ppl ratio improves ~7.5 → ~6.8. All
methods turn jointly positive; the greedy-tier method ranking flips to vote
first — consistent with E0v2/E1/E3's pattern of score methods gaining most
at the greedy tier. Both-improved rates stay far below Gemma's 97–99%
(notebook 26), so **the family gap is real, not a quantization artifact**.

## Most relevant impacts

1. **The headline recipe is quantization-robust.** Beam search + simple
   frame blend + λ ≈ 0.4 is the best configuration on both variants, with
   nearly identical trade-off numbers (E1, E1b, E3b). Nothing in the
   recommended pipeline needs revisiting.

2. **The aligned-blend story needs a correction.** The E2b crossover
   (aligned wins below ρ ≈ 0.30) largely disappears on clean weights; the
   gauge mismatch it repairs seems substantially induced by AWQ noise. On
   bf16, simple and aligned blending are equivalent within noise at all ρ.

3. **Score-space methods are the quantization-sensitive family — in both
   directions.** Greedy voting, k=4 voting, and score-space negative
   steering all improve notably on bf16 (consistent with z-scored voting
   being most exposed to per-concept score calibration, which quantization
   noise degrades). The one reversal: the *beam-tier* score vote loses its
   near-parity with the frame blend (0.383→0.172). Frame-space methods
   barely move anywhere.

4. **Absolute steering strength was being understated by AWQ** — *but see
   the notebook-30 addendum below: much of the raw-delta doubling is a
   metric-scale artifact.* Singles nearly double in raw expression units,
   all showcase methods turn positive, fluency improves slightly.

5. **Cross-model conclusions survive.** Unquantized Llama still improves
   both concepts in only ~52–59% of showcase runs vs Gemma's 97–99%: the
   family difference (notebooks 25/26) is about the model, not the weights'
   precision.

## Addendum (notebook 30) — the raw-delta gap is mostly metric scale

Re-scoring both showcase logs with the gauge-invariant angle metrics
(`frames/linalg/angles.py`, notebook `30_angle_metric_validation.ipynb`)
qualifies impact #4. In bounded, scale-free units the two variants steer
almost equally hard per token — mean token-cosine delta for the blend is
0.0076 (AWQ) vs 0.0070 (bf16); singles 0.0091 vs 0.0077 — while the raw
expression deltas differed ~2×. The doubling therefore reflects the
expression metric's sensitivity to each variant's hidden-state/whitening
scale, not doubled steering. The subspace-level movement (geodesic
distance) is genuinely larger on bf16 (~40%: blend 0.0110 vs 0.0079), so
dequantization does help — but modestly, at the level of the text's
dominant directions rather than per-token alignment. Scale-free rates
(both-improved) and rankings elsewhere in this report are unaffected.

## Provenance

- Rerun notebooks: `16..23_*_unquantized.ipynb` (generated copies; originals
  untouched), executed sequentially 2026-07-16 23:22 → 2026-07-17 01:14.
- Showcase comparison from `24_showcase.jsonl` (AWQ) and notebook 29
  (`29_llama8b_bf16.jsonl`, `29_multimodel_summary.csv`).
- Summary artifact pairs: `resources/<stem>.csv` vs
  `resources/<stem>_unquantized.csv` for stems `16_e0_summary`,
  `17_e0_summary_v2`, `18_e2_summary`, `19_e2b_summary`, `20_e1_summary`,
  `21_e1b_summary`, `22_e3_summary`, `23_e3_beam_summary`.
- Notebook 14 (golden reference) intentionally not rerun: it pins the AWQ
  behavior by design. Notebook 15 (pair selection) intentionally not rerun:
  shared pairs are what make these comparisons direct.
