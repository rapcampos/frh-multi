# Multi-Concept Concept-Guided Decoding — Results Report

Presentation-ready summary of the experimental results. Numbers are reproduced
by `22_results_report.ipynb` from the raw JSONL logs in `resources/`; figures
referenced below are saved there by the experiment notebooks (15–23) and by
notebook 22.

**Scope.** Extends FRH's Top-k Concept-Guided Decoding (CGD) from one concept
to several. All experiments: Meta-Llama-3.1-8B-Instruct-AWQ-INT4, WordNet
concepts (min 11 lemmas, ≤3 tokens), shared 20-prompt chat suite, greedy
k-lookahead with k=4 and 24 generated tokens unless stated. Steering success is
**Δexpression** (metric v2): the post-hoc projection of the generated
continuation onto the concept frame, minus the same projection of the unsteered
continuation. Fluency cost is the continuation-perplexity ratio vs the model's
own greedy output.

---

## 1. What is being compared

Three places to compose concepts C₁, C₂ in the CGD pipeline:

| family | idea | methods tested |
|---|---|---|
| **F1 — frame space** | build ONE composite frame, steer as usual | extrinsic (chordal) mean; **aligned (GPA) mean**; geodesic Fréchet mean; joint subspace (F1.b) |
| **F2 — score space** | score candidates against EACH concept, aggregate scores | weighted sum; softmin (AND); constrained (lexicographic) — all with mandatory per-step z-scoring |
| **F3 — time** | one concept at a time, alternate | round-robin per step; switch at sentence boundaries |

Plus two decoding-loop upgrades, orthogonal to the family choice: a **true
beam-search** variant (the original keeps k children of one parent; beams there
never truly diverge), applicable to F1/F2, and a **fluency dial**
(`fluency_weight` λ: selection score = concept + λ·logprob; λ=0 = original
behavior, λ→∞ = unsteered greedy).

The concept-pair correlation **ρ** (trace correlation of the two frames,
0.19–0.57 across our 30-pair survey) is the experimental axis: it was
hypothesized to predict composition interference.

## 2. Headline findings

1. **True beam search, not composition choice, is the biggest lever** (E1).
   Replacing the original single-parent greedy lookahead with genuine beam
   search flips joint expression from negative to positive in *all three* ρ
   strata at equal or lower fluency cost; beam-F1.a is on the Pareto frontier
   everywhere and is the *only* frontier member at low ρ.
2. **A fluency dial closes the fluency problem** (E1b). Blending the model's
   own cumulative token log-probability into candidate selection
   (`score = concept + λ·logprob`; `fluency_weight` on both loops, λ=0 is the
   untouched golden path, λ→∞ recovers unsteered greedy) turns the Pareto
   frontier into a knob: **λ≈0.4 reaches guardrail-level fluency (2.4–3.1×
   PPL) keeping ~85% of beam's expression**; λ≈1 is near-native (1.3–1.6×,
   ~0 flags) at ~50%. Mild λ (0.1–0.2) is a *free win* — joint expression
   often rises while PPL falls, because the log-prob term vetoes degenerate
   high-projection tokens. **Final recipe: beam search + extrinsic mean +
   λ≈0.4, one method for all ρ.**
3. **Frame composition beats score composition** (E0v2, replicated in E1) —
   the opposite of the pre-registered interference-escape hypothesis. The mean
   frame wins at low and medium ρ; z-scored score voting only draws level at
   high ρ.
4. **The Riemannian metric does not matter, but the gauge does** (E2). The true
   Karcher mean is indistinguishable from the paper's cheap chordal mean (72%
   byte-identical generations, joint-z within 0.04). But *rotation-aligning the
   frames before averaging* (generalized-Procrustes mean) roughly doubles
   per-concept expression at low ρ at lower fluency cost.
5. **A practitioner-usable selection rule with a measured crossover** (E2b):
   use the **aligned mean when ρ < 0.30**, the extrinsic mean above. Confirmed
   on 11 independent pairs, 11/11 sign-consistent (sign test p = 0.001);
   inverse-variance fit puts the crossover at ρ ≈ 0.34. Below the crossover the
   aligned mean also wins on fluency — it dominates on both axes. ρ is
   computable before generating a single token, so the rule is zero-cost at
   deployment. Scope caveat from E1b: the rule is a **greedy-decoder rescue** —
   under beam search the aligned mean's advantage vanishes (paired gap
   +0.02 ± 0.08 at low ρ vs +0.22 under greedy), so it matters when beams are
   unaffordable.
6. **Negative steering: suppression is easy, two-sided steering needs the full
   recipe** (E3 + E3b). At greedy tier both negative-steering spaces suppress
   the repelled concept far beyond the attract-only control (88%/82% of
   prompts, Wilcoxon p < 1e-15), but the repulsion drags the attracted concept
   negative in most cells. Under **frame-diff + beam + λ=0.4** the cell-mean is
   two-sided (attract up AND repel below baseline) in **all six direction
   cells** at 1.8–4.0× PPL; pooled clean rate 0.35 → 0.42 at a third of the
   fluency cost. The remaining limit is geometric: near-synonym twins
   (sorrow/sadness, ρ=0.57) still trade suppression against attraction under
   every operating point tested.
7. **Measurement was the hard part.** Exact-lemma "presence" success floors at
   ~0 even when steering visibly works; the continuous expression metric
   (post-hoc concept projection) is what made every comparison above readable.

## 3. Experiment-by-experiment results

### E0 pair selection (notebook 15) — ρ landscape

- 30 regime-tagged pairs. Regime means: similar 0.48 > hypernym 0.40 >
  antonym 0.34 > compositional 0.32 > unrelated 0.28.
- **Semantic opposition is not geometric distance**: antonyms sit mid-range;
  the most distant pairs are cross-domain (sadness/music ρ=0.187). Strata must
  be assigned by measured ρ, not intuition.
- Selected strata: low = sadness/music (0.187), medium = woman/king (0.321,
  compositional, ground truth `queen`), high = sorrow/sadness (0.566).
- Figure: `resources/15_e0_rho_by_regime.png`.

### E0 pilot (notebooks 16–17) — F1.a vs F2.a, and the metric lesson

- v1 (presence metric, k=4/steps=32): success ≈ 0 in *every* cell while
  steering was visibly working (e.g. low-ρ story set in an antique **music
  shop** with a **pianoforté**). Decision: metric v2.
- v2 (expression metric, k∈{3,4}, steps=24): signal obtained (~60% of records
  positive). **F1.a mean-frame wins at low and medium ρ at both k**; the
  F2.a−F1.a gap *rises* with ρ instead of falling → interference-escape via
  per-step score normalization is contradicted.
- k=4 roughly doubles expression gains over k=3 → adopted operating point.
- Figures: `resources/17_e0_gap_vs_rho_v2.png`.

### E2 Riemannian aggregation (notebook 18) — metric vs gauge (RQ3)

- extrinsic vs Fréchet (Karcher) vs aligned (GPA) means, 3 strata × 20 prompts.
- **Fréchet ≈ extrinsic**: ρ-to-constituents differ < 0.002; 72% byte-identical
  generations; joint-z within 0.04. The paper's cheap chordal mean is a
  near-optimal proxy for the true intrinsic mean at real concept distances —
  a positive justification of existing practice.
- **Aligned mean wins the low-ρ stratum decisively**: joint-z −0.106 vs −0.333,
  per-concept deltas roughly double (0.87/0.60 vs 0.61/0.29), at lower PPL
  (7.5× vs 8.8×). Interpretation: unrelated concepts' frames are
  rotation-mismatched in token gauge, so plain averaging partially cancels;
  aligning first preserves each constituent's signal.
- Static geometry does not predict steering: the aligned mean has the *lowest*
  ρ to its constituents yet steers best at low ρ. Evaluation must stay
  generation-based.
- Figure: `resources/18_e2_method_gap_vs_rho.png`.

### E2b ρ-adaptive crossover (notebook 19) — the headline figure

- 11 pairs spanning ρ 0.187–0.566, extrinsic vs aligned, paired per prompt.
- Gap (aligned − extrinsic) positive for all 5 pairs with ρ < 0.29, negative or
  zero for all 6 pairs with ρ ≥ 0.31. **11/11 sign-consistent, sign test
  p = 0.001**; inverse-variance weighted fit crosses zero at **ρ = 0.336**
  (raw sign flip between 0.284 and 0.306).
- The gap is not monotone: it collapses to ~0 at high ρ (related frames are
  already gauge-compatible — alignment rotates almost nothing).
- Below the crossover, aligned also wins fluency (7.5 vs 8.7× at ρ=0.187) —
  **dominant on both axes, no trade-off to argue**.
- Caveats: ρ and regime are confounded at the low end (every ρ<0.29 pair is
  "unrelated"); and composing *very* distant concepts can fail absolutely under
  either mean (dog+mathematics expresses neither) — the rule says which mean is
  better, not that distant composition works well.
- Figures: `resources/19_e2b_gap_vs_rho.png`,
  annotated version `resources/22_e2b_crossover_annotated.png`.

### E1 family sweep (notebook 20) — all families, Pareto reporting

Design: 9 methods (2 frame means, 3 score aggregators, 2 schedulers, 2 beam
variants) × 3 strata × 20 prompts, reported as success × fluency Pareto
frontiers — at 5–12× PPL no single operating point is honest.

- **True beam search is the largest effect in the entire study.** Both beam
  variants land at *positive* mean joint-z (+0.26 to +0.45) in every stratum
  while all seven greedy methods are negative (−0.11 to −0.65); per-concept
  expression deltas run 1.3–2.0 vs 0.03–0.94 for greedy — and at equal or
  *lower* fluency cost (6.3–7.5×, mid-pack or better). The plan's
  "greedy myopia" pitfall, not the choice of composition operator, was the
  binding constraint on multi-concept steering.
- **Pareto sets per stratum:** low = {beam-F1.a} alone (it dominates all eight
  other methods on both axes); medium = {F1.a-aligned, F1.a-extrinsic,
  beam-F1.a, beam-F2.a}; high = {beam-F1.a, beam-F2.a}. beam-F1.a is on the
  frontier everywhere.
- **Within greedy methods the earlier findings replicate independently:** the
  ρ-adaptive rule holds (aligned wins low: −0.11 vs −0.31; extrinsic wins
  medium: −0.18 vs −0.31); frame beats score space at low/medium; softmin
  (AND) edges both F1.a means at high ρ (−0.12 vs −0.16/−0.22) — the
  OR/AND distinction finally surfaces at k=4, as Step 4 predicted.
- **F2.c constrained is the worst method everywhere** (joint-z −0.62 to −0.65
  at the highest PPL) — lexicographic thresholding over a 4-candidate pool is
  brittle, as anticipated.
- **Schedulers are mid-pack:** passage-level composition (F3.c) is competitive
  at medium ρ but fails to express the second concept at low ρ
  (Δ music = −0.28) — alternating topics is not the same as expressing both.
- Figures: `resources/20_e1_pareto.png`,
  heatmap `resources/22_e1_method_heatmap.png`.

### E1b fluency dial + beam × aligned (notebook 21_e1b)

Design: `fluency_weight` λ added to both decoding loops — selection maximizes
cumulative concept score + λ · cumulative token log-probability. The candidate
log-probs were already computed inside the loop (the top-k is taken on them)
and merely discarded, so the blend is compute-free; λ=0 is byte-identical to
the prior behavior (golden-pinned), λ→∞ provably recovers unsteered greedy.
Sweep: 2 means × λ ∈ {0, 0.1, 0.2, 0.4, 1, 2} × 3 strata under beam search.

- **The λ-trajectory maps the entire steering↔fluency trade** and it is
  favorable: λ=0.4 keeps ~85% of beam's raw expression (e.g. medium: deltas
  1.30/1.30 vs 1.43/1.55 at λ=0) while PPL falls from 6.3–7.1× to 2.4–3.1×
  and flag shares halve. λ=1 reaches 1.3–1.6× with ~0 flags at ~50% expression.
- **Mild fluency pressure is a regularizer, not a cost:** λ=0 is
  Pareto-dominated by λ=0.1 in 4 of 6 mean×stratum cells (low extrinsic:
  joint-z +0.14 at 5.0× vs −0.13 at 6.3×). Mechanism: the log-prob term vetoes
  degenerate high-projection tokens ("flitter-ed and flutter-flutter-ed"
  becomes "a delicate key lay forgotten in a small antique music box"). This
  also closes E1's "beam can find degenerate solutions" caveat — any nonzero λ
  removes them.
- **The aligned mean does not stack with beam search:** paired per-prompt gap
  at low ρ, λ=0: +0.019 ± 0.081 (vs +0.22 under greedy in E2/E2b), and ~0 with
  no consistent sign across λ. The E2/E2b gauge results are decoder-conditional:
  alignment rescues *greedy* decoding; strong search washes the effect out.
- Sanity: λ=0 beam reproduces E1's beam-F1.a raw deltas exactly — raw deltas,
  not joint-z, are the cross-notebook currency (z-populations differ per run).
- Figure: `resources/21_e1b_lambda_dial.png`,
  overview `resources/22_fluency_overview.png`.

### E3 negative / mixed steering (notebook 22_e3)

Design: attract one concept, repel the other, both directions per stratum
(3 strata × 2 directions × 20 prompts). Methods: attract-only control
(measures incidental suppression), frame-space differential (the paper's
`A − B`, = weighted mean with weights [1, −1]), and score-space signed weights
[1, −1] with z-scoring. Two-sided success: Δ(attract) > 0 AND Δ(repel) < 0.

- **Suppression is real and strong.** Both negative methods push the repelled
  concept far below the attract-only control on the same prompts: 14–20 of 20
  prompts more suppressed in every one of the 12 (direction × method) cells,
  paired mean gaps −0.38 to −1.81; pooled: frame-diff −1.13 (88% of prompts,
  Wilcoxon p = 8×10⁻¹⁷), score-neg −1.08 (82%, p = 6×10⁻¹⁶). Frame-diff
  suppresses hardest (deepest gap in 5 of 6 directions); score-neg is cheaper
  on fluency in 5 of 6 (e.g. 6.6× vs 9.6× at high ρ).
- **Two-sided success is fragile: repulsion drags the attracted concept down
  with it.** In 8 of 12 negative-method cells Δ(attract) itself goes negative
  (e.g. +king−woman frame-diff: −0.69), while the attract-only control keeps
  Δ(attract) at +0.31…+0.85 everywhere. Clean two-sided rates peak at 60%
  (+sadness−music, frame-diff) and drop to 5–15% at medium ρ. At equal-magnitude
  weights the repel component dominates the composite — a |w_repel| < 1 weight
  sweep is the obvious next knob.
- **Direction asymmetry is large** (echoes E0 v2's per-concept asymmetry):
  +sadness−music achieves the best mixed steering (attract +0.25, repel −1.24,
  60% clean); the reverse direction collapses attraction (−0.20).
- **The near-synonym hard case fails in two distinct ways:** +sadness−sorrow
  suppresses massively (−1.46) but loses the attracted twin (−0.37);
  +sorrow−sadness keeps attraction (+0.61) but achieves zero suppression
  (+0.01). Expressing one of two geometric near-duplicates while silencing the
  other is beyond word-level composition at this operating point.
- Relevance to FRH's bias-remediation use case: that use case needs
  *suppression*, which works robustly here; it does not need the attracted
  concept to gain, which is the part that breaks.
- Figures: `resources/22_e3_quadrants.png`; table `resources/22_e3_summary.csv`.

### E3b negative steering under the final recipe (notebook 23)

Design: the E3 grid re-run under beam search, {attract-only, frame-diff,
score-neg} × λ ∈ {0, 0.4} × both directions × 3 strata (720 records), quadrant
plots overlaying the greedy tier.

- **The final recipe fixes collateral suppression.** Greedy frame-diff had
  Δ(attract) < 0 in 4 of 6 direction cells; beam alone reduces that to 2/6;
  **beam + λ=0.4 to 0/6** — two-sided cell means everywhere, at 1.8–4.0× PPL
  (vs 7–9.6× greedy). Pooled clean rate: 0.35 → 0.42 at a third of the fluency
  cost. Examples: +king−woman attract goes −0.69 → +0.16 at 2.0×;
  +sadness−sorrow −0.37 → +0.13 while still suppressing −1.12.
- **λ again improves the objective it doesn't optimize:** per-prompt clean
  rates rise or tie in 5 of 6 frame-diff directions as λ goes 0 → 0.4
  (best: +sadness−music 0.65 → 0.80), despite milder raw suppression.
- **The twin problem is geometric, not an optimizer artifact.** sorrow/sadness
  still trades suppression against attraction in both directions under the
  stronger search (+sadness−sorrow: repel −1.12 but attract only +0.13;
  +sorrow−sadness: attract +0.65 but repel −0.08). Linear differentials cannot
  separate geometric twins — now shown robust to search strength.
- **Frame-diff remains the negative-steering method of record at every tier**
  (higher net in 5/6 directions at λ=0.4; score-neg keeps one
  attract-negative cell).
- **Why repulsion is needed at all:** the attract-only control *lifts* the
  other concept in 5 of 6 cells (Δrepel +0.40…+1.46) — all pairs are
  positively correlated, so steering toward A raises B by default; the
  negative methods reverse that sign rather than merely damp it.
- Figures: `resources/23_e3_beam_quadrants.png`;
  table `resources/23_e3_beam_summary.csv`.

## 4. Measurement lessons (worth a slide)

- **Presence → expression.** Exact WordNet-lemma matching floors at zero at
  realistic generation lengths; the continuous expression metric (post-hoc
  frame projection, steered − baseline) tracks eyeballed steering and is
  method-agnostic.
- **Perplexity must be continuation-only.** Re-tokenized pad runs + chat markup
  inflate full-text PPL to garbage values (>20k on fluent text).
- **The PPL baseline is the floor.** Ratios vs the model's own greedy output run
  5–12× for every steering method at λ=0; use them for relative comparisons,
  not absolute quality claims. With the fluency dial the 2.5× guardrail became
  *reachable* rather than needing recalibration — λ≈0.4 sits at it, λ≈1 well
  inside it (E1b).
- **z-normalization is what makes score-space distinct.** Unnormalized F2.a is
  *provably* identical to F1.a (trace linearity) — every F2 number without
  normalization would silently be an F1 number.

## 5. Geometry lessons (thesis RQ3)

- Composition geometry and scoring geometry must match: an SVD joint subspace
  (Grassmannian object) scores near zero under FRH's order/sign-sensitive
  Stiefel correlation — F1.b under-steers by construction, not by accident.
- The GPA mean's fixed points form right-O(k) orbits; since FRH scoring is
  gauge-sensitive, the returned representative matters. `aligned_mean` anchors
  it at the extrinsic-mean gauge (the one E0 validated).
- ρ(mean, constituent) is symmetric and high (0.807 for joy+dog) — the mean is
  a geometric compromise; but geometric closeness does not transfer to steering
  efficacy (see E2).
- Engineering: both intrinsic means converge < 1 s at d=4096, k=3; a
  det = −1 branch in the Stiefel log produced silently-wrong tangents until
  pinned — anything downstream of a Stiefel log inherits the fix.

## 6. Limitations / open items

- ~~Beam × aligned mean is untested~~ — **resolved (E1b): it does not stack.**
  The aligned mean's low-ρ advantage is specific to greedy decoding; under
  beam search the paired gap is +0.02 ± 0.08. The ρ-adaptive rule is the
  budget-tier recommendation, not part of the final recipe.
- ~~Fluency-aware candidate selection is unexplored~~ — **resolved (E1b):**
  `fluency_weight` λ on both loops; λ≈0.4 = guardrail at ~85% expression,
  λ≈1 = near-native at ~50%; mild λ is a free win.
- ~~E3 attraction collapse at weights [1, −1]~~ — **largely resolved (E3b):**
  beam + λ=0.4 gives two-sided cell means in all six directions without
  touching the weights. A |w_repel| < 1 sweep remains the untested knob for
  the residual twin case.
- Beam search costs a k²-wide forward pass per step — it OOMs a 24 GB GPU at
  batch 20 × k=4 and had to run at batch 4; the quality/compute trade-off
  (beam width vs λ vs k) is unmeasured.
- **Geometric twins are a standing limit:** no tested operating point
  expresses one of two near-synonyms (ρ≈0.57) while suppressing the other —
  robust to search strength (E3 → E3b), so likely a property of linear
  differential steering itself.
- Single model (Llama-3.1-8B AWQ); Gemma 2 9B replication is the obvious next
  robustness check.
- Low ρ ⇔ "unrelated" confound cannot be separated at this pair count.
- Presence metric v1 numbers (E0 v1) carry no evidence either way — do not
  cite them as a negative result.
- Prompt suite is 20 short chat prompts; no claim of domain generality yet.
- The E1b/E3b λ sweep covers the three E0 pairs only; the λ≈0.4 operating
  point should be spot-checked on the wider E2b pair set before being quoted
  as universal.
