# Multi-Concept CGD — Findings Summary

Condensed from `multiconcept-cgd-findings.md` (per-step log). State as of Step 9b
(E0 pilot v2, 2026-07). Model: `Meta-Llama-3.1-8B-Instruct-AWQ-INT4`.

## What was built

Three composition families on a single unified generation loop
(`_generate_guided`, scorer seam; original `_generate_with_topk_guide` preserved
bit-exact under a golden-file regression test):

- **F1 (frame space):** `Concept.average` — weighted extrinsic (chordal/Procrustes)
  mean, generalizing the paper's differential guidance exactly
  (ρ(average([a,b], [1,−1]), a−b) = 1.0); `Concept.joint_subspace` — SVD union basis.
- **F2 (score space):** `weighted_sum`, `softmin` (AND), `constrained`
  (lexicographic), with mandatory per-step z-score/rank normalization.
- **F3 (time):** `RoundRobin`, seeded `Stochastic`, `SentenceBoundary` schedulers
  via a stateful-scorer protocol (`reset`/`observe`), per-input state.

Plus: a true beam-search variant (fixes the single-parent myopia of the original
loop), `Frame.rho` similarity, disk-cached concept construction, and an
evaluation harness (presence success, continuation-PPL fluency guardrail,
continuous expression metric, JSONL logging, per-step score traces).
68 CPU tests + GPU golden test; every step has a `playground/` demo notebook.

## Key scientific findings

1. **Linearity confound (proven).** Unnormalized score-space weighted sum is
   *identical* to scoring against the averaged frame — trace(FᵀC) is linear in C.
   Per-step normalization (z-score) is what makes F2 a distinct family; raw F2.a
   is F1.a in disguise. Every F2 experiment must set `normalize`.

2. **Composition geometry must match scoring geometry.** FRH's frame correlation
   is a Stiefel quantity (order- and sign-sensitive); an SVD joint-subspace basis
   is a Grassmannian object (arbitrary rotation of the span). Consequence: the
   joint subspace provably *contains* each constituent's span yet correlates near
   zero or negatively with them (ρ(subspace, joy) = −0.175). F1.b under-steers by
   construction unless the score is made span-aware.

3. **Semantic opposition ≠ geometric distance.** Across 30 pairs, antonyms sit
   mid-range in ρ; the most distant pairs are cross-domain (sadness/music 0.187).
   Regime means: similar 0.478 > hypernym 0.401 > antonym 0.339 > compositional
   0.323 > unrelated 0.284. All ρ are positive and compressed (~0.19–0.57,
   anisotropy) — strata must be assigned by measured ρ, not intuition.

4. **The interference-escape hypothesis is contradicted at pilot scale (E0 v2).**
   Prediction: score-space F2.a escapes frame-averaging interference at low ρ.
   Observed: **F1.a mean-frame wins at low and medium ρ at both k** (k=4 joint-z:
   −0.10 vs −0.34 low; −0.01 vs −0.15 medium); F2.a edges ahead only at high ρ
   with k=3, and the F2−F1 gap *rises* with ρ instead of falling. Frame-space
   composition is the stronger baseline → **RQ3's Riemannian aggregation (E2) is
   now the highest-value experiment**, since the winning family is the one whose
   aggregation method RQ3 studies.

5. **Chordal-mean composition is promising but pair-dependent (RQ3 preview).**
   ρ(mean(woman, child), girl) = 0.488 exceeds both constituents, but
   ρ(mean(woman, king), queen) = 0.515 falls slightly below king alone (0.531).

6. **Cascade reranking was structurally toothless** — the original loop keeps one
   surviving parent per step, so "beams" differ only in their last token; a final
   rerank could only ever choose one token. Removed; true beam search added.

## Measurement lessons

- **Presence-success floors at 0** even when steering visibly works (topic shifts
  without exact member lemmas). Metric v2 — mean post-hoc `fur.project` of the
  continuation, delta vs unsteered baseline — has real signal (~60% positive
  deltas) and is the E1 metric.
- **Continuation-only PPL** is mandatory: full decoded texts contain left-padding
  and chat markup that inflate baseline PPL to >20k.
- **Fluency is the open cost problem:** 5–12× PPL ratios, 90–100% of runs over the
  2.5× guardrail at current settings. E1 must report success × fluency Pareto
  frontiers; F2.a is consistently more expensive than F1.a.
- **k=4 roughly doubles expression gains over k=3** at comparable relative cost —
  recommended E1 operating point.
- Batch-order effect is exactly zero; AWQ decoding is deterministic but
  hardware-local (golden file transfers across identical A5000s only).

## Open questions / next steps

1. **E2 (RQ3):** Riemannian (geodesic Fréchet) mean as a drop-in for
   `Concept.average` — elevated priority by finding 4.
2. **F2.b/F2.c check:** softmin and constrained were never run at E0 settings;
   cheap sanity before concluding score space loses outright.
3. **E1 family sweep** at k=4 with Pareto reporting, including the beam variant.
4. Fluency-aware candidate selection; PPL guardrail calibration.
5. **E3:** negative/mixed steering — score-space w=[1,−1] and frame-space
   differential produce visibly different texts, so the comparison is non-trivial.

## Artifact map

| What | Where |
|---|---|
| Per-step findings log | `plans/multiconcept-cgd-findings.md` |
| Implementation plan | `plans/multiconcept-cgd-implementation-plan.md` |
| Golden reference | `tests/golden/single_concept.json`, `14_golden_reference.ipynb` |
| Pair selection | `15_e0_pair_selection.ipynb`, `resources/15_e0_selected_pairs.json` |
| E0 pilot v1 / v2 | `16_e0_pilot.ipynb` / `17_e0_pilot_v2.ipynb`, `resources/1{6,7}_e0_*` |
| Demos | `playground/stepNN_*.ipynb` |
