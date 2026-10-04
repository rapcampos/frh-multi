# Metrifying multi-concept Top-k CGD — literature review and recommendations

Deep-research pass over the activation-steering, controllable-text-generation (CTG), and
LLM-judge evaluation literature (22 primary sources; claims adversarially verified where
marked). Question: how should we evaluate multi-concept Top-k Concept-Guided Decoding —
steering success, fluency, and side effects — and how do the three proposed metrics
(LLM-as-a-judge, cross-model embedding comparison with Qwen3, KL divergence) hold up?

**Confidence notation.** *verified* = quote re-checked against the source text by an
adversarial pass; *corroborated* = extracted with a supporting quote but not re-checked.

---

## TL;DR

1. **LLM-as-a-judge is the field's standard for steering success and the strongest of the
   three proposals.** Adopt the AxBench protocol: pointwise 0/1/2 rubric per axis (one
   concept score *per target concept*, plus instruction-relevance and fluency), aggregated
   by harmonic mean so failure on any axis collapses the score. Validate the judge once
   with a dual-judge reliability study (ICC + per-judge z-normalization), then run one
   judge at scale.
2. **The Qwen3-embedding idea is validated by the literature, but aim it differently.**
   Embedding similarity between the *continuation and the concept* (gloss + member lemmas)
   is a near-perfect proxy for judge concept scores (Spearman ρ = 0.97). Raw steered-vs-
   unsteered similarity, by contrast, is confounded: it cannot distinguish intended
   concept shift from unintended content damage. Use concept-anchored similarity as the
   bulk success proxy; use steered-vs-unsteered similarity only as an off-target drift
   measure.
3. **KL divergence is a side-effect metric, not a success metric.** Precedents use KL to
   the unsteered model on *benign/off-target* inputs (KL-then-steer) or as a per-step
   control signal (Dynamic Activation Composition). For Top-k CGD — which never modifies
   the model's distribution, only the survivor rule — the per-step exact KL between
   next-token distributions under the steered vs unsteered prefix is computable without
   sampling and measures trajectory displacement. The existing PPL-ratio is the cheap
   sequence-level cousin; keep it, but stop calling it a fluency metric (see §4.2).
4. **Drop the assumption that perplexity measures fluency.** External-model perplexity is
   *uncorrelated* with judge-rated fluency in steered text (Spearman ρ ≈ 0), while simple
   Type-Token Ratio predicts it at ρ = 0.81 (*verified source, corroborated numbers*).
   Add TTR and dist-n to the harness (trivial), keep PPL-ratio as a distribution-shift
   indicator.
5. **Report per-example distributions and sweeps, not means at one operating point.**
   Half the field's cautionary results are about aggregates hiding failure: ~⅓–½ of
   inputs can be *anti-steerable* (steered opposite to intent), and fluency collapses
   non-monotonically past a critical control strength ("attribute collapse"). The repo's
   λ- and k-sweeps are exactly right; add the anti-steerable fraction as a headline
   number.
6. **For multi-concept, the field has a gap we can fill.** The main multi-aspect CTG
   benchmark (CompMCTG) *averages* per-aspect accuracies — no strict joint-satisfaction
   metric. The steering-vector composition paper closest to our setting reports ≥15-point
   trait-expression drops when composing just two vectors and negative/no expression at
   four (*verified*). Report the per-concept score vector, a strict joint metric (all
   concepts judged present), and a degradation curve vs number of concepts.

---

## 1. Why the current battery needs external metrics

The harness (`frames/evaluation/harness.py`) currently measures success with v1 lemma
presence, v2 frame-projection expression, and v3 gauge-invariant angles, plus a PPL ratio
against the unsteered baseline. v2/v3 are computed **with the same model and the same
concept frames that define the steering objective**. Since the CGD loop selects tokens to
maximize frame correlation, frame-based success metrics partially measure "did the
optimizer optimize" rather than "did the text express the concept". The steered-vs-
baseline delta mitigates but does not break this circularity.

The FRH paper itself flags every gap this report addresses: substring compliance matching
is "brittle and potentially over-strict", helpfulness of steered outputs is unassessed,
and interpretability metrics risk being "naively maximized by yet another black box
model" (FRH §6, Limitations). The field has since converged on exactly the external
instruments proposed here. A 2026 systematic comparison of conditioning methods found
that efficient steering methods "frequently achieve conditioning at a steep cost to
fluency" (*verified quote, arXiv:2606.12234*) — effectiveness must be reported jointly
with quality, by instruments independent of the steering objective.

The internal metrics stay valuable — as *mechanism* diagnostics (did the loop move the
representation?) rather than *success* claims (did the text change meaningfully?). The
recommended battery keeps them and adds an external tier on top.

---

## 2. Assessment of the three proposals

### 2.1 LLM-as-a-judge — adopt as the primary success metric

This is now the de-facto standard across the steering literature:

- **AxBench** (arXiv:2501.17148), the standard steering benchmark, has an LLM judge rate
  each generation 0/1/2 on three axes — concept incorporation, instruction relevance,
  fluency — and aggregates by **harmonic mean** (range 0–2), which "heavily penalises
  poor performance on any of these three subscores" (*verified 3-0*). This kills the two
  classic gaming modes: fluent text without the concept, and concept-stuffed gibberish.
- A steering-methods comparison (arXiv:2606.12234) uses the same 0/1/2 three-criterion
  design with an **open-weight judge** (OLMo2-32B-IT) (*verified 3-0*) — precedent if API
  judges are a cost or reproducibility concern.
- The composition study closest to our multi-concept setting (arXiv:2607.01802,
  Subbiah, Hall & McKeown, "On the Limits of Steering Vectors for Preference-Aligned
  Generation") evaluates **entirely** with LLM judges: a 5-point trait-expression judge
  (PPCM, from PLUME), a 0–100 trait judge (from Persona Vectors), and a 0–100 coherence
  judge with acceptance threshold 75 (*verified*).
- MT-Bench (arXiv:2306.05685) grounds the practice: GPT-4 agreement with humans reaches
  85%, above the 81% human–human agreement (*corroborated*).

**Bias mitigations, concretely.** The known judge biases (position, verbosity,
self-preference) are mostly pairwise-comparison pathologies; one corpus source reports
pairwise evaluation *amplifies* bias relative to pointwise scoring (adversarial LLMBar
results; *corroborated, unverified*). Mitigations for our protocol:

- **Pointwise, not pairwise**: score each continuation independently against a rubric
  (position bias becomes moot).
- **Judge from a different model family** than the steered model (Llama/Gemma/Phi are
  steered → GPT-, Gemini-, Claude-, or OLMo-family judge avoids self-preference).
- **Blind the judge** to method, λ, k, and steered/baseline status.
- **Validate once with two judges**: the mechanistic-indicators paper (arXiv:2602.01716,
  Jafari, Xue & Salim) ran ChatGPT-4o-mini and Gemini-Flash-2.5 over 72 steering
  conditions: ICC(3,1) = 0.78, 95% CI [0.67, 0.86] — but raw-score Krippendorff's
  α was only **0.23** despite Pearson r = 0.84, because Gemini scored systematically
  higher (mean 0.40 vs 0.22); **per-judge z-score normalization raised α to 0.85**
  (*verified — all numbers exact*). Lesson: judges agree on rankings, not calibration.
  Never mix raw scores from different judges; z-normalize per judge, and report ICC or
  normalized α from a ~100-item dual-judged subsample.
- **Fixed rubric anchors** (0 = absent, 1 = weakly/awkwardly present, 2 = naturally
  incorporated) — the discrete 3-point scale is deliberately coarse to be reliable.

**Cost control.** The ρ = 0.97 embedding-proxy result (§2.2) means the judge only needs
to run on a subsample (a few hundred continuations per experiment) plus every case where
the cheap proxy and the internal metrics disagree.

### 2.2 Cross-model embedding comparison (Qwen3) — adopt, re-aimed

The literature validates third-party-embedding evaluation, but the target of the
comparison matters more than the embedder:

- **Concept-anchored similarity works.** Sentence-BERT cosine between the *concept* and
  the *continuation* tracks LLM-judge concept-injection/removal scores at Spearman
  **ρ = 0.97/0.96** (arXiv:2606.12234, *verified 2-1*). This is the validated shape of
  the metric: `cos(emb(continuation), emb(concept text))`, where the concept text can be
  the WordNet gloss plus member lemmas — richer than v1's exact-lemma matching and
  external to the steered model's geometry.
- **Content-preservation precedent.** Style-transfer evaluation moved from n-gram overlap
  to embedding similarity a while ago: STRAP (arXiv:2010.05700) uses the SIM subword-
  embedding model for semantic preservation; Mir et al. (arXiv:1904.02295) show
  embedding-family metrics beat BLEU for content preservation (*corroborated*).
- **The confound to avoid.** A recent critique (arXiv:2502.15022, *corroborated*) argues
  embedding-similarity metrics are structurally unsuitable as a single steered-vs-
  unsteered score: **the larger the intended attribute shift, the lower the similarity**,
  so success and damage are conflated. A low Qwen3 cosine between steered and unsteered
  continuations could mean excellent steering or destroyed content — the number alone
  cannot say which.

So the two embedding metrics to implement, both with Qwen3-Embedding (a strong,
model-family-independent open embedder; 0.6B is plenty for bulk scoring):

1. **Concept expression (external)**: `cos(emb(continuation), emb(gloss+lemmas))` per
   target concept, reported as steered-minus-baseline delta — mirrors the harness's
   `expression_record` structure exactly, so it drops into the JSONL schema as an
   `external_expression` sibling.
2. **Off-target drift**: steered-vs-unsteered continuation similarity **on prompts/
   concepts the steering should not touch** — a side-effect measure, where low similarity
   is unambiguously bad.

One caveat: for steered Qwen2-VL runs, prefer a non-Qwen embedder (or report both) to
keep the cross-family independence argument clean.

### 2.3 KL divergence — adopt as a side-effect trace, not a success metric

How the field actually uses KL between steered and unsteered models:

- **KL-then-steer** (arXiv:2406.15518) minimizes `E[KL(steered ‖ unsteered)]` over
  per-token next-token distributions **on benign inputs** — as a *training objective* to
  suppress side effects (*corroborated, two independent extractions agree*). The framing
  transfers: KL to the unsteered model on inputs where steering shouldn't matter is the
  canonical capability-preservation quantity.
- **Dynamic Activation Composition** (arXiv:2406.12319) computes per-step KL between
  unsteered and steered next-token distributions (nucleus-truncated shared support,
  capped at 2) and uses it as the steering-intensity signal — notably in a
  *multi-property* steering setting (*corroborated*). Directly relevant precedent: per-
  step, per-concept KL traces are informative enough to drive adaptation.
- **Pitfall**: sequence-level KL between two LMs is intractable exactly, and the naive
  Monte-Carlo estimator is high-variance and can go negative (*corroborated*,
  arXiv:2504.10637). Avoid sampled sequence-level KL entirely.
- The mechanistic-indicators paper (arXiv:2602.01716) uses a different granularity — a
  logit-lens KL *difference* in vocabulary space, `KL(p_t‖q) − KL(p̂_t‖q)` against the
  concept-induced distribution q (*verified, with correction: both steered and unsteered
  residuals enter*) — and shows internal signals (α, branching factor, KL statistics)
  predict judge scores at R² ≈ 0.47–0.54 (*verified*). Useful ammunition for the thesis:
  internal metrics correlate with, but do not replace, external judgment.

**The CGD-specific subtlety.** Top-k CGD never modifies the model's next-token
distribution — it changes only which candidate survives. "KL between steered and
unsteered model" therefore reduces to **KL between next-token distributions conditioned
on the diverging prefixes**, `KL(p(·|steered prefix) ‖ p(·|unsteered prefix))` per step.
Both distributions are exactly available during generation (full softmax each step), so
no estimator issues arise. Interpretation: how far the survivor rule has displaced the
model from its natural trajectory. Two uses:

1. **Per-step trace** (RecordingScorer-style hook): displacement-vs-step curves per
   method family — the distribution-level cousin of the paper's total-projection curves.
2. **Off-target form**: on prompts unrelated to the target concepts, steered-vs-unsteered
   KL should be ≈ 0 for a well-behaved method. This is the honest "capability
   preservation" number.

Note the sequence-level shortcut already in the harness: PPL-ratio of the steered
continuation under the unsteered model is the cumulative log-prob displacement — the
cheap integral of (1). Keep it, but reframe per §4.2.

---

## 3. What the literature battery looks like, per axis

### 3.1 Steering success

| Instrument | Used by | Notes |
|---|---|---|
| LLM judge, 0/1/2 per axis, harmonic mean | AxBench (*verified*), 2606.12234 (*verified*) | current standard |
| External attribute classifier over N samples/prompt | DExperts (ACL 2021), CompMCTG, Gu et al. (*corroborated*) | needs a classifier per concept — impractical for 100k WordNet concepts; judge/embedding generalizes |
| Concept-anchored embedding cosine | 2606.12234 (*verified*, ρ=0.97 vs judge) | our bulk proxy |
| Logit-difference propensity + steerability slope over multipliers | Tan et al. (arXiv:2407.12404, *verified*), arXiv:2505.22637 | A/B-format tasks; the *slope* idea transfers to our λ/k sweeps |
| Presence/substring matching | FRH paper (compliance regex) | brittle; keep only as v1 lower bound |

### 3.2 Fluency / quality

- External-LM perplexity is the historical standard (DExperts: GPT-2 XL over 25
  generations/prompt, *corroborated*) — but the 2026 evidence says it fails for steered
  text: **Spearman ρ ≈ 0 with judge fluency**, while **Type-Token Ratio hits 0.81**
  (arXiv:2606.12234, *verified 3-0*). CTRLEval had already reported PPL/dist-n correlate
  weakly with human judgment (*corroborated*).
- Degeneration metrics (TTR, dist-1/2/3) directly catch the FRH paper's own documented
  failure mode — repetition loops at high k (FRH Appendix B.1).
- Judge fluency 0/1/2 on the subsample anchors the cheap metrics.

### 3.3 Side effects / capability preservation

- KL to unsteered model on benign inputs (KL-then-steer, *corroborated*).
- Off-target concept drift: expression change on non-target concepts. The harness
  already computes per-concept expression for arbitrary concept dicts — score steered
  output against a fixed panel of *unrelated* probe concepts and report the delta
  (should be ≈ 0).
- Benchmark degradation (MMLU-style) appears in the steering literature but is likely
  out of thesis scope; the off-target KL + drift pair covers the same intent cheaply.

### 3.4 Reporting protocol (the meta-findings)

- **Per-example distributions, not means.** On several datasets almost half of inputs
  are *anti-steerable* — steering moves the model the wrong way (Tan et al.,
  *verified 3-0*; ~⅓ in arXiv:2505.22637, *corroborated*). Report
  `P(per-example success delta < 0)` alongside means.
- **Sweep control strength; report curves.** Air-Decoding names the failure "attribute
  collapse": past a critical strength, fluency collapses rapidly and non-monotonically
  (*corroborated*). DExperts picks its operating point from an explicit
  control-vs-fluency curve (*corroborated*). The repo's λ- and k-sweep notebooks
  already implement this; the Pareto framing should be the headline figure, not a table
  at one operating point.
- **Control prompt-format confounds** (Tan et al., *verified*): vary prompt templates
  when measuring susceptibility.
- **Four requirements for reliable steering evaluation** (arXiv:2410.17245,
  *corroborated*): downstream-like contexts (open-ended generation, not just probes),
  likelihood-aware measurement, cross-behavior comparability (standardized scales), and
  explicit baselines. The harness's steered-vs-baseline delta design already satisfies
  the last two.

### 3.5 Multi-concept specifics

- **The field's benchmarks don't measure joint satisfaction.** CompMCTG (ACL 2024)
  *averages* per-aspect classifier accuracies (*corroborated — explicitly noted as no
  strict joint metric*); Gu et al. likewise report the per-aspect mean. A strict joint
  metric (all concepts simultaneously present) is a contribution opportunity, not
  established practice to copy.
- **Composition degrades measurably.** Composing two steering vectors drops trait
  expression by ≥15 points (0–100 scale) across all four composition methods tested on
  both Qwen2.5-7B and Llama3.1-8B; at four traits most methods reach negative or no
  expression (arXiv:2607.01802, *verified — note the paper loosely calls the point drop
  a "% reduction"*). Their protocol — per-trait expression tracked as traits are added,
  with a coherence gate (threshold 75/100) — is the closest published template for our
  multi-concept evaluation.
- Recommended multi-concept report per cell: per-concept score vector; strict joint
  success (all concepts ≥1 by judge, or all concept-similarities above a calibrated
  threshold); min-aggregate (harmonic mean or min — both punish dropping a concept);
  and the degradation curve success-vs-number-of-concepts (1→2→3→4).

---

## 4. Recommended battery v4, wired to the existing harness

### 4.1 Add (new external tier)

| Metric | Definition | Where it lives |
|---|---|---|
| Judge scores | 0/1/2 per target concept + fluency + instruction-relevance; harmonic mean; pointwise; blind; different family from steered model | offline pass over JSONL (subsample + disagreements); one-time dual-judge reliability study with z-normalization, report ICC/α |
| External concept expression | Qwen3-Embedding cosine: continuation vs gloss+lemmas, steered−baseline delta per concept | new `external_expression` field mirroring `expression_record` |
| Off-target drift | (a) expression delta on a fixed panel of unrelated probe concepts; (b) Qwen3 similarity steered-vs-baseline on off-target prompts | reuse `expression_record` with a probe-concept dict; small off-target prompt set |
| Degeneration | TTR + dist-1/2/3 of the continuation | trivial pure-Python addition to `evaluate()` |
| Per-step KL trace (optional) | `KL(p(·|steered prefix) ‖ p(·|unsteered prefix))` per step; off-target form ≈ 0 check | RecordingScorer-style hook; needs one extra baseline forward per step, so keep optional |

### 4.2 Reframe (existing metrics, new labels)

- **PPL-ratio** → distribution-shift indicator, not fluency ("fluency_flag" is
  mislabeled given the ρ ≈ 0 result). Keep the 2.5× threshold as a shift alarm.
- **v2 expression / v3 angles** → internal alignment (mechanism) metrics. Strong for
  method development and traces; not citable as success evidence on their own — pair
  with the external tier in any writeup. The R² ≈ 0.5 result (internal signals predict
  only half the variance in judge scores) is the citable justification.
- **v1 presence** → cheap lower bound; already superseded by E0's finding that lemmas
  rarely surface even when steering works.

### 4.3 Report

- Per-example distributions + anti-steerable fraction per cell.
- λ/k Pareto curves (success axis = judge or external expression; quality axis = TTR/
  judge fluency) as the headline figures.
- Multi-concept: per-concept vector, strict joint, min-aggregate, degradation curve.

---

## 5. Sources

Verified in this pass (adversarial re-check against source text):
- arXiv:2501.17148 — AxBench (judge protocol, harmonic mean)
- arXiv:2407.12404 — Tan et al., steering-vector generalization/reliability (steerability slope, anti-steerable, prompt confounds)
- arXiv:2606.12234 — steering-methods comparison (OLMo2 judge; PPL↮fluency; TTR 0.81; SBERT proxy ρ=0.97)
- arXiv:2602.01716 — Jafari, Xue & Salim, mechanistic indicators (ICC 0.78; α 0.23→0.85 after z-norm; logit-lens KL-difference; RF R²≈0.47–0.54)
- arXiv:2607.01802 — Subbiah, Hall & McKeown, limits of steering-vector composition (≥15-point drop at 2 vectors; negative at 4; judge-only eval, coherence threshold 75)

Corroborated (quote-backed, not re-checked):
- DExperts (ACL 2021, aclanthology 2021.acl-long.522) · CTRLEval (arXiv:2204.00862) ·
  Air-Decoding (arXiv:2310.14892) · CTG survey (arXiv:2408.12599) · CompMCTG
  (arXiv:2404.04232) · Gu et al. multi-aspect (arXiv:2210.02889, 2405.07875) · MT-Bench
  (arXiv:2306.05685) · reliable-steering-eval properties (arXiv:2410.17245) ·
  KL-then-steer (arXiv:2406.15518) · Dynamic Activation Composition (arXiv:2406.12319) ·
  KL-estimation pitfalls (arXiv:2504.10637) · STRAP/SIM (arXiv:2010.05700) · Mir et al.
  style-transfer eval (arXiv:1904.02295) · embedding-metric critique (arXiv:2502.15022) ·
  pointwise-vs-pairwise judge bias (corpus source, unverified attribution).
