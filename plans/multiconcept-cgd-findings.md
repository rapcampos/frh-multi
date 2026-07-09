# Multi-Concept CGD — Findings Log

Companion to `multiconcept-cgd-implementation-plan.md`: what each completed step
built, what its gates showed, and what we learned. Updated as steps complete.

| Step | Deliverable | Status |
|---|---|---|
| 1 | Golden file + pytest infra | ✅ gate passed (2× consecutive) |
| 2 | Unified loop, cascade removed | ✅ golden token-for-token |
| 2b | True beam-search variant | ✅ k=1 degeneracy + CPU tests |
| 3 | Concept-pair toolkit + ρ | ✅ ρ-ordering sanity passed |
| 4 | Family 2 aggregators + normalization | ✅ 19 CPU tests + golden re-verified |
| 5 | Family 1: weighted mean + true joint subspace | ✅ 12 CPU tests + golden re-verified |
| 6 | Family 3 schedulers | ✅ 10 CPU tests + seeded-reproducibility assert |
| 7 | Evaluation harness | ✅ 11 CPU tests + gate on golden prompts |
| 8 | E0 pair selection | ✅ 3 strata selected, dissociation confirmed |
| 9 | E0 pilot matrix (v1) | ⚠️ presence metric floored at 0 — prompted metric v2 |
| 9b | E0 pilot v2 (expression metric, k∈{3,4}) | ✅ signal obtained; interference hypothesis contradicted — F1.a wins at low/medium ρ |
| 10 | E2: Riemannian aggregation (RQ3) | ✅ frechet ≈ extrinsic (43/60 identical texts); aligned mean wins at low ρ |

---

## Step 1 — Golden file

**Built:** `14_golden_reference.ipynb`, `tests/golden/single_concept.json`,
`tests/test_golden.py`, pytest infra (`gpu` marker, GPU tests deselected by default).
Pins `generate_with_topk_guide`: 10 chat prompts × 3 guides (`dog.n.01`, `joy.n.01`,
`woman.n.01 − man.n.01`), k=3, steps=16, `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` on one device.

**Findings:**
- Decoding is deterministic greedy (no seed needed), but the golden file is only valid
  for the same batch composition and GPU/driver — AWQ inference is hardware-local.
  It transfers across the machine's identical RTX A5000s (verified: created on GPU 0,
  passes on GPU 1 via `CUDA_VISIBLE_DEVICES`).
- Steering is visibly working in the reference outputs (joy → "Laugh with loved one…").

## Step 2 — Unified generation loop, cascade removed

**Built:** `_generate_guided(input_text, concepts, weights, k, steps, scorer)` — one
shared loop with a swappable scorer seam; original method and multi-guide are thin
wrappers. `_project_hidden_states` dedupes the pad/unfold/correlate block that existed
in four copies. Removed ~200 lines: both cascade implementations + wrappers, dead
`_select_best_candidates`/`_total_probe`.

**Findings:**
- **Cascade was structurally toothless** (plan Key finding 2): the original loop keeps a
  single surviving parent per step, so the k "beams" always share every token except the
  last — a final rerank by a second concept could only ever choose the last token. Hence
  removal rather than repair.
- Wrapper equivalence is bit-exact: multiplying projections by weight 1.0 and summing over
  a singleton concept dimension is IEEE-exact, confirmed by the golden test after refactor.
- `12_multiconcept_guided_generation.ipynb` still references the removed cascade method in
  one cell; its cached baseline results in `resources/` are untouched. Rerunning it
  requires skipping/removing that cell.
- Even a naive hand-written hard-min scorer (playground §4) already produced different
  generations than weighted-sum — the scorer seam is doing real work.

## Step 2b — True beam-search variant

**Built:** `generate_with_topk_beam_guide` (+ quick wrapper): at each step all pool
candidates compete and the top-k *sequences* survive, instead of one parent's k children.
Selection logic isolated in `_topk_beam_selection` (5 CPU unit tests). Same
concepts/weights/scorer interface as the unified loop. Original method untouched.

**Findings:**
- Beams genuinely diverge; cost is a k²-wide forward pass per step after the first
  (watch memory at large k × batch_size).
- No length normalization is needed: all beams share the same length by construction
  (padded prompts, one token per step). EOS handling mirrors the original.
- **k=1 degeneracy verified on the model:** original and beam produce byte-identical
  greedy output at k=1 — pins the pool bookkeeping (reshapes/indexing) to a case with
  known ground truth.
- Ops note: with a Jupyter kernel holding GPU 0, verification runs are routed to free
  GPUs via `CUDA_VISIBLE_DEVICES`; golden results transfer across the identical A5000s.

## Step 3 — Concept-pair toolkit + ρ

**Built:** `Frame.rho(other)` (rank-neutral zero-padding to a common k — codifies the
codebase's unequal-rank policy, relevant to thesis RQ3), `get_concept_cached` (disk cache
in `cache/concepts/`, keyed by model + synset + filter args + language set),
`concept_pair(a, b) → (Concept, Concept, ρ)`. 7 CPU unit tests.

**Findings (from the 22-synset survey, Llama-3.1-8B-AWQ, min_lemmas=11, max_tokens=3):**
- Sanity gate passed: ρ(woman, man) = 0.418 > ρ(woman, dog) = 0.318.
- **Antonyms are NOT geometrically distant** — as suspected in the Step 8 taxonomy
  discussion. The lowest-ρ pairs are all cross-domain (`sadness`/`music` 0.187,
  `puppy`/`mathematics` 0.242); antonym pairs sit mid-range. Near-synonyms rank top
  (`sorrow`/`sadness` 0.566, `joy`/`happiness` 0.493), then hypernyms (`dog`/`puppy` 0.467).
  ⇒ E0 strata must be assigned by measured ρ, not semantic intuition, and a
  ρ-vs-regime dissociation is itself a reportable finding.
- **All ρ values are positive and compressed** (≈0.19–0.57 in this sample): low/medium/high
  strata will be relative bands, not sign-based.
- **Composition signal (positive RQ3 preview):** ρ(mean(woman, child), girl) = 0.488,
  *higher than either constituent* (0.402 / 0.466) — the chordal mean moves toward the
  semantic composition before any generation.
- `child.n.01` correlates suspiciously well with everything (0.43–0.47 even vs `dog`,
  `cat`) — likely a lemma-richness/rank artifact (a "hub" concept). Exactly the
  scale-incommensurability issue Step 4's normalization addresses; avoid hub concepts
  or normalize when selecting E0 pairs.
- Filter coverage is uneven: 22/23 candidates survive, but `boy.n.01` does not (while
  `girl.n.01` does). Always check survival before planning a pair.

## Step 4 — Family 2 aggregators + mandatory normalization

**Built:** `frames/representations/aggregators.py`: `weighted_sum` (F2.a, signed weights
repel), `softmin` (F2.b, AND, τ→0 = hard min), `constrained` (F2.c, lexicographic),
`normalize_scores` (z-score / rank over the candidate pool), scorer factories. `normalize=`
flag wired into both loops + multi-guide (default None = golden path). 19 CPU unit tests.

**Findings:**
- **The linearity confound is proven and visible.** Numerically: unnormalized weighted-sum
  scores exactly equal scoring against the single merged frame (trace linearity), and
  z-scoring provably flips the winner when one concept's scale dominates
  (`TestLinearityConfound`). In generation: F1.a mean-frame and raw F2.a produced
  near-identical continuations on the test prompts, while `normalize="zscore"` diverged.
  ⇒ Every E0/E1 F2 run must set `normalize`; raw F2.a is F1.a in disguise.
- **OR vs AND saturate at small k:** weighted-sum and softmin (τ=0.5) chose identical
  outputs at k=3 on the test prompts — with a 3-candidate pool both often agree on the
  argmax. Expect the distinction to surface at larger k, smaller τ, or lower-ρ concept
  pairs; treat k as an experimental variable for aggregator comparisons.
- **Negative steering differs between spaces:** score-space `w=[1,−1]` and frame-space
  differential (`joy − dog`) produced clearly different texts — E3's comparison is
  non-trivial.
- Implementation notes: z-score maps constant scores to 0 (no NaN on shared beam
  prefixes); `constrained` needs a finite penalty inside the loop because scores are
  cumulatively summed over positions (−inf would poison the suffix).

## Step 5 — Family 1: weighted mean + true joint subspace

**Built:** `Concept.average(concepts, weights=None)` — weighted extrinsic (chordal/
Procrustes) mean with unequal-rank handling; trailing all-zero columns of the weighted
sum are excluded from the polar decomposition (their factor is arbitrary) and restored
as zeros, so `weights=[1,0]` no longer risks garbage columns. Docstring fixed: extrinsic
mean, NOT geodesic Fréchet (RQ3 drop-in point). `Concept.joint_subspace(concepts, rtol)`
— real F1.b: SVD basis of the union of spans, rank-truncated. Renames:
`quick_generate_with_topk_mean_frame_guide` (honest F1.a name, supports `guide_weights`)
and `quick_generate_with_topk_subspace_guide` (now actually F1.b). 12 CPU unit tests.

**Findings:**
- **Joint-subspace composition is geometrically correct but mismatched with FRH's
  scoring.** The subspace provably contains each constituent's span (unit-tested via
  projection norms), yet its trace-based correlation with its own constituents is
  near zero or NEGATIVE: ρ(subspace, joy) = −0.175, ρ(subspace, dog) = −0.133, and
  ρ(subspace(woman, child), girl) = 0.034 vs 0.488 for the mean. Cause: FRH's frame
  correlation is a *Stiefel* quantity — it depends on vector order and sign — while an
  SVD basis is an arbitrary rotation of the span (a *Grassmannian* object). OR-semantics
  via subspaces needs a span-aware score (e.g. projection-norm, or per-candidate
  Procrustes alignment before the trace); with the current correlation, F1.b will
  under-steer and its E1 numbers must be interpreted accordingly. Thesis-relevant:
  composition geometry and scoring geometry must match.
- **The mean is a perfect compromise geometrically:** ρ(mean, joy) = ρ(mean, dog)
  = 0.807 (from ρ(joy, dog) = 0.334) — symmetric by construction, and much closer to
  both constituents than they are to each other.
- **Weighted average generalizes the paper's differential guidance exactly:**
  ρ(average([joy, dog], weights=[1, −1]), joy − dog) = 1.0000. Frame-space negative
  steering and `Concept.__sub__` are the same operation; the weighted mean is the
  continuous family containing it.
- Weight sweeps visibly shift generation content (joy-heavy → cheerful village tale;
  dog-heavy → different lexical field), giving F1.a a tunable knob for E1 Pareto sweeps.
- Subspace rank adds up as expected (joy k=3 + dog k=3 → rank 6, minus overlaps), which
  also inflates the correlation denominator √(rank·rank) — a second, mundane reason
  subspace scores run low.

## Step 6 — Family 3 schedulers

**Built:** `frames/representations/schedulers.py`: `RoundRobin` (F3.a), `Stochastic`
(F3.b, seeded `torch.Generator`, `reset` re-seeds → identical schedule per generation
call), `SentenceBoundary` (F3.c, punctuation heuristic on each input's representative
beam, per-input state). Stateful-scorer protocol in both loops: `reset(n)` at generation
start, `observe(tokens, n)` per step — both `hasattr`-guarded, so plain function scorers
(incl. the golden path) are untouched. 10 CPU unit tests.

**Findings:**
- Per-input scheduler state works because both loops keep candidate rows grouped by
  input through every reshape — different prompts can be on different concepts at the
  same step (verified in unit tests).
- Family 3 semantics decision: schedulers IGNORE the loop's weights tensor; Stochastic
  takes probabilities at construction. One concept at a time is the definition.
- The three families now produce visibly distinct generations from identical concepts
  and budget (playground §4): F2.a weaves both concepts per step, F3.a alternates
  mid-sentence, F3.c switches topic per sentence. Qualitatively, F3.c reads most
  fluent — consistent with the co-occurrence hypothesis (passage-level composition is
  linguistically easier than word-level).
- Seeded stochastic scheduling is byte-reproducible end-to-end (asserted in the
  notebook on real generations, not just the schedule).

## Step 7 — Evaluation harness

**Built:** `frames/evaluation/harness.py`: `EvaluationHarness` (decoupled from
generation — evaluates any (prompts, texts) pair; presence success from WordNet member
lemmas matched in the continuation only; continuation-PPL vs unsteered greedy baseline;
fluency flag at ratio > 2.5; JSONL log with config verbatim) and `RecordingScorer`
(wraps any scorer/scheduler, logs per-step per-concept + aggregated scores of each
input's representative beam). 11 CPU tests over a FakeFur stub.

**Findings (first quantitative pass — golden prompts, joy+dog, k=3, steps=16):**
- **Two measurement bugs caught by the gate itself:** (1) `get_all_synsets` drops the
  lemma column (groupby keeps only padded tokens) — member lemmas must come from
  `get_dataframe`; (2) full-text PPL is garbage: batched generation left-pads shorter
  prompts, and re-tokenized pad runs + chat markup produced baseline PPL up to 21,843
  on perfectly fluent text. PPL is now computed on the continuation only (unconditional
  on the prompt — a documented v1 simplification).
- **Presence-success at k=3/steps=16 is near zero for every family** (joy 0–10%,
  dog 10%, both 0%). Sixteen tokens and a 3-candidate pool rarely surface an exact
  member lemma even when steering visibly shifts topic. Implications for E0: longer
  generations and/or larger k, and the plan's fallback ("classifier if presence proves
  too crude") is likely to be needed — or presence should count stemmed/partial matches.
- **PPL ratios run ~4× with 8–9/10 runs flagged** at the 2.5 threshold. The greedy
  baseline is the model's own argmax text — the PPL floor — so unconditional
  continuation-PPL ratios are structurally harsh. Fine for *relative* comparisons
  across methods (which is what E0 needs); the absolute threshold needs calibration
  before being used as a hard filter.
- Between-family differences are within noise at this scale (F1.a 4.06× vs F2.a 4.62×
  vs F3.a 4.75× mean PPL ratio) — no conclusions until E0's proper sample.

## Step 8 — E0 pair selection

**Built:** `15_e0_pair_selection.ipynb` — 30 regime-tagged candidate pairs, per-synset
multi-token richness stats, ρ for all pairs (disk-cached frames), regime-vs-ρ scatter
(`resources/15_e0_rho_by_regime.png`), deterministic low/median/high selection rule
(hub concepts excluded, near-ties broken toward multi-token richness). Selection +
full pair table saved to `resources/15_e0_selected_pairs.json` — Step 9 reads from it.

**Selected pairs:**

| stratum | pair | ρ | regime |
|---|---|---|---|
| low | `sadness.n.01` / `music.n.01` | 0.187 | unrelated (word-level co-realizable) |
| medium | `woman.n.01` / `king.n.01` | 0.321 | compositional (ground truth: `queen.n.02`) |
| high | `sorrow.n.01` / `sadness.n.01` | 0.566 | similar (near-synonyms — easy control) |

**Findings:**
- All 30 candidate synsets survive the filter (the Step-3 `boy.n.01` gap doesn't recur).
- **Regime–ρ dissociation confirmed at scale** (n=30): regime means are
  similar 0.478 > hypernym 0.401 > antonym 0.339 > compositional 0.323 >
  unrelated 0.284. Antonyms sit squarely mid-pack — semantic opposition is NOT
  geometric distance, so E0 strata by measured ρ (as done) was the right call.
- The medium pair being compositional is a bonus: E0 can additionally check
  `queen.n.02`-presence as a composition-success signal at no extra cost.
- **Composition benefit is pair-dependent:** ρ(mean(woman, child), girl) = 0.488
  exceeds both constituents, but ρ(mean(woman, king), queen) = 0.515 falls slightly
  below king alone (0.531). The chordal mean does not uniformly approach the
  composed concept — direct input for RQ3.

## Step 9 — E0 pilot (3 pairs × F1.a/F2.a × 20 prompts × 2 orders, k=4, steps=32)

**Built:** `16_e0_pilot.ipynb`; artifacts `resources/16_e0_pilot.jsonl` (240 records),
`resources/16_e0_summary.csv`, `resources/16_e0_gap_vs_rho.png`.

**Result: the primary question is unanswered — the metric floored, not the methods.**
- **Presence-success is ~0 in every cell** (both-rate exactly 0 across all strata and
  methods; only single-concept blips: music 5% under F1.a-low, king 10% under
  F1.a-medium). Yet steering is *visibly* working: the low-stratum F1.a story is set in
  an antique **music shop** with a **pianoforté** — topic steering without exact member
  lemmas. The gap-vs-ρ plot is flat at 0 by metric floor, so it carries no evidence
  about the interference hypothesis either way.
- **Fluency cost explodes at k=4/steps=32:** mean PPL ratios 6.8–13.2× (vs ~4× at
  k=3/steps=16), 85–90% of runs flagged. Longer, harder steering drifts further
  off-distribution. F2.a is consistently more expensive than F1.a (e.g. 13.2× vs 8.5×
  at low ρ) — the only method-difference signal in the pilot, and it favors F1.a on
  fluency at surface level.
- **Batch-order robustness is perfect:** forward vs reversed both-rate delta = 0.000.
  One less confound to worry about.

**Decision (what E0 was designed to produce):** engineering effort goes to
**measurement, not new composition methods**:
1. Success metric v2 — continuous concept-expression: project each generated
   continuation onto each concept post-hoc (`fur.project`), method-agnostic and
   already implemented; and/or soften presence (stemming/partial match); classifier
   (plan's fallback) if needed.
2. Re-run E0 with metric v2 and a gentler operating point (k=3, steps~24) before
   drawing any ρ-gap conclusion.
3. PPL guardrail threshold needs calibration against the observed 7–13× range.

## Step 9b — E0 pilot v2 (continuous expression metric, k ∈ {3, 4})

**Built:** metric v2 in the harness — `concept_expression` (post-hoc `fur.project` of the
continuation, mean over positions, cached) and `expression_record` (steered vs baseline
delta per concept); `17_e0_pilot_v2.ipynb` (3 strata × 2 methods × k∈{3,4} × 20 prompts,
steps=24, single order per v1's zero order-delta). Artifacts:
`resources/17_e0_pilot_v2.{jsonl,csv,png}` (240 records). Joint measure: per-concept
deltas z-scored across records, joint = min(z_a, z_b).

**Findings:**
- **Metric v2 has signal:** mean expression delta +0.30/+0.27, ~60% of records positive,
  magnitudes track the eyeballed examples. Presence-both remains 0 everywhere,
  confirming v1's metric floor was real.
- **The interference-escape hypothesis is CONTRADICTED at pilot scale.** Prediction:
  F2.a (score space) escapes frame-averaging interference at low ρ, so the F2−F1 gap
  should be most positive at low ρ. Observed: **F1.a mean-frame wins at low and medium
  ρ at both k** (e.g. k=4 joint-z: −0.10 vs −0.34 at low; −0.01 vs −0.15 at medium);
  F2.a edges F1.a only at high ρ with k=3. The gap *rises* with ρ instead of falling.
  Interpretation candidates for E1: (a) the mean frame at low ρ still lands in a
  usable region while per-step z-scored voting fragments the trajectory; (b) F2.a's
  per-step normalization discards magnitude information that the single-frame method
  retains. Either way, frame-space composition is the stronger baseline going into E1,
  which raises the priority of RQ3's Riemannian aggregation (E2) — the winning family
  is the one whose aggregation RQ3 studies.
- **k=4 roughly doubles expression gains over k=3** in nearly every cell (e.g. F1.a-high
  0.35/0.33 → 0.64/0.64) — steering benefits from a wider candidate pool more than it
  loses to noise. Recommended E1 operating point: k=4.
- **Fluency remains the elephant:** 5.3–11.5× PPL ratios, 90–100% flagged everywhere;
  F2.a is more expensive at k=4 (11.5× at low). At current settings both methods are
  far outside the 2.5× guardrail — E1 needs either fluency-aware candidate selection
  or Pareto reporting (success × fluency), as the plan's E1 section already prescribes.
- Per-concept asymmetries are large at low ρ (sadness gains ≫ music gains under F1.a) —
  the mean frame does not distribute its effect evenly across constituents.

## Step 10 — E2: Riemannian aggregation (RQ3)

**Built:** `frames/linalg/stiefel.py` — canonical-metric Stiefel geometry:
`stiefel_exp` (Edelman/Arias/Smith closed form), `stiefel_log` (Zimmermann 2017
shooting algorithm; cheap because it works on 2k×2k matrices regardless of d),
`frechet_mean` (Karcher mean via exp/log fixed-point iteration, initialized at the
extrinsic mean), `aligned_mean` (generalized-Procrustes: rotate each frame within
its span to match the mean before chordal averaging). Exposed as
`Concept.average(..., method="extrinsic"|"aligned"|"frechet")` (default byte-identical
to the E0 baseline; verified by test + golden) and `average_method=` on the F1.a quick
wrapper. 26 CPU tests; `playground/step10_riemannian_mean.ipynb`;
`18_e2_riemannian.ipynb` (3 strata × 3 methods × 20 prompts, k=4, steps=24 — the
E0 v2 recommended operating point). Artifacts: `resources/18_e2_riemannian.jsonl`
(180 records), `18_e2_summary.csv`, `18_e2_method_gap_vs_rho.png`.

**Findings:**
- **A silent-garbage failure mode in the Stiefel log, caught before it touched any
  result:** the orthogonal completion inside Zimmermann's algorithm can come out
  with det = −1, which has no real principal logarithm — the log then "converged"
  to a wrong tangent (reconstruction error ~1.6 on unit-scale frames) with no error
  raised. Fixed by flipping one free completion column to force det = +1;
  regression-pinned (d=16, seeds (0,1)). Anything using a Stiefel log downstream
  (e.g. future geodesic interpolation experiments) inherits the fix.
- **Metric barely matters: the geodesic Fréchet mean ≈ the extrinsic mean at real
  concept distances.** ρ to constituents differs by <0.002 on all three E0 pairs,
  and 43/60 generations are byte-identical under greedy k-lookahead. Joint-z
  differences are ≤0.04 everywhere (e.g. high: −0.121 vs −0.125). RQ3 answer,
  part 1: the paper's cheap chordal mean is a near-optimal proxy for the true
  Karcher mean in this regime (geodesic distances 1.68–2.40) — a positive
  justification of existing practice, not a null result.
- **Gauge matters more than metric: the aligned (GPA) mean wins exactly in the
  low-ρ interference regime.** At ρ=0.187: joint_z −0.106 vs extrinsic −0.333,
  per-concept deltas roughly double (0.87/0.60 vs 0.61/0.29), at LOWER fluency
  cost (7.5× vs 8.8×). At medium/high ρ it loses moderately (−0.231 vs −0.085
  at medium). Interpretation: unrelated concepts' frames are rotation-mismatched
  in token gauge, so the plain weighted sum partially cancels; aligning first
  preserves each constituent's signal. Related concepts are already
  gauge-compatible, and re-gauging only distorts them. This *revives* E0's
  interference-escape hypothesis in frame space after its score-space version
  died in Step 9b: the escape exists, but via rotation alignment, not per-step
  normalization.
- **Static geometry does not predict steering efficacy:** the aligned mean has
  the LOWEST ρ to its constituents (0.741 vs 0.803 at medium) yet steers best at
  low ρ — mean-to-constituent ρ is not a proxy for downstream expression. E1
  metrics must stay generation-based.
- **Composition probes (RQ3 preview):** frechet edges out extrinsic on both
  (`woman+child→girl`: 0.4893 vs 0.4883; `woman+king→queen`: 0.5156 vs 0.5150);
  aligned is lowest on both. None of the means beats `king` alone for `queen`
  (0.531) — the Step-8 pair-dependence stands.
- **GPA gauge subtlety (thesis-worthy):** the aligned-mean fixed points form
  right-O(k) orbits — rotating any input leaves the aligned frames unchanged,
  and any rotation of a solution is also a solution. Since FRH's trace scoring
  is gauge-sensitive, the returned representative matters; `aligned_mean`
  anchors it via extrinsic-mean initialization (the gauge that won E0). Same
  geometry lesson as Step 5's subspace failure, resolved by construction this
  time.
- Practical: both intrinsic means converge in <1 s on real concept pairs
  (d=4096, k=3) — cost is not a factor in choosing between them.

---

## Cross-cutting engineering notes

- Test suite: 31 CPU tests (beam selection, ρ, aggregators) run by default; golden
  regression is `@pytest.mark.gpu` (`uv run pytest -m gpu`).
- Every step ships a `playground/stepNN_*.ipynb`, verified end-to-end via papermill.
- `resources/` is the untouchable baseline; concept frames cache to `cache/concepts/`
  (gitignored); golden files live in `tests/golden/`.
