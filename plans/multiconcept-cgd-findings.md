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
| 5–9 | Family 1 completion, schedulers, eval harness, E0 | ⏳ pending |

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

---

## Cross-cutting engineering notes

- Test suite: 31 CPU tests (beam selection, ρ, aggregators) run by default; golden
  regression is `@pytest.mark.gpu` (`uv run pytest -m gpu`).
- Every step ships a `playground/stepNN_*.ipynb`, verified end-to-end via papermill.
- `resources/` is the untouchable baseline; concept frames cache to `cache/concepts/`
  (gitignored); golden files live in `tests/golden/`.
