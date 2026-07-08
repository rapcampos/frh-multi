# Multi-Concept CGD — Implementation Plan

Detailed step-by-step plan derived from `RESEARCH_PLAN.md`, adapted to the current state of the codebase. Each step is independently verifiable before the next.

**Ground rules (agreed 2026-07):**

- `_generate_with_topk_guide` in `frames/representations/frame.py` is the **original paper function**. Its behavior must be preserved exactly through every change; all coding decisions defer to it. It is kept as-is *despite* the greedy-myopia pitfall (single surviving parent per step) — it is the paper baseline. The other generation methods (`multi_guide`, `subspace`) are later prototypes and are negotiable.
- **Cascade is removed entirely** (both `_generate_with_topk_cascade_guide` and `_v1`, plus their wrappers) — structurally toothless, see Key finding 2.
- A **true beam-search variant is implemented in parallel** to the original (new method, original untouched): k diverse beams selected per step by cumulative guide score, fixing the myopia limitation.
- Golden-file reference model: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, with **k=3, steps=16**.
- Testing policy: `tests/` + pytest for math/aggregator logic (CPU where possible); steering experiments stay in notebooks.
- **Existing `resources/` content is the baseline — never modify or overwrite it.** New results may be added to `resources/` under new filenames; golden files live in `tests/golden/`.

---

## Current state vs. roadmap (audit)

| Roadmap item | Status |
|---|---|
| Single-concept CGD | ✅ `generate_with_topk_guide` |
| Golden-file regression test | ❌ missing |
| Swappable scoring function | ❌ three near-identical loops with inlined scoring |
| Concept-pair toolkit + ρ | 🟡 `get_concept` + `Frame.correlation` exist; no pair helper, no disk cache |
| F2.a weighted sum | 🟡 `generate_with_topk_multi_guide`, but **no normalization** (see confound below) |
| Score normalization flag (MANDATORY) | ❌ missing |
| softmin / constrained aggregators | ❌ missing |
| F1.a mean frame | 🟡 `Concept.average` — Procrustes mean, unweighted only |
| F1.b joint subspace | ❌ `quick_generate_with_topk_subspace_guide` is misnamed — it calls `Concept.average` (= F1.a) |
| Family 3 scheduler | ❌ missing |
| Eval harness / perplexity | ❌ missing (`loss()` exists as a building block) |

### Key finding 1 — the F1.a/F2.a linearity confound

Frame correlation `trace(FwᵀC)/√(rank·rank)` is linear in the concept frame, so an **unnormalized** weighted sum of per-concept scores equals the score against a single weighted-average matrix — i.e. unnormalized F2.a and F1.a differ only in per-concept rank scaling and the Procrustes step. They are one method, not two families. Without per-step per-concept normalization, E0's F1-vs-F2 gap is ~zero at every ρ *by algebra*, and the pilot would produce a baked-in null result. Per-step z-scoring (data-dependent, nonlinear) is what makes F2 a genuinely distinct family → normalization must land before any pilot runs.

### Key finding 2 — cascade rerank is structurally toothless

The steering loop keeps a single surviving parent per step: the k "beams" are always the k next-token children of one sequence, so they share their entire prefix and differ only in the final token. Cascade's c2 rerank therefore only ever chooses the last token — it can never meaningfully steer toward c2. `_generate_with_topk_cascade_guide` and `_generate_with_topk_cascade_guide_v1` are functionally equivalent (same loop, same rerank criterion). **Decision: remove both cascade implementations entirely** (Step 2), and pull the true beam-search variant forward (Step 2b) — with genuinely diverse beams, a cascade-style final rerank can be revisited later as a trivial post-hoc filter if wanted.

---

## Phase A — Lock down current behavior

### Step 1: Golden file

- Notebook `14_golden_reference.ipynb`: run `generate_with_topk_guide` on 10 fixed prompts × 2–3 paper concepts (e.g. `dog.n.01`, `joy.n.01`) with **k=3, steps=16** on the AWQ Llama 3.1 8B; save texts + final probe values to `tests/golden/single_concept.json`.
- `tests/test_golden.py` (pytest, `@pytest.mark.gpu`): re-run the prompts, assert byte-identical texts; fall back to score comparison with tolerance if near-tie flakiness appears.
- Determinism notes: decoding is greedy (no seed), but batch padding and CUDA non-determinism can flip near-ties — golden run and check must use the same batch composition. The golden file is hardware-local (AWQ), not portable across GPUs.
- **Gate:** golden file committed; test passes twice in a row.

### Step 2: Unify the generation loops, remove cascade

- Extract the shared candidate-gen → project → cumsum → select loop into one `_generate_guided(input_text, concepts, weights, k, steps, scorer)` in `frames/representations/frame.py`. `_generate_with_topk_guide` keeps its name and public behavior (thin wrapper or left verbatim — whichever provably preserves it); multi and mean-frame become thin wrappers.
- **Remove cascade entirely:** `_generate_with_topk_cascade_guide`, `_generate_with_topk_cascade_guide_v1`, `generate_with_topk_cascade_guide`, `quick_generate_with_topk_cascade_guide`, and the cascade section of `MULTICONCEPT_GUIDE.md`.
- Cleanup: confirm `_select_best_candidates` is unused (grep) and remove.
- **Gate:** golden test passes token-for-token with `concepts=[guide]`, `mode="sum"`.

### Step 2b: True beam-search variant (parallel to the original)

- New method `generate_with_topk_beam_guide` alongside the original — the original is **not** modified.
- Loop difference vs. original: expand all k beams × k next-token candidates → k² sequences, score each by cumulative guide projection, keep the **top-k sequences** (not the k children of one parent). Beams stay diverse; final output is the top-1 beam (and the k finalists are available for post-hoc reranking/analysis).
- Accepts the same swappable scorer as `_generate_guided`, so all Family 1/2/3 modes work with both greedy and beam decoding for free.
- Practical notes: k² forward-pass width (same single-pass trade-off the original documents); length normalization decision for cumulative scores must be documented (beams can end at different EOS steps).
- **Gate:** with k=1 it degenerates to greedy unsteered decoding (sanity check); qualitative check that beams actually diverge on golden prompts; unit test for the top-k sequence-selection logic on hand-made score tensors (CPU).

### Step 3: Concept-pair toolkit + ρ

- Helper (new `frames/representations/pairs.py` or a method on `FrameUnembeddingRepresentation`): `concept_pair(synset_a, synset_b) -> (Concept, Concept, rho)` using `get_concept` + `Frame.correlation`; per-concept frame tensors cached to disk (`cache/concepts/`) keyed by model id + synset + tokenizer args.
- **Gate:** sanity-check ρ ordering on known pairs — high (`dog.n.01`/`cat.n.01`) vs low (`dog.n.01`/`algebra.n.01`).

## Phase B — The three families

### Step 4: Family 2 (score space) — first, no new frame math

- `frames/representations/aggregators.py`: pure-tensor `weighted_sum`, `softmin(s, tau)`, `constrained(s, thresholds)`, plus `normalize_scores(s, method="zscore"|"rank"|None)` applied per-step per-concept over the k candidates. Normalization is a flag on `_generate_guided` — off only in the golden path; mandatory for experiments (Key finding 1).
- `tests/test_aggregators.py`: CPU-only unit tests on hand-made score vectors (known argmax per mode; z-score invariances; softmin → min as τ→0).
- **Gate:** unit tests green before any model run.

### Step 5: Family 1 (frame space)

- Extend `Concept.average(concepts, weights=None)` — weighted sum then `solve_procrustes`.
- New `Concept.joint_subspace(concepts)` — concatenate frame vectors, orthonormalize via QR/SVD (the real F1.b). Rename `quick_generate_with_topk_subspace_guide` → mean-frame naming; point "subspace" at the new method.
- Document the unequal-rank policy: keep the codebase convention — zero-pad to common k, rank-aware correlation denominator (resurfaces in thesis RQ3; must be written down).
- Terminology nit for the thesis: `Concept.average` is the *extrinsic/chordal* (Procrustes) mean, not the geodesic Fréchet mean — fix the docstring; RQ3 compares exactly these.
- `tests/test_composition.py`: orthonormality of outputs; `w=[1,0]` recovers concept A; subspace contains each constituent's span.
- **Gate:** unit tests green.

### Step 6: Family 3 (scheduling)

- `frames/representations/schedulers.py`: `RoundRobin`, `Stochastic(weights)`, `SentenceBoundary` (punctuation heuristic); each returns the active-concept index per step, plugging into `_generate_guided` as a scorer that masks all but one concept.
- `tests/test_schedulers.py`: deterministic round-robin sequences; distribution check for stochastic; boundary detection on sample texts.

## Phase C — Measurement

### Step 7: Evaluation harness

- `frames/evaluation/harness.py`: one JSONL record per generation — prompt, full text, per-step chosen candidate + raw and normalized score vectors, config. Success metric v1: concept-word presence from the WordNet member-lemma list (via `MultiLingualWordNetSynsets`); classifier only if presence proves too crude.
- Perplexity of steered text under the unsteered model via existing `loss()`; guardrail flag when steered PPL > ~2.5× unsteered baseline on the same prompt.
- **Gate:** run on golden prompts; JSONL parses; PPL numbers plausible.

## Phase D — E0 pilot (the decision point)

### Step 8: Pair selection

- Notebook `15_e0_pilot.ipynb`: compute ρ over ~30 candidate pairs (prefer synsets rich in multi-token lemmas — filterable from the synsets DataFrame); pick 3 pairs at low/medium/high ρ.

### Step 9: Pilot matrix

- 3 pairs × {F1.a mean-frame, F2.a equal-weight + z-score} × 20 prompts × a few batch orders; optionally × {greedy, beam} since the beam variant (Step 2b) is available. Deliverable: **one plot** — F1-vs-F2 success gap as a function of ρ, plus PPL guardrail flags. Decides where engineering effort goes next.

## Phase E — Deferred until E0 results exist

- Riemannian aggregation (Procrustes-aligned / geodesic Fréchet) as `mean_frame` drop-ins → extrinsic RQ3 evaluation (E2).
- (Beam search moved up to Step 2b; only its compute-heavy full E1 sweep waits for E0 results.)

---

**Pacing:** Steps 1–2 are one work session (the refactor is the riskiest part — hence golden file first). Steps 3–6 are mostly CPU-testable. Step 7 is plumbing. E0 is where GPU time gets spent.
