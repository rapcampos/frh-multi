# Multi-Concept Top-k Concept-Guided Decoding — Working Summary

Context document for implementation work. Self-contained: no other files needed.

## 1. Project Context

This work extends **Top-k Concept-Guided Decoding (CGD)** from the Frame Representation Hypothesis paper (Valois, Souza, Shimomoto & Fukui, *TACL* 2025 — "Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation"; public code available) from single-concept to **multi-concept steering**. It is one work stream within a master's thesis on the geometry and mechanistic reality of frame representations; it doubles as the extrinsic evaluation for the thesis's concept-aggregation research question (RQ3), because composing concept frames and aggregating word frames into a concept are the same mathematical operation.

**FRH essentials.** A multi-token word = a *k-frame*: an ordered sequence of k token unembedding vectors. A concept = an aggregate (paper: Euclidean average) of the word frames of its WordNet member words, built from Open Multilingual WordNet. Vanilla Top-k CGD, per decoding step: complete the model's top-k token proposals into candidate words → build each candidate's word frame → score by similarity/correlation to the target concept frame → emit the argmax candidate. FRH also defines a concept–concept correlation **ρ**, reused here to predict steering interference. Models used in the paper and thesis: **Llama 3.1 8B, Gemma 2 9B**.

## 2. The Three Composition Families

The single-concept pipeline has three intervention points → three method families for concepts C1..Cn with weights w1..wn.

### Family 1 — Frame space (compose before scoring)
Build one composite frame F*, then run vanilla CGD unchanged.
- **F1.a mean frame:** F* = Σ wi·Fi componentwise, renormalized. Upgrades: Procrustes-aligned averaging, componentwise Fréchet means (chordal/geodesic) — these are the thesis RQ3 methods.
- **F1.b joint subspace:** concatenate all frames' vectors, orthonormalize (QR/SVD). OR-like semantics (aligned with *any* constituent scores well).
- **Known failure:** interference — averaging dissimilar concepts lands near none of them. Hypothesis: the gap between F1 and F2 grows with dissimilarity, i.e. is predicted by ρ. This is experiment E0.

### Family 2 — Score space (compose after scoring)
Score each candidate against each concept → score vector s = (s1..sn) → aggregate.
- **F2.a weighted sum:** Σ wi·si. Signed weights ⇒ negative steering (wi < 0 repels; generalizes FRH's bias remediation).
- **F2.b soft-min:** −τ·logsumexp(−s/τ). AND semantics (must satisfy every concept). Hard min = τ→0 limit, brittle under greedy decoding.
- **F2.c constrained/lexicographic:** maximize s1 subject to si ≥ τi. Topic + style/safety-constraint use case.
- **MANDATORY: per-step, per-concept score normalization** (z-score over the top-k candidates, or rank-based) before aggregation. Frames differ in rank/norms across concepts; without normalization the weights are meaningless. Implement as a toggleable flag for later ablation.

### Family 3 — Time (scheduling)
Single-concept scoring; vary the *active* concept per step: round-robin (F3.a), stochastic with weights as probabilities (F3.b), segment-level switching at sentence boundaries (F3.c). Niche: concepts that cannot co-occur in one word can co-occur in a passage; also the control condition distinguishing method failure from linguistic impossibility.

## 3. Reference Skeleton

```python
def multi_concept_cgd_step(model, context, concepts, weights, k,
                           mode="sum", tau=1.0):
    cand_words = get_topk_word_candidates(model, context, k)  # FRH word-completion loop
    best, best_score = None, -np.inf
    for w in cand_words:
        Fw = word_frame(w)                        # ordered token unembedding vectors
        s = np.array([frame_similarity(Fw, C) for C in concepts])
        s = zscore_over_candidates(s)             # normalization — see Family 2
        if mode == "sum":
            score = weights @ s
        elif mode == "softmin":
            score = -tau * logsumexp(-s / tau)
        elif mode == "constrained":
            score = s[0] if np.all(s[1:] >= THRESHOLDS) else -np.inf
        if score > best_score:
            best, best_score = w, score
    return best
```
Family 1 = pass `concepts=[composite_frame]`. Family 3 = scheduler wrapper choosing the active concept per step. Sequence-level variant: beam search over partial sequences scored by accumulated multi-concept satisfaction (deferred to step 11).

## 4. Implementation Roadmap (ordered; each step testable before the next)

**Phase A — Foundation**
1. Fork FRH repo; reproduce single-concept CGD on Gemma 2 9B with 2–3 paper concepts. No new code until a known result regenerates.
   - Golden file: run vanilla CGD on 10 fixed prompts, fixed seed; save outputs. Regression-test every later step against it.
2. Locate the decoding loop (candidate gen → frame build → score → select). Minimal refactor: make scoring a swappable function `score(candidate_frame, concepts: list, weights, mode) -> float`. Verify length-1 concept list reproduces the golden file token-for-token.
3. Concept-pair toolkit: given two WordNet concept IDs → both concept frames + their ρ. Cache frames to disk. Sanity-check ρ on obvious high/low pairs.

**Phase B — Families**
4. Family 2 first (no new frame math): (a) per-concept candidate scoring → vector s; (b) per-step z-score normalization behind a flag; (c) aggregators `weighted_sum` / `softmin` / `constrained`, unit-tested on hand-made score vectors before model runs.
5. Family 1: `mean_frame(frames, weights)` with renormalization; `joint_subspace(frames)` via QR/SVD. **Decide and document the unequal-rank policy** (words of different token lengths → truncate / pad / rank-normalize); this decision resurfaces in thesis RQ3.
6. Family 3: scheduler object returning active-concept index per step (round-robin, stochastic, sentence-boundary segments via punctuation heuristic).

**Phase C — Measurement**
7. Evaluation harness, JSONL logs per generation: prompt, full text, per-step chosen candidate + score vector, per-concept success (start: concept-word presence from WordNet member list; classifier only if presence proves too crude), perplexity under an unsteered reference model.
8. Fluency guardrail: batch perplexity post-process; flag runs where steered PPL > ~2–3× unsteered baseline on the same prompts.

**Phase D — E0 pilot**
9. Compute ρ over ~30 candidate concept pairs; select three at low/medium/high ρ. Prefer concepts rich in multi-token member words.
10. Pilot matrix: 3 pairs × {F1.a, F2.a equal-weight} × ~20 prompts × a few seeds. Deliverable: one plot — F1-vs-F2 performance gap as a function of ρ. Decides where engineering effort goes next.

**Phase E — After E0 only**
11. Beam-search variant with accumulated per-concept satisfaction (compute-heavy; needs best scoring mode known first).
12. Drop in Riemannian aggregation (Procrustes-aligned / Fréchet-mean) as `mean_frame` replacements → extrinsic RQ3 evaluation (E2).

## 5. Experiments Beyond E0

- **E1 family comparison:** concept pairs/triples stratified by ρ and WordNet distance; shared prompt suite; compare F1.a, F1.b, F2.a, F2.b, F3.b, greedy vs beam. Report Pareto frontiers over weight sweeps: success(C1) × success(C2) × perplexity — never single operating points.
- **E2 aggregation transfer:** best configs re-run with Riemannian concept frames (thesis RQ3 extrinsic test).
- **E3 negative/mixed steering:** one attracted + one repelled concept; compare suppression against FRH's single-concept bias-remediation results.

## 6. Pitfalls Checklist

- **Greedy myopia:** word-level argmax vs sequence-level satisfaction; beam variant is the fix and a required baseline.
- **Scale incommensurability:** fixed by per-step per-concept normalization; ablate the flag.
- **Fluency collapse:** track reference perplexity everywhere; use the guardrail threshold.
- **Co-occurrence limits:** some pairs are word-level impossible; Family 3 is the control that reveals this.

## 7. Open Questions

Compute ρ on raw frames or after Procrustes alignment? Track per-concept satisfaction with decaying credit over the sequence (toward controlled *proportions* of expression)? Are learned weights worth the machinery over hand-set sweeps?