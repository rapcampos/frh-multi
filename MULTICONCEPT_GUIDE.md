# Multi-Concept Guided Generation

All guided generation now runs through a single unified loop,
`_generate_guided(input_text, concepts, weights, k, steps, scorer)`:
at each step it generates `k` candidate continuations per beam, projects the
hidden states onto every concept, aggregates the per-concept projections into
one score via a swappable `scorer` (default: weighted sum), and keeps the `k`
children of the beam with the highest cumulative score.

The original single-concept method `generate_with_topk_guide` is a thin
wrapper over this loop (one concept, weight 1.0) and is pinned token-for-token
by the golden regression test (`tests/test_golden.py`).

---

## Simultaneous multi-concept (`generate_with_topk_multi_guide`)

**Idea:** At every step, score candidates by the weighted sum of alignments to all guides. Beam selection is jointly driven by all concepts.

**How it works:**

1. For each concept `g_i` with weight `w_i`, computes `w_i * (hidden_frames * g_i)`.
2. Sums all weighted projections into a single score and uses it to select beams, exactly as the single-concept method does with that combined score.
3. Weights default to uniform (`1 / len(guides)`).

**Use when:** you want generation pulled toward multiple concepts simultaneously (e.g., both "joy" and "music" influence every step).

```python
texts, probe = rep.generate_with_topk_multi_guide(
    sentences, guides=[concept_joy, concept_music], weights=[0.6, 0.4], k=4, steps=16
)
```

**Caveat (linearity):** without per-step score normalization, the weighted sum
of frame correlations is mathematically equivalent to scoring against a single
weighted-average frame (correlation is linear in the concept frame). Treat
this method as distinct from frame averaging only once score normalization is
enabled. See `plans/multiconcept-cgd-implementation-plan.md`, Key finding 1.

---

## Mean-frame composition (`quick_generate_with_topk_mean_frame_guide`)

F1.a — averages the guide concepts into a single frame via `Concept.average`
(weighted extrinsic/chordal Procrustes mean; supports `guide_weights`, negative
weights repel) and runs vanilla single-concept guidance with it. Frames of
different ranks are zero-padded to a common k (rank-neutral policy).

## Joint-subspace composition (`quick_generate_with_topk_subspace_guide`)

F1.b — `Concept.joint_subspace` concatenates all guide frames' vectors and
orthonormalizes via SVD (rank-truncated), producing a basis of the union of
spans. OR semantics: alignment with any constituent scores well. This is the
true subspace method; before Step 5 this name incorrectly pointed at the
mean-frame composition.

---

## Removed: cascade methods

`generate_with_topk_cascade_guide` and variants were removed. The steering
loop keeps a single surviving parent per step, so the k final beams differ
only in their last token — a final rerank by a second concept could never
meaningfully steer. A true beam-search variant with genuinely diverse beams
replaces this idea (see the implementation plan, Step 2b). Historical cascade
results remain in `resources/12_cascade_probe.*`; notebook
`12_multiconcept_guided_generation.ipynb` still references the removed method
and will need its cascade cells skipped or removed if rerun.

---

## Output format

Guided methods return `(list[str], torch.Tensor)`:

| Return value | Shape | Description |
|---|---|---|
| `texts` | `list[str]` length `n` | One generated string per input |
| `probe` | `(n, T)` | Cumulative concept alignment over generation steps |

For the simultaneous method, the probe reflects the combined weighted score.
