# Multi-Concept Guided Generation

Two new generation strategies built on top of `_generate_with_topk_guide`, both accepting two concepts `c1` and `c2`.

---

## Method 1 — Cascade (`generate_with_topk_cascade_guide`)

**Idea:** Use `c1` to steer the beam search at every step. At the end, pick the surviving beam most aligned to `c2`.

**How it works:**

1. Maintains `k` beams throughout generation, selecting at each step the beam with highest cumulative alignment to `c1` (identical to the single-concept method).
2. After generation completes, scores all `k` surviving beams against `c2`.
3. Returns the single beam per input with the highest `c2` alignment.

**Use when:** you want generation driven by one concept but the final output filtered by another (e.g., steer toward "joy" during generation, then pick the output that also reads as "music").

```python
texts, c2_probe = rep.generate_with_topk_cascade_guide(
    sentences, c1=concept_joy, c2=concept_music, k=4, steps=16
)
```

---

## Method 2 — Simultaneous (`generate_with_multi_topk_guide`)

**Idea:** At every step, score candidates by the weighted sum of alignments to all guides. Beam selection is jointly driven by all concepts.

**How it works:**

1. For each concept `g_i` with weight `w_i`, computes `w_i * (hidden_frames * g_i)`.
2. Sums all weighted projections into a single score and uses it to select beams, exactly as the single-concept method does with that combined score.
3. Weights default to uniform (`1 / len(guides)`).

**Use when:** you want generation pulled toward multiple concepts simultaneously (e.g., both "joy" and "music" influence every step).

```python
texts, probe = rep.generate_with_multi_topk_guide(
    sentences, guides=[concept_joy, concept_music], weights=[0.6, 0.4], k=4, steps=16
)
```

---

## Output format

Both methods return `(list[str], torch.Tensor)`, matching `generate_with_topk_guide`:

| Return value | Shape | Description |
|---|---|---|
| `texts` | `list[str]` length `n` | One generated string per input |
| `probe` | `(n, T)` | Cumulative concept alignment over generation steps |

For the cascade method, the probe reflects `c2` alignment of the selected beam. For the simultaneous method, it reflects the combined weighted score.

---

## Bug fixed

`_generate_with_multi_topk_guide` had a missing `projections = None` initialization before the per-concept accumulation loop, causing `UnboundLocalError` on every call. This is now fixed.
