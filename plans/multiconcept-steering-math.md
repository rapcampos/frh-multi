# Multi-Concept Steering — Mathematical Exposition

How concept-guided steering works mathematically, how each composition family
adapts the method to multiple concepts, and why the families are (or are not)
genuinely distinct. This is the expanded, self-contained companion to
`multiconcept-cgd-methods.md`, which maps each method to its code and
empirical results. Notation: `d` = unembedding dimension, `k` = vectors per
frame, `n` = number of concepts, `V` = vocabulary size.

## 0. The objects steering operates on

Take the model's unembedding matrix `W ∈ R^{V×d}`, whiten and center it:

```
W̃ = (W − μ) Σ^{-1/2}
```

where `Σ` is the row covariance of `W` and `μ` a "meaningless" reference
vector (mean or pad embedding) — `representations/unembedding.py`. Rows of
`W̃` are the token vectors.

A **word frame** for a multi-token word is the *ordered* matrix of its token
vectors, orthonormalized: `F ∈ R^{d×k}` with `FᵀF = I_k`, a point on the
Stiefel manifold `St(d, k)`. A **concept frame** `C` is the normalized
Euclidean average of the word frames of a synset's member words.

The similarity that drives everything is the **frame correlation**
(`linalg/frame.py::correlation`):

```
corr(F, G) = tr(Fᵀ G) / √(rank(F) · rank(G))
```

Unequal ranks are handled by zero-padding to a common `k`; padding is
rank-neutral because rank counts nonzero columns. Two properties of this
scalar decide the entire design space:

1. **Linearity in each argument**: `tr(Hᵀ(Σᵢ wᵢFᵢ)) = Σᵢ wᵢ tr(HᵀFᵢ)`.
   This single identity is what causes the score-voting degeneracy in §3.
2. **Gauge sensitivity**: `tr(FᵀG)` depends on the order and sign of the
   columns. Replacing `G` by `GR` for `R ∈ O(k)` changes the value even
   though `span(GR) = span(G)`. Frame correlation is a *Stiefel* quantity,
   not a subspace (*Grassmann*) quantity — this decides which compositions
   work and which fail (§2, subspace union).

## 1. The single-concept steering loop

Per decoding step (`representations/frame.py::_generate_guided`, the
golden-pinned path):

1. From the current candidate, take the model's top-k next tokens → `k`
   child sequences, evaluated in one forward pass.
2. For each child, slide a `k`-wide window over its last hidden states,
   `H_t = [h_t, …, h_{t+k−1}] ∈ R^{d×k}`, and score `s_t = corr(H_t, C)`.
3. A child's cumulative score is `S = Σ_t s_t` over all its positions.
4. Keep the argmax child; its own top-k children become the next step's
   pool.

Steering never touches logits or activations — it is a **rerank of the
model's own top-k proposals by concept alignment**. Every multi-concept
method changes exactly one ingredient of this loop:

| family | what changes |
|---|---|
| frame blending (F1) | the guide `C` |
| score voting (F2) | the scalar in step 2 |
| concept scheduling (F3) | *which* concept step 2 uses |
| beam search (§5) | the survivor rule in step 4 |

## 2. Family 1 — frame blending: build one composite guide

Compose frames `F₁ … Fₙ` into a single `F*` before decoding; the loop is
untouched.

**Simple blend (chordal/extrinsic mean)** (`Concept.average`). Sum, then
project back onto the Stiefel manifold:

```
F* = Π(Σᵢ wᵢ Fᵢ),    Π(A) = U Vᵀ  where  A = U S Vᵀ (thin SVD)
```

`Π` is the polar/Procrustes projection — the orthonormal frame nearest to
`A` in Frobenius (chordal) distance. It is scale-invariant, so only weight
*ratios* matter; negative weights repel. With `w = [1, −1]` this is exactly
the paper's differential guidance: `Frame.__sub__` computes `Π(F_A − F_B)`.
The weighted blend is the continuous family containing it.

**Aligned blend (generalized-Procrustes mean)**
(`linalg/stiefel.py::aligned_mean`). A frame is only defined up to a right
rotation of its columns (the token "gauge"). Dissimilar concepts arrive in
mismatched gauges, so terms in the raw sum partially cancel. Fix by
alternating:

```
Rᵢ ← polar(Fᵢᵀ X)         (best right-rotation of Fᵢ toward the current mean)
X  ← Π(Σᵢ wᵢ Fᵢ Rᵢ)       (simple blend of the aligned frames)
```

Fixed points form right-O(k) orbits, and since `corr` is gauge-sensitive the
*output* gauge matters — it is anchored by initializing `X` at the simple
blend. Empirically it wins for dissimilar pairs (ρ < 0.30) and loses
slightly above (E2/E2b).

**Geodesic blend (Fréchet/Karcher mean).** The true Riemannian center of
mass under the canonical Stiefel metric:

```
X* = argmin_X Σᵢ wᵢ d²(X, Fᵢ),   iterate  X ← Exp_X(Σᵢ wᵢ Log_X(Fᵢ))
```

using the Edelman–Arias–Smith closed-form geodesic and Zimmermann's shooting
algorithm for `Log` (works on 2k×2k matrices, so cost is independent of
`d`). Empirically indistinguishable from the simple blend at real concept
distances — **the metric doesn't matter, the gauge does**.

**Subspace union (negative result)** (`Concept.joint_subspace`). Take an SVD
orthonormal basis of `[F₁ … Fₙ]`, truncated at `σ < rtol·σ_max`.
Geometrically this is correct OR-semantics — its span contains every
constituent's span — but it fails because of gauge sensitivity (§0, property
2): an SVD basis is an *arbitrary rotation* of the span, and `tr(FᵀG)` is
rotation-sensitive, so the union's correlation with its own constituents is
≈ 0 or negative. Lesson: **composition geometry must match scoring
geometry** — a Grassmann construction cannot feed a Stiefel score.

## 3. Family 2 — score voting, and the degeneracy claim

Keep the individual frames; step 2's scalar becomes a vector
`sⁱ_t = corr(H_t, Cᵢ)`, aggregated by a swappable scorer
(`representations/aggregators.py`).

### Why unnormalized voting collapses into Family 1

Apply linearity, writing `rᵢ = rank(Fᵢ)` and `r_H = rank(H)` (constant —
it is the window width):

```
Σᵢ wᵢ · corr(H, Fᵢ) = Σᵢ wᵢ · tr(HᵀFᵢ) / √(r_H · rᵢ)
                    = (1/√r_H) · tr(Hᵀ · Σᵢ (wᵢ/√rᵢ) Fᵢ)
                    = (1/√r_H) · tr(Hᵀ A),    A := Σᵢ (wᵢ/√rᵢ) Fᵢ
```

The right-hand side is — up to a candidate-independent constant, invisible
to the argmax in step 4 — precisely the single-concept score against the
summed frame `A`. So an **unnormalized weighted vote is frame blending with
rank-absorbed weights `wᵢ/√rᵢ`, minus the Procrustes step `Π`**. It is not
a second method; it is a slightly worse parameterization of the first one
(worse because skipping `Π` leaves the guide off-manifold: correlated
concepts double-count their shared directions, and the score leaves its
calibrated range).

### What breaks the degeneracy: pool normalization

Per step and per concept, normalize scores over the candidate pool
(`aggregators.py::normalize_scores`):

```
ẑⁱ = (sⁱ − μⁱ_pool) / σⁱ_pool        (z-score; or rank-normalize to [0, 1])
```

This is a *nonlinear function of pool statistics* — `μⁱ` and `σⁱ` depend on
all candidates jointly — so `Σᵢ wᵢ ẑⁱ` can no longer be rewritten as
`tr(Hᵀ · something)` for any fixed frame. Only now is score voting a
genuinely distinct family. Normalization also fixes a practical problem:
concepts differ in rank and score variance, so raw scores are
scale-incommensurable and raw weights meaningless — one concept's scale
silently dominates the vote.

### The aggregators

Given `ẑ = (ẑ¹ … ẑⁿ)`:

- **Weighted vote** — `Σᵢ wᵢ ẑⁱ`. OR-like: a strong concept compensates a
  weak one; signed weights give negative steering.
- **Softmin vote** — `−τ · logsumexp(−w·ẑ / τ)`. AND semantics: a candidate
  is only as good as its worst weighted concept. The two limits: `τ → 0`
  recovers the hard `min` (brittle under greedy decoding — one noisy score
  kills a trajectory); `τ → ∞` tends to the average plus a
  candidate-independent `−τ log n`, i.e. it degenerates back into the
  weighted vote. `τ` interpolates between AND and OR.
- **Threshold vote (lexicographic)** — maximize `ẑ⁰` subject to `ẑⁱ ≥ τᵢ`
  for the rest. Violations get a **finite** penalty (−10⁴, not −inf):
  scores are cumulatively summed over positions, so a −inf at one position
  would poison the entire suffix and make every continuation incomparable
  rather than merely bad.

## 4. Family 3 — concept scheduling

Loop and score untouched; step 2 uses one *active* concept `a(t)` chosen by
a stateful scheduler (`representations/schedulers.py`):

- **Round-robin**: `a(t) = t mod n`.
- **Stochastic**: `a(t) ~ Categorical(w)`, seeded generator, reproducible.
- **Sentence-switching**: advance `a` when the current beam emits `.!?\n`;
  per-input state, so batched prompts can be on different concepts.

The semantic move: no single word has to satisfy two concepts — the
*passage* does, by time-multiplexing. This sidesteps the composition problem
entirely instead of solving it in frame or score space.

## 5. The search dimension (orthogonal to all three families)

**What the original loop actually is.** Despite tracking k candidates, it is
**greedy with one-step lookahead**: all k candidates at any step are children
of a single winning parent, so they share every token except the last.
Consequences: (a) a locally attractive token commits the whole trajectory
(greedy myopia); (b) any re-ranking after the fact can only ever change the
final token.

**True beam search** (`_generate_with_topk_beam_guide`) changes only step 4:
each of k beams expands into its k children (pool `k²` after the first
step), and the top-k *sequences by cumulative score* survive regardless of
parent (`_topk_beam_selection`). Beams genuinely diverge and a
temporarily-second-best trajectory can win later. No length normalization is
needed since every beam grows one token per step. `k = 1` degenerates to
plain greedy decoding, byte-identical to the original — used as the
regression pin.

**Fluency-blended selection** (both loops, `fluency_weight = λ`): survivors
are chosen by

```
Σ_t s_t + λ · Σ_t log p(token_t)
```

`λ = 0` is the pure concept objective (golden path); `λ → ∞` recovers
unsteered decoding — so `λ` interpolates between maximal steering and the
base model. The reported probe stays the concept trajectory.

Empirically (E1/E1b): beam search is the largest single improvement in every
ρ regime, and **beam + simple blend + λ ≈ 0.4** is the current recommended
recipe.

## 6. The one-line summary of the degeneracy structure

The correlation's linearity means "combine frames then score" and "score
then combine linearly" are the same operation; every genuine second method
must insert a nonlinearity somewhere — `Π` onto the manifold (Family 1),
pool normalization + min/threshold (Family 2), or time-multiplexing
(Family 3).
