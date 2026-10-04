# Mathematical-foundation review: FRH implementation vs. Fukui, *Subspace Methods*

**Scope.** Checks whether the linear-algebra core of this codebase
(`frames/linalg/`, `frames/representations/`) is consistent with the
mathematical foundation of the subspace method (SM), mutual subspace method
(MSM), constrained MSM (CMSM), and whitening MSM (WMSM) as laid out in the
uploaded encyclopedia entry (Fukui, *Subspace Methods*, in *Computer Vision*,
Springer 2020).

**Bottom line.** The numerical primitives are correct and cleanly implemented.
The whitening, the orthonormalization onto the Stiefel manifold, and the
union-subspace construction all match their classical counterparts. The one
foundational thing to be explicit about: the core similarity operator
(`Frame.__mul__`) is a **frame (Stiefel) Frobenius-cosine**, *not* the
rotation-invariant **canonical-angle (Grassmann) similarity** `cos²θ` that
*defines* SM/MSM. This is a deliberate FRH design choice — concepts are ordered
frames, not bare subspaces — and it is internally consistent, but the
correspondence to Fukui's SM/MSM is analogical, not literal. Two secondary
constructions (frame subtraction, rank-by-nonzero-count) are sound heuristics
that should not be conflated with their classical namesakes (the generalized
difference subspace, and the true matrix rank).

---

## 1. What the paper defines

- **SM** (Eq. 1): similarity of an input vector **p** to a class subspace with
  orthonormal basis `{φ_i}` (PCA eigenvectors of the class correlation matrix
  `Σ xᵢxᵢᵀ`):

  `S = cos²θ = Σᵢ (p·φᵢ)² / ‖p‖²`.

  This is the squared cosine between **p** and its projection onto the subspace;
  it is **invariant to the choice of orthonormal basis** of the subspace.

- **MSM** (Eq. 2): similarity of two subspaces `P`, `Q` via canonical angles.
  With orthonormal bases `A=[Φ₁…]`, `B=[Ψ₁…]`, form `X = AᵀB`; its singular
  values `κᵢ` are the cosines of the canonical angles, and
  `S̃ = (1/t) Σᵢ cos²θᵢ`. Again **rotation-invariant** within each subspace.

- **CMSM**: run MSM on a *generalized difference subspace* (GDS) built from the
  **sum of orthogonal projection matrices** of the class subspaces, then
  re-orthonormalize (the paper explicitly notes the projection destroys
  orthonormality, so Gram–Schmidt follows).

- **WMSM**: whiten the class subspaces so their relationship approaches
  orthogonality before MSM.

The unifying theme: everything of interest lives on the **Grassmann manifold**
(subspaces up to rotation), and similarity is measured through **canonical
angles**.

---

## 2. Primitive-by-primitive check

### 2.1 Whitening — `symsqrtinv`, `LinearUnembeddingRepresentation` ✔ correct

`symsqrtinv(M)` eigendecomposes `M = V diag(L) Vᵀ` (`torch.linalg.eigh`) and
returns `V diag(L^{-1/2}) Vᵀ`, thresholding `L ≤ eps` to zero. This is the
symmetric (ZCA) inverse square root — correct, and the symmetric variant is the
right choice (it is the orthonormal-closest whitening, preserving orientation).

`_euclidean_representations = W · Σ^{-1/2}` with `Σ = cov(Wᵀ)` (the `d×d`
row-covariance of the unembedding matrix). This yields `⟨w̃ᵢ, w̃ⱼ⟩ = wᵢ Σ⁻¹ wⱼᵀ`,
i.e. the Mahalanobis inner product becomes Euclidean. **This is exactly the
preconditioning that makes subsequent angle/cosine similarities meaningful** in
a space whose natural metric is `Σ`-weighted. It is more fundamental than — and
serves a different purpose from — WMSM's *inter-subspace* whitening: here the
goal is isotropy of the token representation space, not orthogonalizing class
subspaces against each other. Mathematically sound and well-motivated. Centering
by subtracting the "meaningless" mean/pad vector is standard mean-removal.

### 2.2 Orthonormalization onto the Stiefel manifold ✔ correct

- `solve_procrustes(M)` returns `U Vᵀ` from `SVD(M)` — the orthogonal polar
  factor, i.e. the **nearest orthonormal matrix to `M` in Frobenius norm** and
  the projection onto the Stiefel manifold. Correct. This is used both to build
  concept frames (`average_frames = solve_procrustes(Σ word_frames)`) and for
  the extrinsic mean. It is a valid (indeed optimal) substitute for the paper's
  Gram–Schmidt re-orthonormalization step.

- `gram_schmidt` implements classical GS: `v ← v − Σⱼ ⟨uⱼ,v⟩ uⱼ` against the
  already-orthonormal `Q[:i]`, with an `eps` norm floor that zeroes degenerate
  directions. Verified correct (line 32 broadcasts the projections correctly).
  Classical rather than modified GS — slightly less numerically robust, but
  correct; and it is not on the hot path (Procrustes is).

### 2.3 The similarity operator `Frame.__mul__` / `correlation` — ⚠ correct but *not* SM/MSM similarity

```
F₁ * F₂ = trace(F₁ᵀF₂) / √(rank₁ · rank₂)         # einsum "...mdk,ndk->...mn"
```

The einsum contracts over **both** the embedding axis `d` and the vector axis
`k`, so this is the **Frobenius inner product** of the two `d×k` matrices,
`⟨F₁,F₂⟩_F = Σᵢ aᵢ·bᵢ = trace(F₁ᵀF₂)`, normalized by the geometric mean of the
ranks. By Cauchy–Schwarz it lands in `[−1, 1]`, is `1` for identical frames and
`0` for orthogonal ones — a well-defined **cosine between frames viewed as
points in `ℝ^{d×k}`** (a chordal/embedding similarity on the Stiefel manifold).

**Why this is not the paper's similarity.** `trace(AᵀB)` is *not* invariant to an
independent rotation `A → AR`, `R ∈ O(k)`:
`trace((AR)ᵀB) = trace(RᵀAᵀB) ≠ trace(AᵀB)`. The SM/MSM similarities are built
from `AAᵀ`/singular values of `AᵀB` and *are* rotation-invariant. Concretely:

- MSM uses `Σᵢ cos²θᵢ = ‖AᵀB‖_F²` (rotation-invariant).
- This code uses `trace(AᵀB)`, which satisfies `trace(AᵀB) ≤ Σᵢ σᵢ = Σᵢ cosθᵢ`
  (the nuclear norm / sum of cosines of canonical angles), **with equality only
  when the two frames' columns are aligned to their canonical bases.**

So the frame correlation is a *lower bound* on the MSM-style similarity and
coincides with it only in aligned or degenerate cases. In the **rank-1** case
(`a`, `b` unit vectors) it is `a·b = cosθ`, whereas SM gives `cos²θ` — different
values but monotone-equivalent for `θ∈[0,π/2]`, so top-k rankings agree there.

**This is by design, not a bug.** The Frame Representation Hypothesis represents
each multi-token concept as a *specific ordered frame* (each column ties to a
token position), i.e. a point on the **Stiefel** manifold, and it *wants* the
basis/ordering to matter. Fukui's methods deliberately quotient that away
(Grassmann). The implementation is faithful to FRH; it is simply not a literal
implementation of Eq. (1)/(2). The `√(rank₁·rank₂)` denominator is the analogue
of SM's `1/‖p‖` and MSM's `1/t` averaging — it keeps the score bounded and
comparable across ranks — but it does **not** convert the operator into averaged
`cos²θ`.

*Consistency of `project`.* `_project_hidden_states` correlates a **raw**
(non-orthonormalized) sliding window of hidden states against the orthonormal
concept frame. So the probe is a scaled projection magnitude, not a pure cosine
— appropriate for a "how much does this state align with the concept" score, and
consistent with the FRH probe.

### 2.4 Frame subtraction `Frame.__sub__` — ⚠ heuristic, not the CMSM difference subspace

`F₁ − F₂ = solve_procrustes(F₁.tensor − F₂.tensor)`: the nearest orthonormal
frame to the **element-wise** difference of the two basis matrices. This is a
chordal/Procrustes difference on the Stiefel manifold and works for differential
steering ("toward A, away from B"), but it is **basis/ordering dependent** and is
**not** the *generalized difference subspace* of CMSM (which comes from the sum
of orthogonal **projection matrices**, a rotation-invariant object). The name
"difference" should not be read as the classical difference subspace. It is
defensible as a Stiefel operation given that concept frames are built by a
single consistent construction (so their bases are comparably ordered), but it
is a heuristic.

### 2.5 Concept composition — ✔ sound

- `Concept.average` (extrinsic) `= solve_procrustes(Σ wᵢ Cᵢ)` is the correct
  **extrinsic (projected-arithmetic) mean** on the Stiefel manifold; the
  `aligned`/`frechet` options are the generalized-Procrustes and geodesic
  Karcher means, correctly described and correctly gated (equal effective rank
  required; zero-padded columns are excluded from the polar factor). Good.
- `Concept.joint_subspace` = SVD of concatenated frames with a relative-tolerance
  truncation is the correct **orthonormal basis of the union of spans** — this is
  the construction closest to classical subspace-method spirit (a genuine
  subspace object with OR semantics).

### 2.6 Rank convention — ⚠ upper bound, documented

`vector_count = tensor.ne(0).any(-2).sum(-1)` counts nonzero columns — an **upper
bound** on the true matrix rank (the code says so). It is **exact** for the
orthonormal concept frames that dominate the correlation denominator, and only
approximate for raw (hidden-state) frames. The zero-padding used to reconcile
unequal ranks is genuinely rank-neutral under this definition, so `rho` and the
unequal-rank policy are self-consistent. `relative_rank` uses the true
`matrix_rank` for full-rank diagnostics. Fine as long as one remembers the
denominator normalization is heuristic for non-orthonormal frames.

---

## 3. Summary table

| Component | Paper concept | Implementation | Verdict |
|---|---|---|---|
| `symsqrtinv`, whitening | (WMSM whitening; general preconditioning) | ZCA `Σ^{-1/2}`, `W·Σ^{-1/2}` | ✔ correct, well-motivated |
| `solve_procrustes` | orthonormal basis / re-orthonormalization | polar factor (nearest orthonormal) | ✔ correct (optimal) |
| `gram_schmidt` | Gram–Schmidt | classical GS + eps floor | ✔ correct |
| `Frame.__mul__` | SM/MSM similarity `cos²θ` | Frobenius-cosine `trace(AᵀB)/√(pq)` | ⚠ well-defined, but Stiefel not Grassmann; lower-bounds `Σcosθᵢ`, not `Σcos²θᵢ` |
| `Frame.__sub__` | (CMSM) difference subspace | Procrustes of element-wise diff | ⚠ heuristic, not the GDS |
| `Concept.average` | — | extrinsic / aligned / Fréchet mean | ✔ correct |
| `Concept.joint_subspace` | union / class subspace | SVD union basis | ✔ correct |
| `vector_count` | dimension `d_q` | nonzero-column count | ⚠ upper bound (exact for orthonormal) |

---

## 4. Recommendations (optional, non-blocking)

1. **Document the Stiefel-vs-Grassmann distinction** where `Frame.correlation`
   is defined. State plainly that the operator is a frame Frobenius-cosine
   (order-dependent by design), give the `trace(AᵀB) ≤ Σcosθᵢ` relation, and
   note it is *not* the rotation-invariant `cos²θ` of SM/MSM. This pre-empts the
   natural reader assumption that "subspace correlation" means canonical angles.
2. **Rename or annotate `Frame.__sub__`** so it is not mistaken for the CMSM
   generalized difference subspace; call it a chordal/Procrustes difference.
3. If a *bona fide* MSM similarity is ever wanted (e.g. for an ablation against
   the frame operator), it is one line: `‖AᵀB‖_F² / (rank₁·rank₂)` via the
   singular values of the paired frames — worth having as a comparison baseline
   given the paper is cited as foundation.
4. Keep the note that the `√(rank₁·rank₂)` normalization is a bounding/averaging
   device analogous to SM's `1/‖p‖` and MSM's `1/t`, not a route to `cos²θ`.

**No correctness defects were found.** The concerns above are about
*interpretation and naming* relative to the classical subspace-method
foundation, not about the numerics, which check out.
