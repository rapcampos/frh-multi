# Mathematical reference for FRH multi-concept work

This note records the mathematical contracts that matter for implementation. It is a
navigation aid, not a substitute for citing the original papers in thesis text.

## Source locations

The external corpus is a sibling of the repository:

- `../resources/papers/frh.pdf`: the published 2025 TACL FRH paper.
- `../resources/papers/fukui_k/`: Kazuhiro Fukui's broader reference corpus.

Repository `resources/` is different: it contains generated experimental artifacts.

The most directly relevant Fukui papers are:

- `1806.03125_Text_Classification_based_on_Word_Subspace_with_Term-Frequency.pdf`
  for word-set subspaces and canonical-angle matching.
- `1909.01954_Tensor_Analysis_with_n-Mode_Generalized_Difference_Subspace.pdf`
  for multi-mode GDS, projection/reorthogonalization, and weighted product-Grassmann
  distances.
- `1910.13113_Discriminant_analysis_based_on_projection_onto_generalized_difference_subspace.pdf`
  for GDS as quasi-orthogonalization and its relationship to Fisher discrimination.
- `2111.04352_Grassmannian_learning_mutual_subspace_method_for_image_set_recognition.pdf`
  for basis-invariant subspace matching and Riemannian learning on the Grassmannian.
- `2303.17802_Time-series_Anomaly_Detection_based_on_Difference_Subspace_between_Signal_Subspa.pdf`
  for intersection-aware difference subspaces and numerical eigenvalue thresholds.
- `2409.08563_Second-order_difference_subspace.pdf` for the general unequal-dimension
  first-order definition and the second-order/geodesic interpretation.

The corpus also contains adjacent work on convex cones, tensor analysis, time-series
subspaces, occlusion/augmentation subspaces, and subspace-based humor evaluation.
Consult those papers when a task reaches those topics; they do not override FRH's
ordered-frame steering definition.

## FRH objects

Let the vocabulary-size-by-dimension unembedding matrix be

```text
W in R^(V x d).
```

The paper equips the unembedding space with the inverse-covariance inner product
`M = Cov(W)^(-1)`. The implementation instead whitens rows with
`Cov(W)^(-1/2)`, so subsequent Euclidean dot products encode the same metric. It also
subtracts a meaningless reference vector, normally the whitened row mean.

A tokenized word `(w_1, ..., w_k)` is represented by the ordered column matrix

```text
F = [u(w_1) ... u(w_k)] in R^(d x k).
```

The paper's basic hypothesis is full column rank for ordinary words. These are
non-compact/full-rank frames; they are not merely unordered spans. In code, raw word
and hidden-state frames need not be orthonormal, while concept-composition operators
usually return an orthonormal polar factor.

For frames `A` and `B`, after trailing zero-padding to a common column count, the
implemented correlation is

```text
corr(A, B) = tr(A^T B) / sqrt(rank(A) rank(B)).
```

Equivalently, it pairs columns in order and sums their dot products. Reordering,
sign-flipping, or right-rotating columns generally changes it. This gauge sensitivity is
intentional for token order. When a raw hidden-state window is compared to an
orthonormal concept frame, the value is an unnormalized projection and need not behave
like a bounded cosine.

## Concept and differential frames

Word frames of unequal token length are padded on the right with zero columns. Given
word frames `F_i`, the concept frame is the polar/orthogonal-Procrustes projection of
their sum:

```text
A = sum_i F_i
A = U Sigma V^T
C = U V^T.
```

The same construction gives a weighted extrinsic/chordal concept mean. Scaling every
weight by the same positive constant does not change the polar factor; signed weights
support repulsion.

The paper's differential concept is the closest frame to a frame difference:

```text
D(B, A) = polar(B - A).
```

The appendix interprets `B - A` as a second-order approximation to the velocity of the
Stiefel geodesic from `A` to `B` at its midpoint when the concepts are not too far apart.

## The paper CGD loop

At each step, the model proposes its top-k next tokens. Each proposal completes a
candidate hidden-state window, and the algorithm scores its correlation with the guide.
It chooses the parent with the best cumulative guide score and carries that parent's
top-k children into the next step.

The pool therefore has k candidates but only one surviving parent lineage after each
decision. It is greedy one-step lookahead, not sequence-level beam search. The
repository's true beam variant changes this survivor rule by allowing children from all
surviving parents to compete.

The paper reports that increasing k strengthens steering but can reduce coherence and
induce repetition. Persistent steering may also delay end-of-sequence tokens and cause
verbosity. WordNet concepts can mismatch the model's internal semantics, and harmful
guidance demonstrates an explicit dual-use risk.

## Stiefel frame geometry versus Grassmann subspace geometry

These are related but not interchangeable representations:

- A Stiefel frame retains an ordered, oriented basis. Right multiplication by an
  orthogonal matrix changes its gauge. FRH trace correlation and steering use this
  structure.
- A Grassmann point is only the span. `F` and `F R` represent the same point for any
  orthogonal `R`. Principal angles and projector-based GDS use this structure.

For orthonormal bases `Phi in R^(d x p)` and `Psi in R^(d x q)`, compute

```text
Phi^T Psi = U diag(sigma_i) V^T
theta_i = arccos(sigma_i),  i = 1, ..., min(p, q).
```

The `theta_i` are principal/canonical angles. Common basis-invariant measures include

```text
Grassmann geodesic distance = ||theta||_2
mean canonical correlation = mean_i cos(theta_i)
projection similarity       = sum_i cos(theta_i)^2
                            = ||Phi^T Psi||_F^2.
```

This distinction explains a central negative result in the project. An SVD basis of the
union of concept spans is a valid Grassmann construction, but its basis orientation is
arbitrary. Feeding that basis to the gauge-sensitive FRH trace score can yield near-zero
or negative correlation even with a constituent whose span is contained in the union.

Use gauge-invariant metrics when the question is whether text landed near a semantic
span. Use the original trace correlation when reproducing the ordered-frame steering
objective.

## Stiefel means used in the project

For concept frames `F_i` and normalized nonnegative weights `w_i`:

- Extrinsic/chordal mean: `polar(sum_i w_i F_i)`.
- Aligned/generalized-Procrustes mean alternates

  ```text
  R_i = polar(F_i^T X)
  X   = polar(sum_i w_i F_i R_i).
  ```

  It aligns each frame within its right-orthogonal orbit before averaging. The output
  gauge is anchored by initialization because the FRH score still cares about it.
- Intrinsic Fréchet/Karcher mean minimizes weighted squared canonical-metric Stiefel
  distance and iterates

  ```text
  X <- Exp_X(sum_i w_i Log_X(F_i)).
  ```

The repository's experiments found the intrinsic and extrinsic means nearly identical
at the tested concept distances, while gauge alignment helped low-rho concepts under
greedy decoding. That is an empirical finding tied to the recorded models/pools, not a
general theorem.

## Difference subspace and GDS

For two orthonormal subspace bases `Phi` and `Psi`, let

```text
Phi^T Psi = U Sigma V^T,  Sigma_ii = cos(theta_i).
```

Their paired canonical vectors are `a_i = Phi u_i` and `b_i = Psi v_i`. For each
non-shared direction, the classical difference-subspace basis is

```text
d_i = (a_i - b_i) / ||a_i - b_i||.
```

In matrix form,

```text
D = (Phi U - Psi V) [2(I - Sigma)]^(-1/2).
```

The paired principal/mean subspace uses normalized sums instead:

```text
M = (Phi U + Psi V) [2(I + Sigma)]^(-1/2).
```

The same split appears spectrally. With projection matrices

```text
P_1 = Phi Phi^T
P_2 = Psi Psi^T
G   = P_1 + P_2,
```

eigenvectors of `G` have these roles:

- positive eigenvalues below 1: difference directions;
- eigenvalues above 1: principal/mean directions;
- eigenvalue 2: exact intersection directions;
- eigenvalue 1: unmatched dimensions when subspace dimensions differ;
- eigenvalue 0: outside the union span.

For multiple class subspaces, generalized difference subspace uses

```text
G = sum_c P_c.
```

Large-eigenvalue directions capture shared/principal structure; small positive
eigenvalues within the union carry discriminative differences. Fukui's classification
pipeline projects the data or class subspaces onto the selected GDS and then
reorthonormalizes before canonical-angle comparison. It does not use an arbitrary GDS
basis as an ordered FRH guide.

Numerically, intersection and null directions are unstable at exact spectral
boundaries. The papers recommend discarding eigenvalues close to zero and treating the
threshold around one carefully. `frames/linalg/gds.py` uses an explicit `eps` and a
`common_threshold` (default 1.05); the latter is a project heuristic margin, not the
theoretical boundary itself.

GDS projection can be understood as removing common components and
quasi-orthogonalizing class subspaces. The gFDA paper shows, under its non-centered-PCA
mean-direction approximation, that GDS plus a small correction is closely related to a
geometric Fisher discriminant analysis. It also finds normalization of projected
vectors important. This supports projection/reorthogonalization and evaluation, not a
claim that all GDS-derived steering objectives are automatically valid.

## Second-order difference subspace

The 2024 extension first handles intersections and unequal subspace dimensions, then
defines acceleration for three sequential subspaces:

```text
D2(S_1, S_2, S_3) = D(S_2, M(S_1, S_3)),
```

where `M(S_1, S_3)` is their principal/Karcher mean subspace. This mirrors the
Euclidean central second difference: compare the middle state to the mean of its two
neighbors. On a Grassmann manifold, its magnitude can be interpreted as departure
along and away from the geodesic through the outer subspaces.

This construction is useful for future across-layer or across-time representation
dynamics. It is not currently part of the CGD generation loop.

## Implementation crosswalk

| Mathematical object | Implementation | Main tests |
|---|---|---|
| Ordered frame correlation and rho | `frames/linalg/frame.py` | `tests/test_rho.py` |
| Polar/Procrustes projection | `frames/linalg/orthogonalization.py` | composition tests |
| Extrinsic/aligned/Fréchet means | `frames/representations/concept.py`, `frames/linalg/stiefel.py` | `tests/test_composition.py`, `tests/test_stiefel.py` |
| Principal angles and Grassmann metrics | `frames/linalg/angles.py` | `tests/test_angles.py` |
| Difference subspace and GDS split | `frames/linalg/gds.py` | `tests/test_gds.py` |
| Score aggregation and normalization | `frames/representations/aggregators.py` | `tests/test_aggregators.py` |
| Greedy and beam survivor rules | `frames/representations/frame.py` | `tests/test_beam.py`, `tests/test_golden.py` |
| Expression/angle/fluency metrics | `frames/evaluation/harness.py` | `tests/test_harness.py` |

When a new proposal crosses the frame/subspace boundary, state all three choices
explicitly: the represented object (ordered frame or span), the comparison metric, and
the point at which any projection or reorthonormalization occurs.
