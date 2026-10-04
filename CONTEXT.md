# FRH Multi-Concept Steering

Research codebase extending the Frame Representation Hypothesis's
Top-k Concept-Guided Decoding to multi-concept steering. The thesis
keeps the paper's decoding loop fixed and modifies only the
composition-facing ingredients.

## Language

### Decoding loop

**Top-k Concept-Guided Decoding (CGD)**:
The FRH paper's full generation loop: propose top-k candidates, score
each against the guide, apply the survivor rule, repeat.
_Avoid_: top-k guide, guided generation (ambiguous with single steps)

**Ingredient**:
One of the four modifiable parts of the CGD loop: the guide, the
score, the active concept, or the survivor rule. Each method family
modifies exactly one.

**Guide**:
The concept frame the loop steers toward; composed frames (blends)
are still guides.
_Avoid_: target, steering vector

**Score**:
The per-step scalar that rates a candidate: cumulative frame
correlation, optionally plus the fluency term λ·log p.

**Active concept**:
The concept whose score counts at a given step; only schedulers make
this vary over time.

**Survivor rule**:
The per-step decision of which candidate sequence(s) continue to the
next step. The paper's rule — held fixed throughout the thesis — is
greedy: keep the single best child of the single current sequence.
_Avoid_: search algorithm, decoding algorithm (those name the whole
loop)

**Greedy tier / beam tier**:
Provenance labels for results, by the survivor rule used to produce
them. Thesis results are greedy-tier only; beam-tier is future work.

**Fluency weight (λ)**:
The score-shaping coefficient on cumulative token log-likelihood,
added to mitigate the perplexity increase of steering. λ = 0 is pure
steering; λ → ∞ recovers unsteered decoding. A score modification,
not a survivor-rule change.
_Avoid_: fluency-blended selection (conflates it with the survivor
rule)
