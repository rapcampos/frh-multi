# Model-Invariant Concept Information: Assessment
## July 2026

Context: the concept pools are validated per model variant (AWQ-validated
pool, notebook 27; bf16-validated pool, notebooks 32/37). Question posed:
can we extract concept/word-relation information that is model-invariant —
or at least more robust — to enable reliable model-wise comparisons?

Core of the assessment: **"model-invariant" hides three different
questions** — (i) statistics that numerically agree across models
(strong), (ii) statistics whose relational structure correlates across
models (weak, RSA-style), (iii) a model-free reference structure that
each model is scored against (external). Separating them changes what is
possible, what is cheap, and what is a thesis chapter. Reliable
model-wise comparison needs (iii), or a common scale built from
(ii) + IRT — it does not need (i).

---

## 1. What is invariant in principle — and what can never be

The whitening step (the causal inner product) removes the arbitrary
affine gauge *within* a model, but it identifies the geometry only up to
a residual **global rotation**: even if two models implemented identical
semantics, their whitened spaces would be related by an unknown isometry
(different d across families makes it worse). Consequence: **no
first-order object — a vector, a frame — is ever comparable across
models. Only isometry-invariants are.** The complete isometry-invariants
of a constellation of concepts are its *relational* statistics: pairwise
correlations, principal angles, distances. So "model-invariant word
relations" is not just a nice-to-have direction — it is the *only* thing
that can be extracted, mathematically. The v3 machinery (Grassmann
angles, mcc) is exactly the right instrument, because Grassmann
quantities are additionally invariant to the per-frame gauge, which
matters across models: tokenizers assign the same lemma different token
counts, so the "same" concept does not even have the same rank across
models — Stiefel-level cross-model identity is ill-defined from the
start.

Second distinction: **representational vs behavioral invariance**. The
relational geometry of concepts is a property of the unembedding;
*steerability* is a property of (concept × model × decoding stack ×
prompts) jointly — behavior, not representation. Asking for
model-invariant steerability is like asking for drug efficacy
independent of the organism. The right move is not to make it invariant
but to **decompose** it and measure how much is concept-intrinsic. The
two can dissociate, and §5.4 of the full report hints they do — the most
interesting scientific angle here (§4 below).

## 2. Two audits to run before any new design

**A possible confound in the existing cross-family comparisons.**
`SupportedLanguages.from_model_id` gives Llama 8 languages and Gemma
English-only — so if notebooks 25/26 built concepts the default way, the
Gemma and Llama frames for the *same synset* were built from *different
lemma sets*. The 99%-vs-51% family gap was then measured on
non-identical concepts. Audit this; if real, a matched-lemma-policy
rerun (English-only for both) is a mandatory control before the gap goes
in any paper.

**Reliability before validity.** Per-concept steerability is a noisy
estimate, and its test-retest data already exists: the AWQ and bf16
validation runs of the *same weights* (notebooks 27 vs 32). Correlating
per-concept Δ across them gives the measurement's reliability — the
**ceiling** for any cross-model correlation. If AWQ↔bf16 of the same
model correlates only 0.6, then a Gemma↔Llama correlation of 0.5 is
near-ceiling, and "steerability is model-specific" would be the wrong
conclusion — measurement noise. Classic attenuation correction applies:
`r_true = r_obs / √(r_AA · r_BB)`. Also compute split-half reliability
within a single run (split the prompt battery) — free. Critically: run
all correlations on the **full validated slate (~160), never the
selected 40** — the 0.04-that-was-really-range-restriction incident
(notebook 37) will recur with a vengeance on a top-40 pool.

## 3. Designs, cheapest first

**3.1 Reanalyses (zero GPU).** The reliability ladder: split-half →
cross-precision → cross-model-within-family → cross-family. Each rung
isolates one variance source; the output is a variance decomposition —
"X% of steerability variance is concept-intrinsic" — the quantitative
answer to the invariance question.

**3.2 Relational RSA (cheap — frame construction only, no generation).**
Per model, over one fixed synset slate under a matched lemma policy: the
n×n matrix of pairwise mcc / geodesic distances between concept
subspaces, calibrated against that model's empirical null (the k/d floor
differs per model). Compare *matrices* across models (Spearman on
off-diagonals, or CKA), and three-way against WordNet path/hypernym
similarity as the model-free anchor. Directly tests whether
word-relation geometry is shared even where steerability is not. This is
precisely the test the **Platonic Representation Hypothesis** (Huh et
al. 2024) predicts should pass — FRH gives it a lexicon-grounded,
multi-token testbed, a genuinely publishable framing.

**3.3 Anchor coordinates (cheap; a working invariant representation).**
Moschella et al.'s *relative representations* (ICLR 2023): pick a fixed
anchor set of high-reliability concepts; represent every concept in
every model by its similarity vector to the anchors. Those coordinates
are model-invariant by construction and support matching, clustering,
and — the useful test — *predicting* one model's steerability from
another model's coordinates.

**3.4 Shared-pool + IRT (the instrument for reliable model-wise
comparisons).** Validate one slate on all models; fit the success
indicators (the all-positive rate is already Bernoulli) with a
**Rasch/logistic model**: `success ~ concept_difficulty +
model_ability`. Item-response theory was invented for exactly this
problem — comparing test-takers when raw scales are not shared. Yields:
concept difficulties on a common scale (the invariant information),
model abilities (the defensible cross-model ranking), and **differential
item functioning** flags — concepts anomalously hard for one family — a
formal, per-concept detector for the Gemma/Llama gap.

**3.5 Alignment and transport (ambitious pilot).** Align two models'
whitened unembedding spaces over shared vocabulary (orthogonal
Procrustes, or CCA to a common rank since d differs), then *transport* a
concept frame from Llama into Gemma's space and test whether it steers.
The strong operational definition: a concept is invariant iff its
transported rendering works. Connects to model stitching (Lenc &
Vedaldi; Bansal et al.) and the vec2vec line (unpaired embedding
translation). Highest risk, most novel.

## 4. The payoff question

Running 3.1–3.2 localizes where invariance breaks along the stack:
lexicon → tokenization → relational geometry → trajectory geometry →
decoding. If relational RSA says Gemma↔Llama agree (say ρ ≈ 0.7) while
disattenuated steerability correlates ≈ 0.2, the family gap is
**dynamical, not representational** — and the obvious mechanism is
candidate-pool diversity: steering is re-ranking, so its leverage is
bounded by the entropy of the model's top-k proposals; a model with
peaked proposals is unsteerable by re-ranking no matter how clean its
geometry. If per-candidate scores are in the probe traces this is
measurable from existing logs; if not, it is a one-flag logging change.
This would convert conclusion #6 of the full report from "unexplained"
to a mechanistic, testable claim — and "which levels of concept geometry
are model-invariant" slots directly into the PhD proposal set next to
the hierarchy-induction thread.

## 5. Recommendation

1. The two audits first (lemma-policy check; AWQ↔bf16 reliability) —
   hours, not days, zero GPU.
2. Relational RSA (3.2).
3. Shared-pool IRT design (3.4) as the definitive cross-model
   instrument.
4. Anchor coordinates (3.3) and transport (3.5) as follow-ons / PhD
   material.

## References

- Huh, Cheung, Wang & Isola, *The Platonic Representation Hypothesis*,
  ICML 2024.
- Moschella et al., *Relative Representations Enable Zero-Shot Latent
  Space Communication*, ICLR 2023.
- Kornblith, Norouzi, Lee & Hinton, *Similarity of Neural Network
  Representations Revisited* (CKA), ICML 2019.
- Kriegeskorte, Mur & Bandettini, *Representational Similarity
  Analysis*, Frontiers in Systems Neuroscience, 2008.
- Lenc & Vedaldi, *Understanding Image Representations by Measuring
  Their Equivariance and Equivalence*, CVPR 2015; Bansal, Nakkiran &
  Barak, *Revisiting Model Stitching*, NeurIPS 2021.
- Jha et al., *Harnessing the Universal Geometry of Embeddings*
  (vec2vec), 2025.
- Rasch, *Probabilistic Models for Some Intelligence and Attainment
  Tests*, 1960 (IRT); Spearman, 1904 (attenuation correction).
