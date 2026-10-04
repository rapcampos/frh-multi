# Results to Date — Ranked by Impact

Status: August 2026. This ranking covers the current thesis scope: multi-concept
composition inside the FRH paper's fixed greedy top-k decoder. Beam search is
reported separately because it changes the survivor rule rather than the
composition mechanism.

## 1. Validated guides make multi-concept steering work through four concepts

This is the central result. On bf16 Llama-3.1-8B-Instruct, simple frame blending
can raise all target concepts together for groups of two, three, and four. With
pure steering, mean joint expression (the weakest target's improvement) is
**0.93, 0.82, and 0.73** for n = 2, 3, and 4. The corresponding bounded,
gauge-invariant all-positive rates are **76%, 70%, and 68%**. Each point uses 30
groups x 10 prompts.

Guide validation is load-bearing, not a preprocessing detail. Only two of the
27 legacy concepts enter the final 40-guide bf16 pool. At the same model and
decoder tier, the older legacy-pair showcase obtained joint expression 0.27 for
blending, versus 0.93 on the validated pair evaluation. These are different
pair sets, so the change is not a paired causal estimate, but it shows that weak
guides can hide the capability being studied.

Evidence: notebooks 29, 32, 51, and 52;
`resources/29_multimodel_summary.csv`,
`resources/32_concept_pool_bf16.json`, and
`resources/52_composition_summary.csv`.

## 2. The fluency weight turns steering into a practical trade-off

The score modification

\[
S(c) = S_{\mathrm{concept}}(c) + \lambda \log p(c)
\]

provides a smooth control between concept expression and model likelihood while
leaving the greedy survivor rule unchanged. For frame blending, **lambda = 0.4
cuts the perplexity ratio from about 5.9--6.2x to 2.8--3.2x**, while retaining
about **89--92%** of joint expression. At that setting, joint expression is
0.83, 0.75, and 0.64 for n = 2--4, with bounded all-positive rates of 75%, 68%,
and 62%. The useful operating range is lambda about 0.3--0.4; larger values
continue improving fluency but increasingly erase steering.

Evidence: notebooks 40, 41, 44, and 52;
`resources/41_lambda_sweep_summary.csv`,
`resources/44_singles_lambda_summary.csv`, and
`resources/52_composition_summary.csv`.

## 3. Simple frame blending is the strongest default composition method

Across n = 2--4, with lambda both off and at 0.4, frame blending leads score
voting and round-robin scheduling on joint expression and token-cosine change.
Scheduling weakens most as n grows because each concept receives only about
1/n of the decoding steps. Voting is more stable but remains below blending.

The theory explains an important boundary between the methods: because the FRH
score is linear in the guide frame, an *unnormalized* sum of concept scores is
exactly the same operation as scoring against the summed frame. Per-step
normalization is therefore what makes voting a genuinely distinct family.
Likewise, a Grassmannian union of concept subspaces is mismatched to the
gauge-sensitive Stiefel scorer and fails despite containing the constituent
spans.

More elaborate blend geometry has not earned its cost. On the current bf16
evaluation, extrinsic, Procrustes-aligned, and intrinsic Frechet blends have
nearly identical success rates and fluency. The earlier low-similarity advantage
for alignment largely disappears without AWQ quantization, so the cheap
extrinsic mean remains the defensible default.

Evidence: notebooks 17--20, 30, 39, 50, and 52;
`plans/multiconcept-cgd-summary.md`,
`resources/50_blend_geometry_summary.csv`, and
`resources/52_composition_summary.csv`.

## 4. The effect generalizes across model families, but composability is model-dependent

The same greedy composition mechanisms improve both concepts on Llama-3.1-8B
(AWQ and bf16) and Gemma-2 (2B and 9B). In bounded angle units, both-concepts-
improved rates span **74--79% on Llama AWQ, 69--72% on Llama bf16, 88--92% on
Gemma 2B, and 97--99% on Gemma 9B**, depending on the composition method.

An English-only Llama control lowers rather than closes the gap, so the Gemma
advantage is not explained by the earlier lemma-policy mismatch. The remaining
caveat is that the shared concept names were selected for steerability on Llama;
the comparison is therefore evidence of model-family variation, not a neutral
ranking of model quality. Raw expression magnitudes are also not comparable
across models; rates and bounded angle metrics are the valid currency.

Evidence: notebooks 43 and 46;
`resources/43_validated_pool_summary.csv` and
`resources/46_lemma_policy_summary.csv`.

## 5. The evaluation program removed several plausible but false conclusions

Surface-lemma presence floors at zero even when generations visibly change, so
it cannot serve as the main success metric. The project now uses
baseline-subtracted continuation expression, bounded gauge-invariant angle
metrics, weakest-target joint scores, and continuation-only perplexity.

These controls changed the scientific interpretation. The apparent near-2x
increase in raw steering from AWQ to bf16 is mostly hidden-state metric scale:
the blend's token-cosine change is 0.0076 on AWQ and 0.0070 on bf16. The
subspace-level movement is moderately larger on bf16, but the original
"twice as strong" claim is not supported. The same audit identified the
low-similarity aligned-blend result as substantially quantization-dependent.

Evidence: notebooks 16, 17, 29, 30, and 36;
`plans/quantization-impact-report.md` and
`resources/30_angle_summary.csv`.

## 6. Negative steering suppresses reliably, but clean two-sided control is not solved at the greedy tier

For attraction toward A while repelling B, the frame difference A - B gives the
best greedy result. Its mean net contrast is **0.85 on AWQ and 0.79 on bf16**,
versus 0.45 and 0.41 for attraction alone. However, repulsion often suppresses
the desired concept too, and the strict clean rate remains only about 31--35%.
This establishes effective suppression but leaves selective attraction plus
repulsion as an open problem, especially for geometrically similar concepts.

Evidence: notebook 22; `resources/22_e3_summary.csv` and
`resources/22_e3_summary_unquantized.csv`.

## High-impact result reserved for future work: beam search

Changing the survivor rule to true beam search is the largest measured upgrade,
but it is outside the fixed-greedy thesis comparison. With frame blending and
lambda = 0.4, beam obtains joint expression **1.69, 1.52, and 1.41** for n =
2--4, with bounded all-positive rates **85%, 85%, and 82%** at roughly the same
2.7--3.2x perplexity cost as greedy. It also makes mild fluency regularization a
regularizer: lambda = 0.3 can improve both expression and fluency relative to
lambda = 0. This is the strongest next research direction, with the trade-off
of scoring a larger candidate tree.

Evidence: notebooks 42 and 45;
`resources/42_beam_lambda_summary.csv` and
`resources/45_singles_2x2_bf16_summary.csv`.

## Bottom line

The current thesis claim is defensible and compact: **validate concepts first,
blend their frames, use lambda about 0.3--0.4 when fluency matters, and evaluate
with bounded angle metrics plus perplexity**. Under the published greedy top-k
decoder, that recipe composes two to four concepts across two model families.
The largest remaining limitations are the small fixed prompt set, short
continuations, model-specific internal access, WordNet-centered concepts, and
the absence of human fluency/semantic judgments.
