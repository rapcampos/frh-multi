# Report & Project Reorganization Plan — Greedy Top-k Core
## July 2026 — planning document, no changes applied

Scoping decision to implement: present the project as **multi-concept
steering and evaluation for the paper's greedy top-k concept-guided
decoder**. The three composition families and the metrics program are
the contributions. The GDS line goes to the PhD proposal. Alternatives
to the greedy top-k loop — beam search and other survivor-rule changes —
go to future research.

This plan lays out: why the scoping is defensible (§1), the reframed
taxonomy it requires (§2), the contribution structure (§3), a
section-by-section outline of the presented report with sources (§4),
the evidence map with tier provenance (§5), cut/keep/move lists (§6),
audits that must be resolved before presenting (§7), decision points
(§8), stale data to fold in (§9), and the document map (§10).

---

## 1. Why the greedy scoping is defensible (and one reassuring fact)

**The control argument.** The program's own design principle (full
report §2) is that each family modifies exactly one ingredient of the
paper's loop, holding the rest fixed — that is what makes the families
commensurable. Holding the *decoder* fixed at the published greedy
top-k is the same logic applied one level up: it isolates the
composition question from the search question. Beam is an orthogonal
axis, demonstrated complementary (it lifts every family), and is
honestly signposted as future work with its teaser numbers. A fixed
published decoder also strengthens the black-box compatibility claim:
the methods drop into the FRH paper's loop as-is.

**The reassuring fact.** The headline validated-guide composition
results are *already greedy-native*: the 62–74% all-improved and joint
0.72–0.92 figures quoted in full-report §5.5 match
`resources/40_greedy_lambda_summary.csv` (notebook 40, greedy λ
composition, n = 2–4, all three families). The core of the presented
story needs **no new headline runs** — the recut is mostly about §5.1
(method development), §5.7 (drop from core), and tier audits elsewhere
(§7 below).

**The anticipated objections, with built-in answers.**
- *"Why not beam, your own data show 4.5×?"* — Scope control (above);
  beam scores a k² candidate pool per step vs greedy's k, so greedy is
  also the cheap tier; beam is shown complementary and deferred.
- *"Greedy λ=0 perplexity is 6–7.5."* — The λ dial is in-core (§2) and
  fixes this at a known, measured cost (ppl ≈ 2.3–3.2 at λ = 0.4).
- *"Greedy joint gains were near zero."* — Only on unvalidated guides;
  that is Contribution 3 (guide validation), not a weakness.

## 2. The reframed taxonomy

The loop has four ingredients: (i) guide frame, (ii) per-step score,
(iii) active concept, (iv) survivor rule. Reorganized scoping:

| ingredient | modified by | status |
|---|---|---|
| guide | F1 blending (simple / aligned / geodesic; subspace-union negative result) | **core** |
| score | F2 voting (normalized vote, softmin, thresholds) **and the λ fluency regularizer** | **core** |
| active concept | F3 scheduling (round-robin, stochastic, sentence-boundary) | **core** |
| survivor rule | beam search, fluency-blended *beam* selection | **future work** |

One deliberate reclassification vs the full report: **λ moves from the
"search dimension" (§2.4) to the score ingredient.** Justification:
`score + λ·log p(token)` is a per-step score modification that runs
inside the unmodified greedy loop — notebooks 40–41 are exactly that,
greedy-native λ at 0–1.0 for all families and n = 2–4. The survivor
rule (best child of the single surviving parent) is untouched. Beam, by
contrast, changes which *sequences* survive. This makes the in/out
boundary principled rather than convenient: **score shaping is in;
search topology is out.**

## 3. Contribution structure

1. **Methods** — a taxonomy and three composition families for
   multi-concept AND steering in a fixed greedy top-k CGD loop, with
   the degeneracy theorem (unnormalized voting ≡ off-manifold blending;
   normalization is what makes voting a distinct family) and the
   subspace-union negative result (composition geometry must match the
   scorer's geometry) as the theory backbone. λ fluency regularization
   as in-loop score shaping.
2. **Evaluation** — the metrics program as a first-class contribution:
   presence falsified → expression Δ (baseline-subtracted,
   continuation-only) → bounded gauge-invariant angle metrics with
   k/d-null calibration; dual-track logging; the three artifacts caught
   (metric-scale illusion, quantization artifact, min-shrinkage
   confound); perplexity guardrail.
3. **Guide validation** — steerability is not free from the lexicon:
   the mining/validation pipeline, 1-of-27 legacy survival, best
   offline predictor only r ≈ +0.3, and the headline reversal (Llama
   joint ≈ 0.02 → 0.72–0.92 with validated guides). Selection-metric
   robustness (Δ vs d_cos pools, notebooks 36–38).
4. **Findings** — greedy-tier family ranking on validated guides
   (blend > schedule > vote), the λ Pareto dial, scaling n = 2→4,
   quantization robustness, and the cross-family composability gap.

Suggested working title: *"Multi-Concept Concept-Guided Decoding:
Composition and Evaluation under the Frame Representation Hypothesis."*

## 4. Presented-report outline, with source mapping

Sources refer to sections of `research-report-full.md` (FR) unless
noted. "Recut" = rewrite anchored to greedy-tier evidence.

1. **Foundations** — FR §1 unchanged.
2. **Problem & design principle** — FR §2 preamble; add the scoping
   statement (decoder held fixed; search deferred, see §Future).
3. **Methods** — FR §2.1–2.3 essentially unchanged; §2.4 shrinks to a
   short "λ score shaping" subsection (greedy framing); beam text moves
   to Future Work.
4. **Evaluation methodology** — FR §3 unchanged (already tier-neutral).
5. **Infrastructure** — FR §4; drop or soften the beam-specific golden
   pin phrasing if needed (the pin itself is the greedy loop — keep).
6. **Findings**
   - 6.1 Method development, greedy tier — recut of FR §5.1: 30-pair
     results (notebooks 16–21), greedy family ranking, near-zero joints
     on legacy guides *as motivation for guide validation*. Drop the
     beam recipe sentence.
   - 6.2 Quantization — FR §5.2 (greedy-era experiments; keep,
     including the aligned-blend artifact dissolution).
   - 6.3 Metric validation — FR §5.3 unchanged (re-scoring,
     tier-agnostic).
   - 6.4 Guide quality — FR §5.5, now load-bearing; verified
     greedy-native (notebook 40 numbers).
   - 6.5 Scaling n = 2→4 — re-anchor on notebooks 40–41 (greedy;
     includes n = 4, which FR still lists as open — see §9). Blend
     holds ~62–74% all-improved across n; schedule degrades ~1/n;
     vote stable but last.
   - 6.6 The λ dial — new subsection from notebooks 40/41 (+ greedy
     rows of 31 for singles): singles Δ 0.41→0.81 with ppl 7.5→2.3 at
     λ = 0.4; composition λ-response curves; λ ≈ 0.3–0.4 as the
     recommended operating point. Under greedy the dial is *better than
     free* — steering and fluency improve together on singles.
   - 6.7 Negative steering, greedy tier — recut of FR §5.1's negative
     clause to notebook 22 (E3 greedy) findings only.
   - 6.8 Cross-model — anchor on **notebook 43** (running as of
     2026-07-20): fresh tier-consistent design, all four models (Llama
     bf16 + AWQ, Gemma-2-2B/9B) × 30 ρ-spanning validated-pool pairs ×
     three methods, greedy λ=0. Sole cross-model anchor — per user
     decision (2026-07-20), legacy-pool results (25/26/29) are not
     presented; they are mentioned only qualitatively as superseded.
     Claims in both-improved rates and angle units only.
     Carry 43's own caveats: the pool is *Llama-validated* (Gemma is
     tested on Llama-favorable concepts), and the lemma-policy
     asymmetry (Llama 8-language, Gemma English-only) is deliberate but
     uncontrolled — see audit 6.
7. **Conclusions** — recut of FR §6: drop "search contributed more than
   any representational refinement" (conclusion 2's beam sentence);
   promote guide-quality and evaluation conclusions; keep
   geometry-must-match, family gap.
8. **Future work**
   - Survivor-rule upgrades: beam search — with teaser numbers (27/27
     concepts, ~4.5× singles Δ, dominance of beam+λ0.4; FR §5.7 /
     notebooks 22–23, 31, 42) as *motivation*, explicitly measured but
     out of scope.
   - GDS / operator–objective correspondence — one paragraph + pointer
     to `phd-proposal-ideas.md` (FR §7 text moves there or stays as
     summary).
   - Model-invariance program — pointer to
     `model-invariance-assessment.md` (audits, relational RSA,
     shared-pool IRT).
   - Remaining opens: simplex steering, usage frames, placebo control,
     more architectures.

## 5. Evidence map with tier provenance

| notebook(s) | content | tier | status for core |
|---|---|---|---|
Tier provenance verified 2026-07-20 by grepping every flagged notebook
for the beam API (`generate_with_topk_beam_guide`) and `lam=` markers:
**beam only ever appears in notebooks 23, 31 (partly), and 42** — the
program is greedy-native everywhere else.

| notebook(s) | content | tier | status for core |
|---|---|---|---|
| 16–21 | E0/E1/E2 pilots, family sweep, fluency pareto (+ unquantized reruns) | greedy | core (6.1, 6.2) |
| 22 | E3 negative steering | greedy ✓ | core (6.7) |
| 23 | E3b negative under beam | beam | future work |
| 24 | showcase, 30 pairs | greedy-era | core showcase |
| 25 / 26 / 29 | multimodel, Gemma-9B, Llama bf16 | greedy ✓ | superseded by 43; not presented (user decision) |
| 27 / 32 | pool validation AWQ / bf16 | greedy ✓ | core 6.4 (in-scope decoder) |
| 28 / 33 / 38 / 39 | scaling 3, dcos-pool scaling, scaling 4 | greedy ✓ | supporting 6.5 |
| 30 / 36 | angle re-score + plots | n/a (re-scoring) | core 6.3 |
| 31 | singles 2×2 (search × λ) | both | greedy rows → core 6.6; beam rows → future |
| 34 / 35 | GDS pilots | n/a | PhD proposal |
| 37 | d_cos pool selection | no generation ✓ | core 6.4 robustness |
| 40 / 41 | greedy λ composition + λ sweep, n = 2–4 | greedy ✓ | **core headliners (6.4–6.6)** |
| 42 | beam λ sweep | beam | future work |
| 43 | cross-model on validated pool, 4 models × 3 methods | greedy λ=0 ✓ (running) | **core headliner (6.8)** |

Figure shortlist for the presented report: `40_greedy_lambda.png`,
`41_lambda_response.png`, `18_e2_method_gap_vs_rho*.png`,
`21_e1b_lambda_dial*.png`, `16/17_e0_gap_vs_rho*.png`, pool-pipeline
figure from 27/32/37, scaling figure from 40/41.

## 6. Cut / keep / move lists

**Cut from core** (beam-derived; must not appear as core claims): the
"beam + simple blend + λ ≈ 0.4" recipe line; "beam beats greedy 27/27";
"under beam the fluency dial is free"; the beam rows of the §5.7 table;
E3b clean-rate claims; conclusion 2's search sentence. All reappear,
clearly labeled, in Future Work.

**Keep in core**: degeneracy theorem; subspace-union lesson; presence
falsification; expression + angle metrics with calibration;
quantization findings incl. aligned-blend artifact; metric-scale
illusion; min-shrinkage caveat; pool pipeline and 1/27; greedy λ
effects; n = 2–4 scaling; family gap (rates/angles only); one
tier-consistent wrinkle to report honestly: on *legacy* guides, vote
won the greedy tier on bf16 Llama (FR §5.4) while blend wins on the
validated pool — report both, do not smooth over.

**Move**: FR §7 GDS text → `phd-proposal-ideas.md` (or keep as
one-paragraph pointer); beam machinery description (FR §2.4) → Future
Work; `model-invariance-assessment.md` → cited under Future Work.

## 7. Audit checklist — status as of 2026-07-20

1. **Pool validation tier (notebooks 27, 32, 37).** ✅ **Resolved:
   greedy.** 27/32 call only `generate_with_topk_guide`; 37 runs no
   generation (re-scores existing logs). Guides were selected under the
   in-scope decoder — no transfer argument or re-validation needed.
2. **Cross-model tier (notebooks 25, 26, 29).** ✅ **Resolved: all
   greedy** — and superseded for the headline by notebook 43 (fresh
   tier-consistent design on the validated pool, running now). 25/26/29
   remain as the legacy-pool (weak-guide) reference row.
3. **Scaling notebooks 28/33/38/39.** ✅ **Resolved: all greedy.**
4. **Showcase / cross-model 43.** ✅ **Resolved: greedy λ=0** (no beam,
   no λ in the notebook). Note: λ=0 is fine for the rate-based
   cross-model claims; the λ recommendation stays anchored on 40/41.
5. **Negative-steering provenance.** ⚠️ **Editorial only** — tier split
   confirmed (22 greedy, 23 beam); remaining work is attributing FR
   §5.1's two clauses ("Π(A−B) maximizes net contrast" vs "λ = 0.4
   maximizes the strict clean rate") to the correct notebook so only
   the greedy part stays in core.
6. **Lemma-policy confound.** ⚠️ **Premise confirmed, control open.**
   Notebook 43's caveat block documents the asymmetry as deliberate
   (Llama 8-language min_lemmas=11, Gemma English-only min_lemmas=4,
   matching 25/26). The matched-policy control — Llama English-only —
   is the one substantive open experiment before the family-gap claim
   is presented without qualification. Until run, 6.8 carries the
   caveat verbatim.

## 8. Decision points (recommendations first)

1. **λ in core?** — **Yes (recommended).** Greedy-native (40/41), fixes
   the fluency objection, gives the report its trade-off curve, and the
   score-shaping classification keeps the taxonomy principled.
   Alternative: pure λ=0 core with fluency as a limitation — weaker.
2. **Beam numbers in Future Work: teaser or appendix?** — **Teaser
   (recommended):** the 27/27 and ~4.5× figures in two sentences,
   clearly labeled preliminary/out-of-scope. Full tables stay in the
   program archive (FR) and notebooks. Hiding them entirely would be
   misleading; foregrounding them undermines the scoping.
3. **Headline numbers.** Recommended: "62–74% all-improved (n = 2–4,
   blend, validated guides, Llama bf16 greedy)" + singles λ effect
   (0.41→0.81, ppl 7.5→2.3) + cross-model rates pending audit 2.
4. **Which document becomes the presented report?** Recommended: write
   the new outline as a fresh document (e.g.
   `plans/report-greedy-core.md`) and leave `research-report-full.md`
   untouched as the unabridged program archive; update
   `research-report-2026-07.md` (the ~2000-word version) to the new
   framing afterwards, since short versions drift worst.

## 9. Stale data to fold in

- **n = 4 exists.** FR §7 lists "n = 4 scaling" as open, but notebooks
  39–41 contain n = 4 (40/41 summaries have full n = 2/3/4 blocks).
  Fold into §6.5 and delete from the open list.
- The greedy λ=0.4 singles row (notebook 31) was recently added to FR
  §5.7 — in the new structure it moves *into* core §6.6 while the beam
  rows move out.

## 10. Document map after reorganization

| document | role | audience |
|---|---|---|
| `research-report-full.md` | unabridged program archive (keeps beam + GDS; untouched) | self, collaborators |
| **new** `report-greedy-core.md` | the presented project (this plan's §4 outline) | presentation, paper draft |
| `research-report-2026-07.md` | short version, re-cut to greedy framing after the core doc exists | quick readers |
| `phd-application-statement.md`, `phd-proposal-ideas.md` | GDS line + beam/search line + model-invariance as proposal material | admissions |
| `multiconcept-cgd-methods.md`, `multiconcept-steering-math.md`, `metrics-derivations.md`, `metrics-explained.md` | supporting/reference; mark beam sections "future-work material", no rewrite needed | reference |
| `model-invariance-assessment.md` | future-research design doc, cited from Future Work | self, proposal |

### Suggested execution order (updated 2026-07-20)

1. ~~Tier audits~~ — done; all core evidence is greedy-native (§5, §7).
2. Wait for notebook 43 to finish; fold its table into §6.8.
3. Draft `report-greedy-core.md` from the §4 outline (recuts of §6.1,
   6.5–6.7 are the only real writing; the rest is transplant). Do the
   §7.5 clause attribution while recutting 6.7.
4. Run the lemma-policy matched control (§7.6) — the only open
   experiment gating an unqualified family-gap claim.
5. Re-cut the short report; move GDS text to the proposal doc.
