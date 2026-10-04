# Notebook index

One row per notebook. Numbering is **frozen** — `resources/` artifact names embed
the producing notebook's number, so notebooks are never renumbered. Retired
notebooks live in `archive/` (see `archive/README.md` for the old→new map);
their `resources/` outputs stay in place and keep feeding active notebooks.

Status legend: **foundation** (paper era, kept as-is) · **active** (canonical /
load-bearing for the multi-concept program) · **future-work** (unique data
backing the future-work tier) · **archived** (→ path under `archive/`).

Notebooks 17–23 are **parameterized**: one notebook covers the AWQ-INT4 and
bf16 runs via a papermill `QUANTIZATION` parameter (`"awq"` | `"bf16"`);
bf16 outputs carry the `_unquantized` suffix in `resources/`. Headless:
`make run_notebook N=<nb>.ipynb P=bf16`. The pre-merge originals are in
`archive/premerge/`, executed run records in `archive/executed/` and `outputs/`.

## Foundation — paper era (01–14, plotting)

| NB | Title | Status | Notes |
|----|-------|--------|-------|
| 01 | START_HERE — model downloader | foundation | Downloads all models in `models.yaml` |
| 02 | Tokenization frames | foundation | |
| 03 | Concept vs word frames relationship | foundation | |
| 04 | Model family concept space comparison | foundation | |
| 05 | Guided generation samples | foundation | |
| 06 | Top-k guided generation ablation (k) | foundation | |
| 07 | Guided generation language ablation | foundation | |
| 08 | Guided generation model comparison | foundation | |
| 09 | Language × k ablation | foundation | |
| 10 | Multilingual SafeBench ablation | foundation | `frames.experiments.figstep_cdg_comparison` needs Python ≥ 3.12 (`itertools.batched`); fails on the current 3.11 venv |
| 11 | Token superposition analysis | foundation | |
| 12 | Multiconcept guided generation (pre-E0) | foundation | References the removed cascade API — not re-runnable as-is |
| 13 | WordNet lemma exploration | foundation | |
| 14 | Golden reference (single-concept CGD) | active | Pins `tests/golden/` for the `@pytest.mark.gpu` regression — load-bearing test infra |
| — | plotting.ipynb | foundation | Regenerates paper figures from `resources/` |

## Multi-concept program — E-series (15–23)

All Llama-3.1-8B-Instruct; 17–23 parameterized AWQ/bf16 (see above).

| NB | Title | Exp | Status | resources/ in → out | Notes |
|----|-------|-----|--------|--------------------|-------|
| 15 | E0 pair selection | E0 | active | → `15_e0_selected_pairs.json` | Read by 17–23 and 48 |
| 16 | E0 pilot (presence metric) | E0 v1 | archived → `archive/superseded/` | → `16_e0_*` | Metric floored; superseded by 17 |
| 17 | E0 pilot v2 (expression metric) | E0 v2 | active | `15_…pairs.json` → `17_e0_*{,_unquantized}` | Blending vs voting, k ∈ {3,4} |
| 18 | E2 Riemannian aggregation | E2 | active | `15_…pairs.json` → `18_e2_*{,_unquantized}` | Metric vs gauge (RQ3) |
| 19 | E2b ρ-adaptive crossover | E2b | active | `15_…pairs.json` → `19_e2b_*{,_unquantized}` | Crossover ρ≈0.33 on AWQ; substantially a quantization artifact on bf16 — the pair of runs together is the finding |
| 20 | E1 family sweep (Pareto) | E1 | active | `15_…pairs.json` → `20_e1_*{,_unquantized}` | Beam dominates greedy; unquantized JSONL feeds 49 |
| 21 | E1b fluency dial (λ) | E1b | active | `15_…pairs.json` → `21_e1b_*{,_unquantized}` | λ≈0.4 guardrail; hardcoded E1 reference values in comments are AWQ |
| 22 | E3 negative steering (greedy) | E3 | active | `15_…pairs.json` → `22_e3_*{,_unquantized}` | Merged notebook standardizes `22_e3_*` paths; the historical AWQ run wrote `21_e3_*` (pre-rename, see `archive/README.md`) |
| — | 22_results_report | E0–E3 | active | `17/18/19/20/21/22_*` → `22_master_summary.csv` + figures | Reproduces the report numbers |
| 23 | E3b negative under beam + λ | E3b | future-work | `15_…pairs.json`, `22_e3_summary{SUFFIX}.csv` → `23_e3_beam_*{,_unquantized}` | Final-recipe negative steering |

## Showcases & pool era (24–33)

| NB | Title | Status | resources/ in → out | Notes |
|----|-------|--------|--------------------|-------|
| 24 | Showcase: 30 pairs, 3 families (AWQ) | active | → `24_showcase.jsonl` | Read by 30 |
| 25 | Showcase multimodel | archived → `archive/superseded/` | → `25_multimodel.jsonl` | Legacy pool; 43 is the sole cross-model anchor |
| 26 | Showcase Gemma-2-9B | archived → `archive/superseded/` | → `26_gemma9b.jsonl` | Same decision |
| 27 | Steering concept pool (AWQ) | archived → `archive/superseded/` | → `27_concept_pool.json` | Replaced by 32 (bf16 validated pool) |
| 28 | Scaling to 3 concepts (AWQ) | archived → `archive/superseded/` | `27_concept_pool.json` → `28_scaling3.jsonl` | Replaced by 33 |
| 29 | Showcase Llama-8B bf16 | archived → `archive/superseded/` | → `29_llama8b_bf16.jsonl` | Read by 30/37; legacy pool |
| 30 | Angle metric validation (v3) | active | `24_showcase.jsonl`, `29_llama8b_bf16.jsonl` → | Corrected the "bf16 steers ~2×" reading |
| 31 | Single-concept beam 2×2 | archived → `archive/superseded/` | `24_showcase.jsonl` → `31_single_beam.jsonl` | Reconstituted by 44/45 on the validated pool; origin of the "27/27, ~4.5×" teaser |
| 32 | Concept pool bf16 (validated) | active | `27_concept_pool.json` → `32_concept_pool_bf16.json` | THE validated pool — read by 33/39/43/44/45/47/48 |
| 33 | Scaling3 bf16 + GDS variants | active | `32_…bf16.json` → `33_scaling3_bf16.jsonl`, `33_scaling_groups_bf16.json` | bf16 scaling baseline; groups read by 40/41/42 |

## Branches & metric follow-ups (34–38)

| NB | Title | Status | Notes |
|----|-------|--------|-------|
| 34 | Common-blend pilot (Branch A) | archived → `archive/branches/` | Parked for PhD proposal, not refuted |
| 34b | Common/difference joint expression through n=4 | future-work | `33/34/39_*` → `34b_gds_blend_n4.jsonl`, joint-expression summaries + plot; adds the missing n=4 difference-only arm, no rho analysis |
| 35 | Hierarchy alignment pilot (Branch C) | archived → `archive/branches/` | Same |
| 36 | Angle-metric (v3) figures | active | GPU-free plotting for already-logged runs |
| 37 | Pool selection by d_cos | active | → `37_pool_dcos.json`; robustness control (Δ-pool stands) |
| 38 | Scaling with d_cos pool | active | `37_pool_dcos.json` →; supporting result |

## Greedy core & midterm (39–55)

All bf16 Llama-3.1-8B unless noted.

| NB | Title | Status | resources/ in → out | Notes |
|----|-------|--------|--------------------|-------|
| 39 | Scaling to n=4 | active | `32_…bf16.json` → `39_scaling4.jsonl`, `39_quads.json` | Quads read by 40/41/42 |
| 40 | Greedy λ dial for composition | active | `33_scaling_groups…`, `39_quads.json` → | Core headliner; midterm λ figure |
| 41 | λ precision sweep 0.3–1.0 | active | `39_quads.json` → | Core headliner; λ operating point |
| 42 | Beam-tier λ sweep | future-work | `33_scaling_groups…`, `39_quads.json` → | |
| 43 | Showcase: validated pool, cross-model | active | `32_…bf16.json` → `43_validated_pool.jsonl`, `43_pairs.json` | Canonical cross-model endpoint (Llama + Gemma) |
| 44 | Singles λ (bf16 pool) | active | `32_…bf16.json` → `44_singles_lambda_bf16.jsonl` | Presented singles numbers |
| 45 | Singles beam (bf16 pool) | future-work | `44_…jsonl` → | Completes 31's 2×2 on the validated pool |
| 46 | Lemma-policy control | active | `43_pairs.json`, `43_validated_pool.jsonl` → `46_lemma_policy_summary.csv` | Gates the family-gap claim |
| 47 | Qualitative pool gallery | active | `32_…bf16.json` → | Midterm inline example |
| 48 | Midterm figures & tables | active | `15/32/43/46` outputs → `48_crossmodel_table.csv`, `48_slate_histogram.png` | |
| 49 | E1 method heatmap (unquantized) | active | `20_e1_family_sweep_unquantized.jsonl` → | GPU-free; papermill smoke test |
| 50 | Blend geometry — E2 redo (bf16 pool) | active | `43_pairs.json` → `50_blend_geometry_*` | extrinsic vs aligned vs frechet under current conventions; tests whether any E2b crossover survives on clean weights in joint_z and joint token-cos; frechet≈extrinsic equivalence |
| 51 | n=2 top-up for the scaling arm | active | `32_…bf16.json`, `33_scaling_groups_bf16.json` → `51_pairs_topup.json`, `51_scaling2_topup.jsonl` | 20 pairs, same recipe seeded with the original 10; blend/vote/schedule × λ∈{0,0.4}; balances n=2 to 30 groups |
| 52 | Composition showcase figures (n-sweep) | active | `33/39/40/51` JSONLs → `52_*.png`, `52_composition_summary.csv` | GPU-free; per-metric presentation figures; companion doc `plans/showcase-composition-explainer.md` |
| 53 | Long-generation composition | active | `long-generation-inputs.yaml` → `53_long_composition*` | 200 steps; blend/vote/schedule × λ∈{0,0.4}; safe resume and benchmark-only path; full run user-triggered |
| 54 | Woman/man differential steering | active | `long-generation-inputs.yaml` → `54_long_negative*` | 24 steps; all YAML prompts; attract-only/frame-diff/score-neg in both directions; full run user-triggered |
| 55 | Signed gender-composition samples | active | → `55_signed_composition*` | Atomic signed guides vs differential preblends; blend/vote/schedule × λ∈{0,0.4}; 3 prompts; full metrics + manual-review gallery |
| 56 | Subspace: pooled vs hierarchical | active | `32_…bf16.json`, `33_scaling_groups_bf16.json` → `56_subspace_singles*`, `56_subspace_blend*` | `subspace/` module, English lemmas; pooled (token-vote) vs hierarchical (lemma-vote) builds; A. single-concept steering + B. blend-pair composition (`ConceptSubspace.blend`) at λ∈{0,0.4} with the FRH joint/both-improved metric; RANK=8, R_LEMMA=1; `SMOKE` flag gates the full run |
| 57 | Subspace: pooled vs hierarchical (multilingual) | active | `32_…bf16.json`, `33_scaling_groups_bf16.json` → `57_subspace_ml_singles*`, `57_subspace_ml_blend*` | Multilingual twin of 56: same format, 8-language FRH set (`SupportedLanguages.Llama`) at `MIN_LEMMAS=11`; `SMOKE` flag gates the full run |
| 58 | Subspace: English vs multilingual comparison | active | `56_subspace_*`, `57_subspace_ml_*` → `58_lang_comparison*`, `58_per_concept_lang_delta.csv` | GPU-free; side-by-side 56-vs-57 tables + ML−En deltas + per-concept singles gains + figure; finding: multilingual lifts pooled, leaves hierarchical language-invariant |
| 59 | Subspace vs frame steering | active | `44_singles_lambda_bf16.jsonl`, `33_scaling3_bf16.jsonl`, `40_greedy_lambda.jsonl`, `56/57_subspace_*` → `59_frame_vs_subspace*` | GPU-free; frame pipeline (NB 44/33/40) vs subspace pooled/hier on scale-invariant metrics (positive-Δ / both-improved / ppl) at λ∈{0,0.4}; subspace steers harder on-target, frame is more fluent |
| 60 | Subspace qualitative gallery | active | `32_…bf16.json` → `60_subspace_gallery*` | Subspace counterpart of NB 47: novel/traveler quads, n=1→4 blend progression + λ=0/0.4 contrast, pooled vs hierarchical text side by side; `k=4`, `STEPS=32`, multilingual |
| 61 | Subspace λ scan (blend pairs) | active | `33_scaling_groups_bf16.json`, `57_subspace_ml.jsonl` → `61_subspace_lambda_scan*` | Subspace counterpart of NB 41: λ∈{0,0.3,0.4,0.5,0.6,0.8,1.0} on 10 pairs, pooled vs hierarchical; reuses NB57 λ=0/0.4; knee = max joint Δ s.t. ppl≤2.5; `SMOKE` gates the full sweep |
| 64 | Custom-concept battery: multi-FRH vs subspace | active | `64_custom_concepts.json` → `64_battery*.jsonl`, `64_battery_summary*.csv`, `64_battery*.png` | Concepts from hand-provided word lists (Borda `core` field) via synthetic `{synset, tokens}` slices; battery of 4 singles + 4 repels + 4 blends (joy/sorrow/safe/harm); multi-FRH (score-space voting) vs subspace at λ∈{0,0.4}, `k=4`/`steps=24`; directed-effect sum per 1-concept vs 2-concept case; `SMOKE` gates the full run |
| 65 | Concept source: custom core vs 8-language WordNet | active | `64_custom_concepts.json`, WordNet synsets → `65_source_battery*.jsonl`, `65_source_summary*.csv`, `65_source_battery*.png` | Same battery built two ways — curated English `core` words vs FRH's 8-language WordNet synsets (`joy.n.01`/`sorrow.n.01`/`safe.a.01`/`injury.n.01`, `add_exomw`); compares concept *source* per method at λ∈{0,0.4}; neutral axes = core-word success + ppl; `SMOKE` gates the full run |
| 66 | multi-FRH vs subspace at best λ (custom + WordNet) | active | `65_source_battery.jsonl` → `66_mfrh_vs_subspace*` | Method-primary re-analysis of NB65 across both concept sources at λ∈{0, best}; best λ = 0.4 for both (mfrh: λ improves both axes; subspace: fluency knee per NB61); GPU-free |
| 68 | Matched FRH blending vs subspace, 1–4 units | active | Four `concepts/*.json` + eight-language WordNet → `68_comparison{,_smoke}*` | Consensus top 100; 16 concepts, 60 named experiments/74 directed configurations, both directions of eight counterfactual families; true FRH frame blending vs pooled rank-8 subspace with signed repulsion; λ∈{0,0.4}, 24 steps, bf16 Llama-3.1-8B; four shared cosine-reference views and absolute continuation ΔPPL; immutable resume identity; default 192-output smoke, full 4,736-output battery |
| 69 | Lemma-count-matched custom vs WordNet, FRH vs subspace | active | NB68 manifest/runs + `concepts/*.json` → `69_topc{,_smoke}*` | Adds `custom_top_c` (first c ranked custom words, c = WordNet member lemmas across eight languages) to NB68's WordNet/custom sources; only the new source is generated (2,368 runs in NB68's eight-prompt batches, after an exact 128-run reproduction check), NB68 runs rescored on the new views; six shared cosine views, ΔPPL; p07 excluded at analysis; qualitative gallery `69_topc_gallery.md` flags degenerate (early-stop/repetitive) outputs |

## Playground (`playground/stepNN_*.ipynb`)

Per-implementation-step demo notebooks (one per step of the multi-concept
build-out): step02 unified loop · step02b beam search · step03 concept pairs +
ρ · step04 score aggregators · step05 frame composition · step06 schedulers ·
step07 eval harness · step10 Riemannian mean · step11 fluency blend ·
step12 angle metric · step13 GDS.

## Subspace development notebooks

| Notebook | Status | Notes |
|----|----|----|
| `subspace/notebooks/positional_subspace_pilot.ipynb` | active | Opt-in positional subspaces with FRH sliding windows; English dog/war singles, blend and signed repulsion; bf16 Llama-3.1-8B, pooled rank-8/rank-24 and FRH controls, reversed positions, raw/normalized scores; `SMOKE=True` selects one prompt/single; new timestamped outputs under `resources/positional_subspace_pilot_unquantized/`; prepared without GPU execution |
