# Archived notebooks

Notebooks retired from the repo root during the 2026-07 cleanup. Nothing here was
deleted: sources were moved verbatim, and the executed run records (from the
gitignored `outputs/`) are preserved in `executed/`. Their result files in
`resources/` were **not** moved — active notebooks keep reading them. Historical
docs under `plans/` cite the old root paths; this file carries the mapping.

See `NOTEBOOKS.md` at the repo root for the full index of active notebooks.

## superseded/ — replaced by a later notebook

| Archived (old root path) | Superseded by | Why |
|---|---|---|
| `16_e0_pilot.ipynb` | `17_e0_pilot_v2.ipynb` | E0 v1: presence metric floored at ~0 despite visible steering (findings log, Step 9); v2's expression metric answered the same question. Its negative result (presence-metric falsification) survives in `resources/16_*` and the reports. |
| `16_e0_pilot_unquantized.ipynb` | `17_e0_pilot_v2.ipynb` | bf16 twin of the above. |
| `25_showcase_multimodel.ipynb` | `43_showcase_validated_pool.ipynb` | Legacy-pool cross-model showcase; per user decision (2026-07-20), 43 is the sole cross-model anchor. |
| `26_showcase_gemma9b.ipynb` | `43_showcase_validated_pool.ipynb` | Legacy-pool Gemma-2-9B showcase; same decision. |
| `27_steering_concept_pool.ipynb` | `32_concept_pool_bf16.ipynb` | AWQ concept pool replaced by the validated bf16 pool (only 1 of 27 legacy concepts survived validation). `resources/27_concept_pool.json` stays — still read by 28/32/43/47-era outputs. |
| `28_scaling_3concepts.ipynb` | `33_scaling3_bf16_gds.ipynb` | AWQ 3-concept scaling baseline replaced by the bf16 baseline. |
| `29_showcase_llama8b_bf16.ipynb` | `43_showcase_validated_pool.ipynb` | Legacy-pool bf16 showcase; same 2026-07-20 decision. `resources/29_llama8b_bf16.jsonl` stays (read by 30/37). |
| `31_single_concept_beam.ipynb` | `44_singles_lambda_bf16.ipynb` + `45_singles_beam_bf16.ipynb` | Singles greedy×beam×λ 2×2 reconstituted on the validated pool by 44/45. Origin of the "27/27, ~4.5×" future-work teaser numbers — citable from here. |

## branches/ — parked research branches (not refuted; PhD-proposal material)

| Archived | Branch | Status |
|---|---|---|
| `34_common_blend_pilot.ipynb` | Branch A: steering toward the COMMON subspace | Produced the common-subspace localization finding; deliberately held for Future Work. |
| `35_hierarchy_alignment_pilot.ipynb` | Branch C: siblings' common subspace vs hypernym | Held for Future Work alongside Branch A. |

## premerge/ — originals of the merged quantized/unquantized pairs

Notebooks 17–23 each existed as an AWQ original plus a bf16 `_unquantized` twin
differing only in the banner, the model-load config, and the `_unquantized`
output suffix. Each pair was merged into a single papermill-parameterized
notebook at the original root path (`QUANTIZATION = "awq" | "bf16"`); the
originals live here verbatim:

| Merged root notebook | Originals here |
|---|---|
| `17_e0_pilot_v2.ipynb` | `17_e0_pilot_v2.ipynb`, `17_e0_pilot_v2_unquantized.ipynb` |
| `18_e2_riemannian.ipynb` | `18_e2_riemannian.ipynb`, `18_e2_riemannian_unquantized.ipynb` |
| `19_e2b_rho_crossover.ipynb` | `19_e2b_rho_crossover.ipynb`, `19_e2b_rho_crossover_unquantized.ipynb` |
| `20_e1_family_sweep.ipynb` | `20_e1_family_sweep.ipynb`, `20_e1_family_sweep_unquantized.ipynb` |
| `21_e1b_fluency_pareto.ipynb` | `21_e1b_fluency_pareto.ipynb`, `21_e1b_fluency_pareto_unquantized.ipynb` |
| `22_e3_negative_steering.ipynb` | `22_e3_negative_steering.ipynb`, `22_e3_negative_steering_unquantized.ipynb` |
| `23_e3_beam_negative.ipynb` | `23_e3_beam_negative.ipynb`, `23_e3_beam_negative_unquantized.ipynb` |

Note: the archived AWQ `22_e3_negative_steering.ipynb` still writes stale
`resources/21_e3_*` paths (it was numbered 21 when run; the artifacts were later
renamed to `22_e3_*` on disk). The merged notebook standardizes on
`22_e3_*{SUFFIX}`, matching the on-disk artifacts and every report citation.

## executed/ — run records

Executed papermill copies (with cell outputs) of the archived notebooks, rescued
from the gitignored `outputs/` directory so the actual runs stay in history.
`outputs/21_e3_negative_steering.ipynb` (still in `outputs/`) is the executed
copy of what is now notebook 22 — the pre-rename run.
