# Notebook cleanup report — 2026-07-29

Record of every change made during the notebook reorganization. Scope was
notebooks only; `resources/` was untouched, no notebook was renumbered, and
nothing was deleted — every retired file was moved, not removed. As of this
report **nothing is committed**: tracked-file moves are staged (`git mv`),
everything else is untracked, pending review.

## Goal

The multi-concept sprint left 51 notebooks at the repo root: 8 near-identical
AWQ/bf16 twin pairs, ~10 superseded or parked notebooks, and 35 files not in
version control. The cleanup gathers the twins, archives the retired
notebooks, indexes everything, and extracts reusable boilerplate — while
preserving the lab-journal property that the code which produced each logged
result stays inspectable, byte-for-byte.

Two hard constraints shaped the design: notebooks only run with CWD = repo
root (`frames` is not an installed package and all `resources/` paths are
root-relative), so active notebooks stayed flat at the root; and `resources/`
artifact names embed the producing notebook's number, so numbering is frozen
forever.

## 1. Archived notebooks (moved, with run records rescued)

New `archive/` tree at the repo root. `archive/README.md` documents every move
with a supersession pointer and the old→new path map.

**`archive/superseded/`** — replaced by a later notebook:

| Notebook | Superseded by | Reason |
|---|---|---|
| `16_e0_pilot` + `_unquantized` twin | 17 | E0 v1's presence metric floored at ~0; v2's expression metric answered the same question. The negative result survives in `resources/16_*`. |
| `25_showcase_multimodel` | 43 | Legacy-pool showcases; per the 2026-07-20 decision, 43 is the sole cross-model anchor. |
| `26_showcase_gemma9b` | 43 | Same. |
| `27_steering_concept_pool` | 32 | AWQ pool replaced by the validated bf16 pool (1 of 27 legacy concepts survived validation). |
| `28_scaling_3concepts` | 33 | AWQ scaling baseline replaced by the bf16 baseline. |
| `29_showcase_llama8b_bf16` | 43 | Legacy pool; same decision as 25/26. |
| `31_single_concept_beam` | 44 + 45 | Singles 2×2 reconstituted on the validated pool; origin of the "27/27, ~4.5×" teaser numbers, citable from the archive. |

**`archive/branches/`** — parked, not refuted (PhD-proposal material):
`34_common_blend_pilot` (Branch A, common-subspace steering) and
`35_hierarchy_alignment_pilot` (Branch C, hypernym alignment).

**`archive/executed/`** — 24 executed papermill copies (with cell outputs)
of all archived notebooks, copied out of the gitignored `outputs/`. Rationale:
for the untracked-era notebooks these executed copies are the *only* record of
the actual runs, and `outputs/` offers them no protection. Includes
`21_e3_negative_steering.ipynb`, the pre-rename run record of notebook 22.

Kept at root despite being superseded-adjacent: 37/38 (the d_cos pool branch)
because the reports cite them as core §6.4/6.5 robustness controls, and 24
because `24_showcase.jsonl` feeds notebook 30.

## 2. Merged quantized/unquantized pairs (17–23)

Each of the 7 pairs existed as an AWQ original plus a bf16 `_unquantized`
twin. A cell-by-cell diff confirmed the twins differ only in: a banner cell,
the model-load config, and the `_unquantized` suffix on output paths (plus
black-formatting noise on 17, which predates the black-jupyter hook). That is
exactly the papermill-parameter shape, so each pair became **one notebook at
the original number** built from the twin (the black-formatted, newer copy):

- A markdown banner explaining the parameterization and pointing to the
  preserved originals.
- A cell tagged `parameters`: `QUANTIZATION = "awq"` (default preserves
  historical behavior; override with `-p QUANTIZATION bf16`).
- A `CONFIGS` mapping prepended to the model-load cell selecting
  `(MODEL_ID, DEVICE_MAP, TORCH_DTYPE)` — AWQ-INT4/`cuda:0`/fp16 vs
  `meta-llama/Llama-3.1-8B-Instruct`/`auto`/bf16 — and
  `SUFFIX = "" | "_unquantized"`.
- Every resource-path literal containing `_unquantized` rewritten to an
  f-string with `{SUFFIX}`, keeping the historical file-naming convention
  exactly.

Per-notebook specifics:

- **22**: the archived AWQ original still writes stale `21_e3_*` paths (it was
  numbered 21 when run; the artifacts were later renamed on disk). The merged
  notebook standardizes on `22_e3_*{SUFFIX}`, matching the on-disk artifacts
  and every report citation.
- **23**: its cross-notebook *input* (`resources/22_e3_summary.csv`) is also
  suffix-parameterized, since it must read the matching tier.
- **21**: one comment inherited from the twin claimed the hardcoded
  `E1_BEST_GREEDY` reference values came from the unquantized CSV; they are
  AWQ values (per the twin's own banner), so the comment now says
  `20_e1_summary.csv (AWQ reference)`.
- All 7 normalized to nbformat 4.5 (the twins mixed 4.4/4.5 and some cells
  lacked ids).

Originals: AWQ notebooks `git mv`ed, twins `mv`ed, all 14 verbatim into
`archive/premerge/`. No historical notebook was edited — the merged files are
new artifacts.

**Verification (structural, no GPU):** textually rendering each merged
notebook at `QUANTIZATION="bf16"` reproduces its twin's code cells **exactly**;
rendering at `"awq"` reproduces the original exactly, except the three intended
deviations above and black re-wrapping on 17. All 7 pass `black-jupyter` and
`nbqa-ruff` unchanged (so the Phase-5 commit hooks will not rewrite them),
validate under `nbformat`, and papermill `--prepare-only` confirms the
parameter injection.

## 3. New shared helper module

`frames/experiments/notebook_helpers.py` — the stable helpers that were
redefined inline across notebooks, extracted from the newest bf16-era copies
(32/40/43/45/46): `llama_chat` / `gemma_chat` / `chat_fn` (the chat formatter
defined in ~37 notebooks), `answer`, `read_jsonl`, and `pareto_front` (with
configurable cost/gain columns). Covered by `tests/test_notebook_helpers.py`
(8 CPU tests).

**Existing notebooks were deliberately not retrofitted** — their inline copies
are the record of what produced the logged results. The module is for future
notebooks.

**Deliberately not extracted:** `make_methods` and `rows_from`. Inspection
showed each notebook defines a different variant (different method grids,
different result columns), so a "shared" version would misrepresent the
per-notebook record.

**`frames/experiments/__init__.py`** changed from an eager to a lazy (PEP 562)
re-export of `MultilingualModelGenerator`. Reason: `figstep_cdg_comparison.py`
uses `itertools.batched` (Python 3.12+), which made the whole
`frames.experiments` package un-importable on the project's 3.11 venv — a
pre-existing break (it also means notebook 10 currently cannot run; the lazy
import doesn't fix figstep, it just stops the break from poisoning the
package). Notebook 10's `from frames.experiments import
MultilingualModelGenerator` still resolves on 3.12+.

## 4. Index and documentation

- **`NOTEBOOKS.md`** (new, repo root): one row per notebook, grouped by era
  (foundation 01–14 / E-series 15–23 / showcases & pool 24–33 / branches
  34–38 / greedy core 39–49 / playground), with status, resource
  inputs→outputs, and quirks (e.g. notebook 12 references the removed cascade
  API and is not re-runnable; notebook 10 needs Python ≥ 3.12).
- **`archive/README.md`** (new): per-file reason, superseded-by pointer,
  old→new path map. Historical `plans/` docs that cite old root paths were
  *not* rewritten — they are lab records; the README carries the mapping.
- **`makefile`**: `run_notebook` gained optional `P=` (injects
  `-p QUANTIZATION <val>`; `P=bf16` names the executed copy `_unquantized`)
  and `OUT=` (explicit executed-copy name). Behavior without `P`/`OUT` is
  unchanged.
- **`CLAUDE.md`**: "Running notebooks" documents the `P=` flag and points to
  `NOTEBOOKS.md`/`archive/`; "Notebook structure" documents the frozen
  numbering rule.
- **`README.md`**: one sentence in Quick Start pointing to `NOTEBOOKS.md` and
  `archive/`.

## 5. Verification summary

1. **Resource-reference check** (before vs after): every `resources/…` string
   in every root notebook extracted and existence-checked. Zero newly-missing
   files; `{SUFFIX}` paths verified for both expansions. The 7 remaining
   "missing" entries are pre-existing (f-string regex false positives,
   paper-era pickles never on disk, and 43's write-target PNG).
2. **Merged-pair structural equivalence**: see §2.
3. **CPU test suite**: 136 passed (including the 8 new helper tests).
4. **Papermill smoke test**: notebook 49 (GPU-free) executed end-to-end
   through the updated makefile.
5. `make -n` dry runs confirm `P=` injection and the no-`P` path is
   byte-compatible with the old behavior.

## 6. Explicitly unchanged

- `resources/` — no file moved, renamed, or edited.
- All active notebooks' code (the only root-notebook content changes are the
  7 merged files, which are new artifacts).
- Pre-existing uncommitted modifications (01/03/12/13/24, `frames/*` edits,
  `models.yaml`, `resources/24_*`, tests) — untouched, still pending from
  before the cleanup.
- Git history — no commits. The commit sequence (Phase 5) is specified in the
  cleanup plan: gitignore `references/` (301 MB), commit pre-existing
  modifications separately, then snapshot the untracked keepers and the
  reorganization in logical commits, using `--no-verify` where a commit
  contains historical notebooks so the black-jupyter hook cannot rewrite the
  record.

## Net effect

Repo root: 51 notebooks → 34. Archived: 12 sources + 14 pre-merge originals
+ 24 executed run records, all recoverable at their documented paths. New
infrastructure: parameterized E-series notebooks, notebook index, shared
helper module, makefile parameter passthrough.
