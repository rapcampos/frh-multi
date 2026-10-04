# Plan: Presentation showcase — blend vs. vote vs. schedule across n = 2, 3, 4 concepts

Status: awaiting Rodrigo's go (planned 2026-08-02; nothing implemented yet).
Axis corrected by Rodrigo: the sweep is over **concurrent concept count
n = 2, 3, 4** (at the loop's standard K=4), not top-k width. This
supersedes the earlier k-axis draft entirely.

## Headline

**All required data already exists on disk.** Every (method × n × λ) cell
of both comparisons is covered, with angle metrics, on matched
groups/prompts/model/settings. No new GPU runs needed. Notebook 51 becomes
a GPU-free plotting + summary notebook (like 36/49); the remaining work is
the plans/ document.

## 1. Coverage — (method × n × λ) → source

All rows: bf16 Llama-3.1-8B-Instruct, K=4, STEPS=24, MAX_TOKENS=3,
MIN_LEMMAS=11, same 10 QUESTIONS, angles=True. Groups:
`resources/33_scaling_groups_bf16.json` (10 pairs + 30 triples, drawn from
the 32 validated pool) + `resources/39_quads.json` (30 quads). λ=0 and
λ=0.4 arms share the same groups (verified in nbs 33/39/40).

| method | n=2 | n=3 | n=4 |
|---|---|---|---|
| blend / vote / schedule, λ=0 | `33_scaling3_bf16.jsonl` (100 rows each) | `33_scaling3_bf16.jsonl` (300 each) | `39_scaling4.jsonl` (300 each) |
| blend / vote / schedule, λ=0.4 | `40_greedy_lambda.jsonl` (100 each) | `40_greedy_lambda.jsonl` (300 each) | `40_greedy_lambda.jsonl` (300 each) |

Load-time filters: keep only method ∈ {blend, vote, schedule} (drop
`gds-*`, `soft-vote` from 33; `common-blend`, `schedule-scaled` from 39 —
the latter ran STEPS=32); `lam = config.get("lam", 0.0)`. Merged total:
4,200 records → 18 summary cells.

Not mixed in: `43_validated_pool.jsonl` bf16 rows (different pair set than
the λ=0.4 arm — optional robustness footnote only); 41/42 (blend-/beam-
only); 20_e1 (old pairs, no angles); 38 (d_cos pool); 44/45 (singles).

## 2. Notebook 51 — `51_composition_figures.ipynb` (GPU-free)

No model load, no OOM guard, no SMOKE flag needed. Runtime: seconds.

1. md — title, purpose, inputs → outputs.
2. imports — json, Path, matplotlib.pyplot, pandas.
3. config — METHODS, COLORS {blend: tab:blue, vote: tab:orange, schedule:
   tab:green}, input JSONL paths, outputs `resources/51_*.png` +
   `resources/51_composition_summary.csv`.
4. loader — clone of nb40 cell 4: per record, synsets from config a/b/c/d;
   joint = min expression delta; joint_cos = min token_cos delta;
   all_positive(+cos); ppl_ratio; lam; n. Gate assert: 4,200 rows;
   per-(λ,n) counts {(0,2):300, (0,3):900, (0,4):900, (0.4,2):300,
   (0.4,3):900, (0.4,4):900}.
5. summary — groupby(method, n, lam) → mean joint, mean joint_cos,
   all-positive rates, mean ppl_ratio, n_runs → CSV. Cross-check print vs
   `40_greedy_lambda_summary.csv` on overlapping cells (must match ~4 dp).
6. λ=0 figures (1–2).
7. λ=0.4 figures (3–5).
8. all-positive table — printed pivot (method × n, both λ) for the doc.

Upstream call paths (already executed; for the doc narrative): blend λ=0
via `generate_with_topk_guide(guide=Concept.average(cs))`; vote via
`generate_with_topk_multi_guide(..., normalize="zscore")`; schedule via
`scorer=RoundRobin()` (definitively RoundRobin, F3.a); λ=0.4 adds
`fluency_weight=0.4` with blend routed through the multi-guide wrapper
(canonical single-guide wrapper has no fluency dial; paths verified
identical at λ=0).

**Aggregation: raw mean joint delta, not joint_z** — explainable
("improvement of the weakest concept in the group"); z over a 3-file merge
would depend on pool composition; matches nb40/43 reporting. joint_z over
the combined log available as optional appendix figure. token_cos needs no
choice (bounded, scale-free).

## 3. Plot specs — five figures, lines vs n

xticks [2,3,4], xlabel "concurrent concepts n", one line per method,
marker "o", method colors, legend, dpi=150, figsize ≈ (5.5, 4.2):

| # | file | y | extras |
|---|---|---|---|
| 1 | `51_lam0_expression.png` | mean joint expression delta, λ=0 | axhline(0) |
| 2 | `51_lam0_token_cos.png` | mean joint token-cos gain, λ=0 | axhline(0) |
| 3 | `51_lam04_expression.png` | mean joint expression delta, λ=0.4 | axhline(0) |
| 4 | `51_lam04_token_cos.png` | mean joint token-cos gain, λ=0.4 | axhline(0) |
| 5 | `51_lam04_ppl.png` | mean perplexity ratio, λ=0.4 | axhline(2.5, ls=":"), axhline(1) |

Doc caveat (not plot): raw joint shrinks with n by construction (min over
more concepts) — compare methods at each n, not the slope across n.

## 4. Document — `plans/showcase-composition-explainer.md`

Editorial: "the paper's top-k loop" (never the g-word); unquantized
unmarked; λ recipe as 0.3–0.4 interval; no black-box claims; narrative
prose, no notebook numbers in body (appendix only).

1. What this is about — steering toward several ideas at once + how we
   check it worked.
2. How steering works — top-k loop in three sentences; concepts are
   directions built from WordNet entries.
3. Three ways to combine concepts — blend ("mix the paints, then paint"),
   score voting ("panel of judges"), schedule ("taking turns"); one
   sentence each on when each should win.
4. The fluency dial (λ) — concept score vs the model's own preference;
   figures show dial-off (0) and the recommended end (0.4).
5. Where the concepts come from — validated pool in words: tested alone
   first, strongest admitted one at a time, ≤6 per category, no
   near-duplicates (delta floor 0.3, lexname cap 6, ρ-dedup 0.75 —
   narrated, no jargon); groups sampled to cover the similarity spectrum.
6. How we measure success — expression; joint expression ("team as strong
   as its weakest player"); all-positive rate; token cosine (0–1 scale,
   comparable across ideas); perplexity ratio ("gibberish alarm", 2.5×).
   Simplified from plans/metrics-explained.md Part 1.
7. Reading the figures — one paragraph per figure; caveats: joint shrinks
   with n by construction; n=2 arm rests on 10 groups vs 30 for n=3,4.
8. Takeaways — 3 bullets, filled after figures regenerate.
9. Appendix: provenance — file paths allowed here only.

## 5. NOTEBOOKS.md row

`| 51 | Composition showcase figures (n-sweep) | active |
33_scaling3_bf16.jsonl, 39_scaling4.jsonl, 40_greedy_lambda.jsonl →
51_*.png, 51_composition_summary.csv | GPU-free; per-metric presentation
figures; companion doc plans/showcase-composition-explainer.md |`

## 6. Verification

1. `make run_notebook N=51_composition_figures.ipynb` (GPU-free; doubles
   as papermill smoke test, like nb49).
2. Gate assert: 4,200 rows, per-(λ,n) counts as §2.
3. Cross-check vs `40_greedy_lambda_summary.csv` on overlapping cells.
4. Five PNGs + CSV under resources/51_*; git status shows only new files.
5. pre-commit clean; everything uncommitted for review.

## 7. Open questions (defaults if no objection)

1. n=2 arm has 10 groups vs 30 for n=3,4 — matches the published λ figure;
   optional 21-min GPU top-up (1,200 records) if equal weight wanted.
   Default: accept as-is.
2. Error bars: repo figures draw plain lines; SEM whiskers optional.
   Default: plain lines.
3. Doc filename `plans/showcase-composition-explainer.md`. Default: yes.
