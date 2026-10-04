# NB68: matched FRH blending and subspace comparison

`68_matched_frame_subspace.ipynb` implements the approved comparison. The reusable
runner is `frames.experiments.steering_comparison`; it also supports command-line
execution. No previous notebook or result is an input to the new generation log.

## Frozen battery

The four explicit JSON files in `concepts/` supply 16 matching concept/synset
entries. Consensus uses lowercase, trimmed words, first occurrence per source,
and normalized Borda contributions based on each original list's length. Top-100
ties break by shorter word and then alphabetical order. The manifest preserves
all candidate scores and source ranks, the selected words, and input hashes.

WordNet uses the original Llama eight-language loader with `min_lemmas=2` and a
four-token word limit. Both methods receive identical token rows within each
source. Custom words use the same four surface forms as NB65. Construction logs
record token rows, excluded words, effective ranks, and canonical synsets.
The JSON aliases `trust.n.05` and `suspicion.n.02` resolve in WordNet2021 to
`faith.n.02` and `misgiving.n.03`; both identifiers are preserved.

| Units | Named experiments | Directed configurations |
|---|---:|---:|
| 1 | 16 singles + 8 pairs = 24 | 32 |
| 2 | 10 positive + 2 pairs = 12 | 14 |
| 3 | 10 positive + 2 pairs = 12 | 14 |
| 4 | 10 positive + 2 pairs = 12 | 14 |

A pair is one steering unit and two underlying concepts. Every pair experiment
runs both directions with other concepts held fixed. Each count covers all 16
concepts. Exact combinations reside in the manifest module and saved manifest.
This is an illustrative fixed battery, not a balanced combinatorial design.

## Methods and measurements

- FRH builds equal-weight extrinsic blends. A differential frame is first
  `polar(F_A - F_B)` and then receives one equal weight in the outer blend.
  The lambda-capable generation wrapper receives exactly one precomposed guide
  with normalization disabled; this is not the NB65 score-voting configuration.
- Subspace uses pooled rank-8 concepts and fixed-rank positive blends. A
  counterfactual subtracts the negative concept's projection score from the
  positive blend's score. Negating basis vectors would not implement repulsion.
- Both use Llama-3.1-8B-Instruct bf16, greedy survivor rules, top-k 4, 24 steps,
  seed 0, and fluency weights 0 and 0.4. Native score scales are preserved:
  equal lambda values need not produce equal fluency pressure.
- The eight NB65 prompts are shared. The loop emits one initial token plus up
  to 24 expansions; the baseline cap is consequently 25, correcting NB65's
  unmatched 26-token cap. Measurements exclude assistant prefixes and special
  tokens. Perplexity is continuation-only and unconditional on the prompt,
  following the existing harness.
- Every output uses the same four fixed cosine views: WordNet/custom crossed
  with frame/subspace spans. Compare methods and sources within a view; do not
  average the four different reference geometries into one score.
- Log every constituent's raw cosine and baseline delta; summarize signed mean,
  signed minimum, all-components-improved rate, attraction, and repulsion.
  Fluency is absolute delta PPL, with raw PPL and ratio retained. Empty or
  unscorable outputs have explicit invalid flags and null metrics.

Reports separate positive and counterfactual cases. They first average complete
pair directions within each prompt, then prompts within experiments, and finally
give experiments equal weight. Missing/invalid directions do not become zeros;
metric-specific denominators accompany macro means. Paired tables compare
FRH-minus-subspace for the exact same source, configuration, lambda, prompt, and
reference. Report status distinguishes partial from complete logs.

## Run and resume

Inspect the full matrix without GPU/model loading:

```shell
UV_CACHE_DIR=/tmp/frh-uv-cache uv run python -m frames.experiments.steering_comparison --dry-run
```

Write the consensus words, provenance and frozen experiment manifest only:

```shell
UV_CACHE_DIR=/tmp/frh-uv-cache uv run python -m frames.experiments.steering_comparison --prepare-only
```

Run the 192-continuation smoke battery (two prompts; positive and both signed
directions at all four counts), then the 4,736-continuation full battery:

```shell
UV_CACHE_DIR=/tmp/frh-uv-cache uv run python -m frames.experiments.steering_comparison --smoke
UV_CACHE_DIR=/tmp/frh-uv-cache uv run python -m frames.experiments.steering_comparison
```

The notebook defaults to `SMOKE=True`; change it to `False` for the full run.
Set `RUN_GENERATION=False` for manifest inspection and GPU-free reporting of
existing outputs. The runner requires CUDA and the usual repository model setup.

Outputs use `resources/68_comparison{,_smoke}`: manifest, construction provenance,
baselines, generation JSONL, per-concept/run/summary/paired CSVs, cosine–PPL plot,
gallery, and report status. Repeating a command resumes the same frozen matrix.
An interrupted partial prompt batch is regenerated as a whole, with completed
records retained. Changed source files, recipe, implementation, model revision,
or construction refuse resume; use a new `--output-prefix`. Baseline contents are
hashed and never regenerated when resuming a valid existing log.

CPU tests cover consensus/matrix invariants, mathematical scoring and composition,
mocked generation wiring, actual metric-to-report integration, invalid outputs,
paired aggregation, and resume/provenance checks. GPU smoke results are required
before drawing empirical method comparisons.
