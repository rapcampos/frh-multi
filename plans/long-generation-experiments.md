# Plan: Long-Generation Composition and Negative Steering

Status: implemented; the notebooks are ready but the GPU experiments have not
been executed.
The editable source of prompts and concepts is
`long-generation-inputs.yaml` in the repository root. It separates complete,
reusable catalogs from per-experiment ID selections. Both notebooks will load
that file and reject duplicate or unknown IDs before loading the model.

## Shared design

- Model: bf16 `meta-llama/Llama-3.1-8B-Instruct`.
- Survivor rule: the paper's greedy top-k rule; `k = 4`.
- Length and fluency settings are experiment-specific: notebook 53 uses
  `STEPS = 200` at both lambda = 0 and lambda = 0.4; notebook 54 keeps
  `STEPS = 24`.
- Concepts: `min_lemmas = 11`, `max_tokens = 3`, using the current Llama
  concept cache and validated-pool conventions.
- Prompts: each experiment resolves its own prompt IDs from the shared catalog,
  then formats the selected entries with
  `frames.experiments.notebook_helpers.llama_chat`.
- Baselines: one deterministic unsteered continuation per selected prompt,
  generated once and reused by both experiments when settings match.
- Logging: append-safe JSONL with the complete configuration, prompt ID,
  concept/direction ID, generated text, continuation length, termination
  reason, metrics, and timing. Do not delete a prior full log implicitly;
  require an explicit `OVERWRITE = True` or resume missing cells.
- Reproducibility: fixed ordering, no sampling, a small `SMOKE = True` path,
  and a printed run matrix/token budget before model loading. Batch size should
  be explicit and conservative (start at 1) because the loop retains hidden
  states for k candidate rows over a much longer context.

Implementation audit before the full run: the current guided loop seeds one
top-k candidate before its `steps` iterations. We should measure the decoded
continuation length in a smoke run and decide whether `STEPS = 200` means the
project's historical convention or exactly 200 appended tokens. Early EOS can
also shorten a continuation, so actual token counts and termination must be
logged rather than inferred from the setting.

## Notebook 53: long-text composition

Proposed name: `53_long_generation_composition.ipynb`.

### Question

Do the current family ranking and joint-concept gains persist over a roughly
200-token continuation, or do methods drift, collapse, alternate, or become
repetitive at longer horizons?

### Conditions

For every selected concept pair and prompt, run exactly these three methods:

1. `blend`: extrinsic `Concept.average([a, b])` as one guide.
2. `vote`: separate guides with per-candidate-pool z-score normalization.
3. `schedule`: separate guides with a reset `RoundRobin()` scorer.

All three use the same greedy survivor rule, k, length, prompts, and baselines.
Each cell runs twice, once with pure steering (lambda = 0) and once with the
fluency weight enabled (lambda = 0.4). This isolates the composition ingredient
and measures whether method behavior at long horizons depends on the fluency
term.

### Measurements

- Existing whole-continuation metrics: expression delta, token-cosine delta,
  joint minimum, both-positive indicator, and continuation PPL ratio.
- Longitudinal metrics in fixed 20-token bins: per-concept expression and
  token-cosine deltas, their joint minimum, and the active concept for the
  scheduler. The key output is a method-by-position trajectory rather than one
  mean over 200 tokens. Also retain absolute steered and baseline values: if
  either continuation ends early, a baseline-subtracted bin is defined only
  where both have tokens, and coverage is reported rather than silently
  padding the missing text.
- Degeneration checks: distinct-token ratio, repeated 4-gram rate, maximum
  repeated-span length, EOS/length, and PPL by segment.
- Qualitative table: full continuation plus a compact 20-token-bin score strip
  for every selected cell.

### Outputs

- `resources/53_long_composition.jsonl`: one record per
  pair x prompt x method x lambda setting.
- `resources/53_long_composition_summary.csv`: aggregate and final-bin values.
- `resources/53_long_composition_trajectories.csv`: tidy per-bin measurements.
- Figures comparing joint expression, concept balance, PPL, and repetition
  over token position.

## Notebook 54: differential steering

Proposed name: `54_long_generation_negative.ipynb`.

### Question

For the FRH paper's gender differential, does repulsion remain selective at the
established 24-step horizon, or does it suppress both concepts or behave
asymmetrically by direction?

### Directions and controls

Run both requested directions from the shared YAML:

- attract `woman.n.01`, repel `man.n.01`;
- attract `man.n.01`, repel `woman.n.01`.

For each direction and selected prompt, compare:

1. `unsteered`: the shared baseline.
2. `attract-only`: the positive concept alone; controls for correlated lift.
3. `frame-diff`: the paper's projected frame difference A - B; the primary
   negative-steering condition.
4. `score-neg` (recommended control): separate concept scores with weights
   `[+1, -1]` and per-step z-score normalization. This distinguishes the
   paper's frame-space operation from score-space subtraction.

This notebook keeps the existing negative-steering horizon, `STEPS = 24`, and
uses `prompt_ids: all`, currently all ten canonical prompts. Its shorter horizon
keeps the negative-steering question separate from the long-generation intervention.
The fluency setting is lambda = 0.4, resolved from the YAML experiment entry.

### Measurements

- Per direction and 20-token bin: attraction delta, repulsion delta, net
  contrast `delta_attract - delta_repel`, and strict clean success
  `(delta_attract > 0 and delta_repel < 0)`.
- Whole-continuation angle versions of the same quantities, PPL ratio, EOS,
  and the degeneration checks used in notebook 53.
- Direction-symmetry analysis: compare woman-minus-man with man-minus-woman
  rather than averaging them; asymmetry is itself a result.
- Qualitative review flags for gender stereotypes. The notebook should report
  behavior descriptively and avoid treating either direction as a normative
  target.

### Outputs

- `resources/54_long_negative.jsonl`.
- `resources/54_long_negative_summary.csv`.
- `resources/54_long_negative_trajectories.csv`.
- Trajectory figures for attraction, repulsion, net contrast, clean rate, PPL,
  and repetition.

## Runtime control and recommended first selection

With P concept pairs and Q prompts, notebook 53 produces
`3 methods x 2 lambda settings x P x Q` guided continuations, or approximately
`1,200 x P x Q` guided output tokens at 200 steps. Keeping all 30 pairs and 10
prompts would mean 1,800 continuations and about 360,000 guided output tokens:
roughly 16.7 times the token work of the current 24-step showcase.

That 16.7x figure counts outputs, **not wall time**. The current implementation
decodes and reprocesses the complete candidate sequence on every step with
`use_cache=False`; sequence work therefore grows superlinearly with the output
length. Before the full run, benchmark one fixed cell at 24, 50, and 100 steps,
record seconds per completed step and peak GPU memory, then extrapolate to 200.
If the curve is prohibitive, reduce the selected matrix first; changing the
generation loop to a cached implementation would be a separate method and
regression task.

For example, keeping **three prompts and three pairs** gives 54 guided
continuations (about 10,800 output tokens), small enough to inspect every text
while covering method x lambda x prompt x pair interactions. A useful pair
subset spans the current rho range:

- `passing_frisson` (rho 0.221, low);
- `education_sorrow` (rho 0.324, middle and interpretable);
- `universe_initiation` (rho 0.518, high).

For prompts, open-ended continuations such as `short_story`,
`favorite_place`, and `good_life`. List-style prompts may terminate early or
become repetitive over 200 tokens. The final selection is intentionally left to
the user in the root YAML file. Notebook 54 independently resolves
`prompt_ids: all`, so filtering notebook 53 does not reduce its coverage.

Current YAML selection: notebook 53 has 15 pairs x 5 prompts x 3 methods x 2
lambda settings = **450 guided continuations**, approximately 90,000 output
tokens at the 200-step cap. Notebook 54 has all 10 prompts x 2 directions x 3
guided conditions = **60 guided continuations**, approximately 1,440 output
tokens at the 24-step cap. These counts should be printed and confirmed in each
notebook before model loading.

## Implementation controls

- Both notebooks default to `RUN_GENERATION = False`, print the resolved matrix
  before model loading, isolate smoke artifacts, and require an exact
  confirmation string for a full run.
- Existing JSONL rows are resumed by the full experimental cell key. A log with
  rows outside the currently resolved YAML matrix is rejected rather than mixed
  with a changed selection.
- Notebook 53 supports `RUN_BENCHMARK = True` with
  `BENCHMARK_ONLY = True` for the 24/50/100-step timing preflight without
  starting the 450-cell matrix.
- Longitudinal hidden-state and segment-PPL analysis is opt-in through
  `RUN_TRAJECTORIES = True`; it produces the trajectory CSV/figure and adds
  final-bin measurements to the summary.
- Full GPU runs remain user-triggered. CPU configuration, notebook structure,
  and helper tests are the implementation validation path.
