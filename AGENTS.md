# Codex project guidance

## Scope and orientation

- The active repository root is the directory containing `pyproject.toml`. Treat the
  nested `frh-multi/` Git checkout as out of scope unless the user explicitly asks to
  work in it.
- Start by checking `git status --short`. A large dirty worktree is normal here;
  preserve unrelated source, notebook, plan, and result changes.
- Read `CONTEXT.md` before naming domain concepts. For experiment work, also read
  `NOTEBOOKS.md` and the relevant notebook or plan. For geometry or steering work,
  read `docs/agents/math-reference.md` and `plans/multiconcept-steering-math.md`.
- `CLAUDE.md` remains useful extended onboarding for humans and other agents, but this
  file is the repository-level instruction source for Codex.
- Local issue/spec files live under `.scratch/<feature-slug>/`; follow
  `docs/agents/issue-tracker.md` when a task references them.

## Project purpose

This is research code for the Frame Representation Hypothesis (FRH) and its
multi-concept extension of Top-k Concept-Guided Decoding (CGD). It represents an
ordered multi-token word as a frame in the model's whitened unembedding space,
constructs WordNet concept frames, and reranks a language model's top-k token
proposals by concept alignment. The multi-concept program compares composition in
frame space, score space, and time, plus greedy and true-beam survivor rules.

Use the vocabulary in `CONTEXT.md`: guide, score, active concept, survivor rule,
greedy tier, beam tier, and fluency weight. In particular, do not call the whole CGD
loop a "top-k guide," and do not describe the paper's survivor rule as beam search.

## Sources and provenance

- Repository `resources/` contains generated experiment artifacts. Do not confuse it
  with the external reading corpus at `../resources/papers/`.
- Primary paper: `../resources/papers/frh.pdf`.
- Fukui reference corpus: `../resources/papers/fukui_k/`.
- `NOTEBOOKS.md` is the notebook registry. Notebook numbering is frozen because
  artifact filenames contain notebook numbers. Move retired notebooks to `archive/`;
  never renumber active notebooks or casually delete old `resources/` outputs.
- Source notebooks plus their logged artifacts are research provenance. Avoid broad
  notebook rewrites, output clearing, or re-execution unless the task requires it.
  New notebooks should reuse `frames.experiments.notebook_helpers` where applicable.
- Notebooks 17-23 are parameterized over `QUANTIZATION`: historical AWQ is the
  default, while `P=bf16` writes `_unquantized` artifacts. Keep that provenance in
  filenames, tables, and claims.

## Architecture and mathematical contracts

- `frames/linalg/`: frame correlation, principal-angle metrics, Stiefel geometry,
  Procrustes operations, and generalized difference subspace (GDS) utilities.
- `frames/representations/`: whitened unembeddings, concepts, score aggregators,
  schedulers, and the CGD generation loops.
- `frames/models/hf/`: Hugging Face model/tokenizer wrappers and quantization.
- `frames/nlp/`: Open Multilingual WordNet ingestion and cached tokenization.
- `frames/evaluation/`: presence, expression, angle, perplexity, and JSONL logging.
- `tests/`: CPU mathematical/unit tests plus the GPU golden regression.

Preserve these contracts unless the task explicitly changes the method:

- A frame tensor is shaped `(n, d, k)`: count, model dimension, ordered columns.
- FRH steering correlation is a gauge-sensitive Stiefel quantity. Principal-angle,
  canonical-correlation, and GDS analysis are gauge-invariant Grassmann quantities.
  Do not substitute one for the other without changing and justifying the objective.
- Unequal-rank concepts use trailing zero-padding; rank counts nonzero columns, so
  padding is rank-neutral.
- The original single-concept path is pinned by `tests/golden/single_concept.json`.
  Keep its default scorer, normalization, fluency weight, and survivor behavior
  unchanged unless the user requests a method change and accepts a golden update.
- Unnormalized linear score voting is equivalent to scoring a summed frame up to
  rank-dependent constants. Per-candidate-pool normalization is the nonlinearity that
  makes Family 2 distinct; check both the normalization axis and pool shape.
- The paper's loop is greedy with one-step lookahead: it selects one parent and keeps
  that parent's top-k children. Only `_generate_with_topk_beam_guide` is true beam
  search across parents.
- Selection may use concept score plus `lambda * cumulative_logprob`, but reported
  probes remain concept trajectories. Perplexity comparisons use continuations, not
  padded/chat-formatted full texts.

See `docs/agents/codebase-map.md` for the full execution and experiment map.

## Environment and validation

Dependencies are managed by `uv`; Python 3.11+ is required. Model runs additionally
need CUDA, cached or downloadable gated weights, and `HUGGING_FACE_LOGIN_TOKEN` in
`.env`. Never print or commit secrets, and do not start a multi-gigabyte model download
unless the task requires it.

Use the smallest relevant checks first. In a sandboxed Codex environment, keep uv's
cache in `/tmp` and disable the unrelated rerun-failures socket plugin:

```shell
UV_CACHE_DIR=/tmp/frh-uv-cache uv run pytest -q -p no:rerunfailures
```

The configured test selection excludes `@pytest.mark.gpu`. Run the golden test only
when CUDA and the cached Llama AWQ model are available:

```shell
UV_CACHE_DIR=/tmp/frh-uv-cache uv run pytest -m gpu tests/test_golden.py -p no:rerunfailures
```

Run hooks only on files in scope because several hooks autofix files and this worktree
often contains unrelated notebook changes:

```shell
UV_CACHE_DIR=/tmp/frh-uv-cache uv run pre-commit run --files <changed-files>
```

Run a notebook headlessly with:

```shell
make run_notebook N=<notebook>.ipynb
make run_notebook N=<notebook>.ipynb P=bf16
```

Notebook execution can be GPU-heavy and writes an executed copy plus artifacts. Do
not use it as a routine smoke test when unit tests cover the changed code.

## Editing and review guardrails

- Make focused edits and do not clean, restore, stage, or overwrite unrelated changes.
- Do not edit generated plots/JSONL/CSV by hand. Change the producer, then regenerate
  only when requested or needed to validate the result.
- Prefer adding tests for mathematical behavior: orthonormality, gauge invariance or
  sensitivity as intended, zero/rank preservation, normalization axes, and limiting
  cases such as `k=1`, `lambda=0`, identical frames, and orthogonal frames.
- Review scientific claims against the producing notebook and artifact, not against a
  summary alone. Quantization, concept pool, survivor tier, lambda, metric version,
  and model family are part of every result's identity.
- Treat harmful-generation experiments as dual-use research. Do not broaden them
  beyond the requested evaluation scope.
