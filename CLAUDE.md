# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the research codebase for the **Frame Representation Hypothesis** (TACL/EMNLP 2025), a framework for understanding and controlling LLMs using WordNet-derived concept frames. The core idea: represent multi-token WordNet concepts as orthogonal frames (elements of Stiefel manifolds) in the LLM's unembedding space, then use those frames to probe or guide text generation.

## Environment setup

Dependencies are managed with `uv`. Python ≥ 3.11 required.

```shell
uv sync
```

A `.env` file is required with `HUGGING_FACE_LOGIN_TOKEN` to download gated models. Models are listed in `models.yaml` — only un-commented entries are active.

## Running notebooks

Run a single notebook headlessly via papermill:

```shell
make run_notebook N=02_tokenization_frames.ipynb
```

Output goes to `<notebook_dir>/outputs/<notebook_name>`. The notebooks are numbered and correspond to paper experiments starting from `02`.

`01_START_HERE.ipynb` downloads all models configured in `models.yaml`.

`plotting.ipynb` regenerates figures from cached results in `resources/`.

## Docker

```shell
docker compose up
```

Mounts the repo at `/app` and the HuggingFace cache at `~/.cache/huggingface` (override with `HF_CACHE_DIR`). Requires NVIDIA GPU.

## HPC cluster (NEC NQSV)

```shell
make qlogin_debug   # interactive debug node (1h, OpenMPI)
make qsub_gpu       # batch submit qsub.sh
make budget         # check remaining compute budget
```

## Linting and pre-commit

```shell
uv run pre-commit run --all-files
```

Hooks run: `black` + `black-jupyter` (formatting), `ruff` (linting with autofix, selects E/F/I), `nbqa-ruff` (same for notebooks), `vulture` (dead code, ≥80% confidence), and standard YAML/AST/TOML checks.

## Architecture

### Foundation

All domain objects inherit from `frames/abstract/base_model.py::BaseModel`, which is a Pydantic `BaseModel` with `arbitrary_types_allowed=True`. This means fields are validated at construction time and torch tensors can be stored as typed fields.

### Core abstraction layers

**`frames/linalg/`** — Math primitives.

- `Frame` (`frame.py`): central data structure — a `torch.Tensor` of shape `(n, d, k)` (n frames, d embedding dimension, k vectors per frame). The `*` operator between two `Frame` objects computes the correlation: `trace(F1ᵀ F2) / sqrt(rank(F1) * rank(F2))`. Frame subtraction (`F1 - F2`) returns the Procrustes projection of the difference.
- `orthogonalization.py`: `gram_schmidt` and `solve_procrustes` (SVD-based orthogonal projection onto the Stiefel manifold).
- `matrix.py`: `symsqrtinv` — inverse square root of a symmetric PSD matrix via eigendecomposition, used for whitening.

**`frames/representations/`** — Builds on linalg.

- `LinearUnembeddingRepresentation` (`unembedding.py`): wraps a HuggingFace model. Computes whitened token representations: `W_tilde = W @ Σ^{-1/2}` where `W` is the unembedding matrix and `Σ` is its row covariance. Subtracts a "meaningless vector" (mean or pad token embedding) to center representations.
- `FrameUnembeddingRepresentation` (`frame.py`): extends the above to build `Concept` frames from WordNet synsets, project text onto concepts, and run concept-guided generation (`generate_with_topk_guide`). The key guided generation loop: at each step, generates `k` candidate next tokens, scores each by cumulative concept projection, and keeps the best beam.
- `Concept` (`concept.py`): a `Frame` subclass that also carries a `synset` (DataFrame of synset names), enabling `concept_A - concept_B` for differential guidance (e.g., steer toward "woman" and away from "man").

**`frames/nlp/`** — WordNet interface.

- `MultiLingualWordNetSynsets` (`synsets.py`): tokenizes all WordNet lemmas using a given tokenizer, builds a cuDF DataFrame (GPU-accelerated), caches to `~/.cache/wordnet/` keyed by tokenizer class + `hash(self)`. Generates lemma surface-form variations with leading/trailing spaces. `get_all_synsets(tokenizer, min_lemmas_per_synset, max_token_count)` is the main entry point.
- `SupportedLanguages` enum: `Default` = English only; `Llama` = 8 languages. `from_model_id(id)` maps by substring match on model ID (e.g., `"llama"` in the ID → Llama languages).
- `datasets.py`: loads SafeBench (`frames/data/safebench.csv`) and multilingual SafeBench (`frames/data/multilang-safebench.parquet`) for jailbreak evaluation.

**`frames/models/hf/`** — HuggingFace model wrappers.

- `BaseHuggingFaceModel` (`base.py`): handles loading with quantization options — `quantization=4` (4-bit NF4), `quantization=8` (8-bit), `quantization="AWQ"`, or `"auto"` (detects from model ID). Uses `device_map="auto"` by default.
- `LanguageHuggingFaceModel` (`llm.py`): adds tokenization helpers, `forward_last_hidden_state`, integrated gradients, and `use_chat_template=True` for instruction-tuned models. Mistral models get a pad-token fix automatically. `unembedding_matrix` returns `lm_head.weight`.

**`frames/utils/`** — Shared utilities.

- `settings.py`: `load_models()` reads `models.yaml` into a DataFrame.
- `memory.py`: `gc_cuda()` context manager for CUDA memory cleanup between experiments.
- `tensor.py`, `ml.py`, `stdlib.py`: tensor helpers, ML utilities, and stdlib extensions.
- `plotting.py`: shared plotting configuration used across notebooks.

**`frames/data/`** — Bundled datasets and evaluation data (SafeBench CSV/parquet, FigStep images).

**`frames/experiments/`** — Reusable experiment logic (e.g., `figstep_cdg_comparison.py`).

### Data flow

1. Load model via `LanguageHuggingFaceModel(id=...)`.
2. Build `FrameUnembeddingRepresentation(model=...)` — or use `FrameUnembeddingRepresentation.from_model_id(id)` which also sets the correct language set for the model family.
3. Call `get_all_concepts(tokenizer, min_lemmas_per_synset, max_token_count)` → `Concept` with all WordNet synsets as frames (cached in-process via `methodtools.lru_cache`; tokenized data cached to disk on first call).
4. Use `project(text, concept)` to measure concept alignment per token position, or `generate_with_topk_guide(text, guide=concept, k=...)` for guided generation. For differential guidance: `guide = concept_A - concept_B`.

### Notebook structure

Each numbered notebook corresponds to a paper experiment. Results (plots, pickled data) are saved to `resources/`. The `cache/` directory stores HuggingFace datasets and tokenized representations used across notebooks. `results.shelf.db` is a `shelve` database used by some notebooks to persist intermediate results across runs.

### Key dependencies

- `cudf`/`cuml` (RAPIDS): GPU-accelerated DataFrames for WordNet tokenization — requires CUDA.
- `autoawq`: AWQ quantization for AWQ model variants.
- `nltk` with `wordnet2021` + OMW (`add_exomw()`): WordNet corpus.
- `papermill`: headless notebook execution.
- `methodtools`: provides `lru_cache` that works on instance methods (unlike `functools.lru_cache`).
