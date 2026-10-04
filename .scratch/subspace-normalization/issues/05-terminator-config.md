# 05 — The terminator/pad token configuration is inverted

Status: validated in prototype (not applied to project source)
Type: task

## What is wrong

`BaseHuggingFaceModel._fix_llama_model` (`frames/models/hf/base.py:173-177`)
overrides both special tokens for every `meta-llama` model:

```python
self._fix_token("<|eot_id|>", "pad")          # pad  -> 128009
self._fix_token("<|end_of_text|>", "eos")     # eos  -> 128001
```

For a Llama-3.1 **Instruct** model this is backwards. The model ends an assistant
turn with `<|eot_id|>` (128009) and effectively never emits `<|end_of_text|>`
(128001) — the tokenizer's own default `eos_token` is already `<|eot_id|>`, so
the override is what breaks it. Measured on 6,144 candidate observations:
`<|eot_id|>` appears 505 times (8.2%), `<|end_of_text|>` zero times.

Two failures follow.

**The stop condition is dead.** `_is_generation_complete` tests
`eos_token_id` (`frame.py:745`, `subspace/representation.py:302`), so it never
fires for instruct models and generation always runs the full step budget.

**Generated terminators are erased.** `decode` runs `_clean` (`llm.py:66-72`),
which does `text.replace(self.tokenizer.pad_token, "")` — and `pad_token` *is*
`<|eot_id|>`. Every decode strips any terminator the model produced. Combined
with the round trip of issue `04`, the step's token is deleted, the re-encode
shortens the row, and left padding restores the width. The token vanishes.

Subspace steering *prefers* the terminator — raw score 8.33 vs 7.89 mean, 8.14 vs
7.14 median — so it hits this constantly; frame steering rarely picks it and
escapes. That is the whole 0.526-vs-1.005 growth gap.

It also explains NB68's `p07` empty baseline: `model.generate` emits `<|eot_id|>`
immediately, `_clean` strips it, and the continuation is the empty string. Not a
bad prompt — the same bug.

## The fix, and what is deliberately *not* fixed

Take the terminator to be `<|eot_id|>`.

**`pad_token` is left alone.** It is load-bearing well beyond padding:
`pad_vector` (`unembedding.py:149`) is the representation at `pad_token_id` and
serves as FRH's "meaningless vector", the origin every whitened token
representation is centred on; `unembedding.py:102` zeroes that row. Repointing
`pad_token` would move every concept frame and every published number in the
repo, and would silently invalidate the WordNet parquet caches, which pad with
`pad_token_id` (`synsets.py:393`) but are keyed only on tokenizer class name and
`hash(self)` (`synsets.py:227`).

That is a research decision about what "meaningless" means, not a bug fix, and it
should be raised on its own with its own re-run. Keeping `pad_token` unchanged is
what lets the generation fix ship without disturbing the representation layer.

## Validation

Prototyped in `../approach_a.py` together with issue `04`, measured by
`../smoke_approach_a.py`. See issue `04` for results.

## Not yet applied

The prototype subclasses `SubspaceUnembeddingRepresentation` and overrides only
the loop. Nothing under `frames/` or `subspace/` has been changed. Applying this
to the shared model wrapper will change frame-path behaviour too — `p07` baselines
stop being empty, and any generation that silently ran past a turn end will now
stop — so prior cached results become non-reproducible and need re-running.
