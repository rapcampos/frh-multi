"""Approach A prototype — token-space generation loop with a corrected terminator.

Lives outside the project source on purpose: nothing in ``frames/`` or
``subspace/`` changes until this is validated.

Three defects are fixed together, because none of them is sufficient alone
(see ``issues/04-token-space-loop.md`` and ``issues/05-terminator-config.md``):

1. **Dead stop condition.** ``BaseHuggingFaceModel._fix_eos_token_in_llama_model``
   overrides ``eos_token`` to ``<|end_of_text|>`` (128001), which a Llama-3.1
   *Instruct* model never emits — it ends a turn with ``<|eot_id|>`` (128009).
   So ``_is_generation_complete`` never fires. Here the terminator is taken to
   be ``<|eot_id|>``. ``pad_token`` is deliberately left alone: it also defines
   the "meaningless vector" (``unembedding.py:149``), so changing it would move
   every concept frame in the repo.

2. **Silent token erasure.** ``_candidates`` rebuilds state each step with
   ``make_input(decode(tokens), padding=True)``, and ``decode`` runs ``_clean``,
   which strips ``pad_token`` — which *is* ``<|eot_id|>``. Any generated
   terminator is therefore deleted, the re-encode shortens the row, and left
   padding restores the width. The step's token vanishes. Here state stays in
   token space: ``(ids, attention_mask)`` are carried forward and decoded once at
   the end.

3. **Batch-wide completion.** ``_is_generation_complete`` requires *every* row to
   contain the terminator, so in an 8-prompt batch one finished row keeps
   generating (and keeps getting erased). Here finished rows are frozen
   individually and the loop stops when all are done.

The scoring and selection rules are untouched — same ``subspace_alignment``,
same cumulative-max survivor rule — so any measured difference is attributable
to the loop, not to the steering objective.
"""

from __future__ import annotations

import numpy as np
import torch

from subspace.generation import (
    select_best_candidate,
    select_best_candidate_fluent,
    subspace_alignment,
)

EOT = "<|eot_id|>"


class TokenSpaceMixin:
    """Overrides the generation loop to keep state in token space."""

    @property
    def terminator_id(self) -> int:
        """The token that actually ends an assistant turn for this model."""
        tid = self.model.tokenizer.convert_tokens_to_ids(EOT)
        if tid is None:
            raise ValueError(f"{EOT} is not in this tokenizer's vocabulary")
        return tid

    def _forward(self, ids: torch.Tensor, mask: torch.Tensor, k: int):
        """Top-k continuation of each row, without a decode/re-encode round trip."""
        outputs = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits, hidden_states = outputs["logits"], outputs["hidden_states"]
        top = logits.softmax(-1)[..., -1, :].topk(k)
        return top.indices, top.values.log(), hidden_states[-1]

    @torch.inference_mode()
    def generate_token_space(
        self,
        input_text,
        guide,
        k: int = 2,
        steps: int = 16,
        fluency_weight: float = 0.0,
        repel_guide=None,
        normalize_score: bool = False,
    ):
        """Token-space mirror of ``generate_with_topk_guide``.

        Returns ``(texts, probe, stats)``; ``stats`` records per-row generated
        token counts and completion, so truncation is measurable rather than
        inferred from the decoded string.
        """
        input_text = [input_text] if isinstance(input_text, str) else input_text
        n = len(input_text)
        device = self.model.device
        basis = guide.basis.to(device)
        repel_basis = repel_guide.basis.to(device) if repel_guide is not None else None
        term = self.terminator_id
        pad = self.model.tokenizer.pad_token_id

        enc = self.model.make_input(input_text, padding=True)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        # Real (non-pad) prompt length per row; the padded width differs from it.
        prompt_len = mask.sum(-1).repeat_interleave(k, 0)

        # Seed the k-way lookahead, mirroring the original first expansion.
        first, first_logprob, _ = self._forward(ids, mask, k)
        ids = torch.cat(
            [ids.repeat_interleave(k, 0), first.reshape(n * k, 1)], dim=-1
        )
        mask = torch.cat(
            [mask.repeat_interleave(k, 0), torch.ones(n * k, 1, dtype=mask.dtype, device=mask.device)],
            dim=-1,
        )
        cum_logprob = first_logprob.flatten()
        # A row is done once it has emitted the terminator.
        done = (ids[:, -1] == term)

        probe = torch.zeros(n, 1)
        for _ in range(steps):
            child, child_logprob, last_hs = self._forward(ids, mask, k)
            scores = subspace_alignment(
                last_hs, basis, repel_basis, normalize=normalize_score
            )
            if fluency_weight:
                col_idx, probe = select_best_candidate_fluent(
                    scores, cum_logprob, fluency_weight, n, k
                )
            else:
                col_idx, probe = select_best_candidate(scores, n, k)
            col_idx = col_idx.cpu()

            # Expand every row by its k children, then keep the winning block.
            expanded_ids = ids.repeat_interleave(k, 0)
            expanded_mask = mask.repeat_interleave(k, 0)
            expanded_done = done.repeat_interleave(k, 0)
            next_tok = child.reshape(-1, 1)

            # Finished rows are frozen: they append padding, not new content.
            next_tok = torch.where(
                expanded_done.unsqueeze(-1), torch.full_like(next_tok, pad), next_tok
            )
            new_ids = torch.cat([expanded_ids, next_tok], dim=-1)
            new_mask = torch.cat(
                [
                    expanded_mask,
                    (~expanded_done).to(expanded_mask.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            new_done = expanded_done | (next_tok.squeeze(-1) == term)

            def block(t):
                """Keep the winning parent's k children: (n*k*k, ...) -> (n*k, ...)."""
                return t.reshape(n, k, k, *t.shape[1:])[np.arange(n), col_idx].flatten(
                    0, 1
                )

            ids, mask, done = block(new_ids), block(new_mask), block(new_done)

            if fluency_weight:
                rows = torch.arange(n, device=cum_logprob.device) * k + col_idx.to(
                    cum_logprob.device
                )
                cum_logprob = (
                    cum_logprob[rows].unsqueeze(-1) + child_logprob[rows]
                ).flatten()

            if bool(done.all()):
                break

        kept_ids, kept_mask = ids[::k], mask[::k]
        kept_prompt_len = prompt_len[::k]
        stats = []
        for row_ids, row_mask, p_len in zip(kept_ids, kept_mask, kept_prompt_len):
            live = row_ids[row_mask.bool()]
            generated = live[int(p_len) :]
            stats.append(
                dict(
                    generated_tokens=int(generated.numel()),
                    terminated=bool((generated == term).any()),
                )
            )
        return self.model.decode(kept_ids), probe.cpu(), stats
