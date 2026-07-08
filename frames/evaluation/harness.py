"""Evaluation harness for multi-concept guided generation (Phase C).

One JSONL record per generation: prompt, full text, continuation, config,
per-concept success, perplexity vs an unsteered baseline (fluency guardrail),
and optional per-step score traces captured by `RecordingScorer`.

Success metric v1: concept-word presence — any WordNet member lemma of the
target synset appears in the generated continuation (case-insensitive
substring). Upgrade to a classifier only if presence proves too crude.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


class RecordingScorer:
    """Wrap any scorer/scheduler and record per-step score summaries.

    For each step, stores the newest-token per-concept scores and the
    aggregated score of each input's representative row (row i*m is the
    greedy child of the beam selected at the previous step). Values are what
    the aggregator sees — i.e. AFTER the loop's optional normalization.
    """

    def __init__(self, scorer: Callable):
        self.scorer = scorer
        self.steps: list[dict] = []
        self.n = 1

    def reset(self, n: int) -> None:
        self.steps = []
        self.n = n
        if hasattr(self.scorer, "reset"):
            self.scorer.reset(n)

    def observe(self, tokens: torch.Tensor, n: int) -> None:
        if hasattr(self.scorer, "observe"):
            self.scorer.observe(tokens, n)

    def __call__(
        self, projections: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        scores = self.scorer(projections, weights)
        m = projections.size(0) // self.n
        rows = torch.arange(self.n) * m
        self.steps.append(
            {
                "per_concept": projections[rows, -1, 0, :].float().cpu().tolist(),
                "aggregated": scores[rows, -1, 0].float().cpu().tolist(),
            }
        )
        return scores

    def per_input_steps(self, i: int) -> list[dict]:
        """Trace for input i: one entry per step."""
        return [
            {"per_concept": s["per_concept"][i], "aggregated": s["aggregated"][i]}
            for s in self.steps
        ]


class EvaluationHarness:
    """Metrics and JSONL logging for steered-generation outputs.

    Decoupled from generation: pass any (prompts, texts) produced by any
    method/family, plus a config dict describing how they were made.
    """

    def __init__(
        self,
        fur,
        log_path: str | Path,
        min_lemmas_per_synset: int,
        max_token_count: int,
        ppl_ratio_threshold: float = 2.5,
    ):
        self.fur = fur
        self.log_path = Path(log_path)
        self.min_lemmas_per_synset = min_lemmas_per_synset
        self.max_token_count = max_token_count
        self.ppl_ratio_threshold = ppl_ratio_threshold

    def member_lemmas(self, synset: str) -> set[str]:
        """All WordNet member surface forms of a synset (lowercased).

        Uses `get_dataframe` (per-lemma rows), not `get_all_synsets` — the
        latter groups by synset and keeps only the padded token arrays.
        """
        df = self.fur.data.get_dataframe(
            self.fur.model.tokenizer,
            self.min_lemmas_per_synset,
            self.max_token_count,
        )
        lemmas = df[df["synset"] == synset]["lemma"]
        lemmas = lemmas.to_pandas() if hasattr(lemmas, "to_pandas") else lemmas
        return {lemma.strip().lower() for lemma in lemmas.tolist() if lemma.strip()}

    @staticmethod
    def contains_any(text: str, words: set[str]) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in words if word)

    def concept_success(self, continuation: str, synset: str) -> bool:
        """Presence-based success: any member lemma appears in the text."""
        return self.contains_any(continuation, self.member_lemmas(synset))

    @staticmethod
    def continuation(text: str, prompt: str) -> str:
        """Best-effort extraction of the generated part (drop the prompt).

        Needed because success must not be triggered by concept words already
        present in the prompt.
        """
        if text.startswith(prompt):
            return text[len(prompt) :]
        marker = "assistant<|end_header_id|>"
        if marker in text:
            return text.split(marker)[-1]
        shared = 0
        for a, b in zip(text, prompt):
            if a != b:
                break
            shared += 1
        return text[shared:]

    def perplexity(self, text: str) -> float:
        """Perplexity of a text under the unsteered model.

        Callers should pass the CONTINUATION, not the full text: full decoded
        outputs contain left-padding tokens (batched generation) and chat
        markup, both of which distort perplexity wildly (observed >20k on
        otherwise fluent baselines). Continuation-only PPL is unconditional
        on the prompt — a v1 simplification.
        """
        return float(self.fur.loss([text]).exp())

    def generate_baseline(
        self, prompts: list[str], max_new_tokens: int = 48
    ) -> list[str]:
        """Unsteered greedy continuations for the same prompts."""
        return self.fur.generate(
            prompts, max_new_tokens=max_new_tokens, do_sample=False
        )

    def evaluate(
        self,
        prompts: list[str],
        texts: list[str],
        synsets: list[str],
        config: dict,
        probe: torch.Tensor | None = None,
        recorder: RecordingScorer | None = None,
        baseline_texts: list[str] | None = None,
    ) -> list[dict]:
        """Score generations, append one JSONL record per prompt, return records.

        Args:
            prompts: Input prompts.
            texts: Steered generations (any method).
            synsets: Synsets to measure presence-success against.
            config: Method description (family, k, steps, weights, ...) —
                stored verbatim in every record.
            probe: Optional (n, T) probe tensor; final value is logged.
            recorder: Optional RecordingScorer used during generation;
                its per-step traces are logged.
            baseline_texts: Unsteered continuations; generated (greedy) if
                omitted.
        """
        if baseline_texts is None:
            baseline_texts = self.generate_baseline(prompts)

        records = []
        for i, (prompt, text) in enumerate(zip(prompts, texts)):
            continuation = self.continuation(text, prompt)
            baseline_continuation = self.continuation(baseline_texts[i], prompt)
            ppl = self.perplexity(continuation or text)
            baseline_ppl = self.perplexity(baseline_continuation or baseline_texts[i])
            ratio = ppl / baseline_ppl

            records.append(
                {
                    "prompt": prompt,
                    "text": text,
                    "continuation": continuation,
                    "config": config,
                    "success": {
                        synset: self.concept_success(continuation, synset)
                        for synset in synsets
                    },
                    "ppl": ppl,
                    "baseline_ppl": baseline_ppl,
                    "ppl_ratio": ratio,
                    "fluency_flag": ratio > self.ppl_ratio_threshold,
                    "probe_final": (float(probe[i, -1]) if probe is not None else None),
                    "steps": recorder.per_input_steps(i) if recorder else None,
                }
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return records
