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
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from ..linalg.angles import (
    grassmann_distance,
    mean_canonical_correlation,
    text_frame,
    vector_subspace_cosine,
)


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
        self._expression_cache: dict = {}

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

    def concept_expression(self, text: str, concept) -> float:
        """Metric v2: mean per-position projection of a text onto a concept.

        Continuous concept-expression — measures how much the text aligns
        with the concept frame, without requiring exact member-lemma surface
        forms (which E0 showed almost never appear even when steering
        visibly works). Method-agnostic: works for any family's output.
        Results are cached per (text, concept name) since baselines repeat
        across cells.
        """
        if not text.strip():
            return 0.0
        key = (text, str(concept.synset))
        if key not in self._expression_cache:
            with torch.inference_mode():
                projections = self.fur.project(text, concept)
            self._expression_cache[key] = float(projections.float().mean())
        return self._expression_cache[key]

    def expression_record(
        self, continuation: str, baseline_continuation: str, concepts: dict
    ) -> dict:
        """Steered vs baseline expression per concept, with the delta.

        The delta (steered - baseline) is the steering effect: how much more
        the steered continuation expresses the concept than the unsteered one.
        Raw values are logged; scale-sensitive aggregation across concepts
        (e.g. standardized min for a joint measure) belongs in the analysis.
        """
        record = {}
        for name, concept in concepts.items():
            steered = self.concept_expression(continuation, concept)
            baseline = self.concept_expression(baseline_continuation, concept)
            record[name] = {
                "steered": steered,
                "baseline": baseline,
                "delta": steered - baseline,
            }
        return record

    def concept_angles(self, text: str, concept) -> dict:
        """Metric v3: gauge-invariant angle/distance between text and concept.

        Grassmann geometry against the concept's span — bounded and
        scale-invariant, unlike `concept_expression` (a linear trace
        correlation into which hidden-state norms enter directly), so values
        are comparable across models and precisions:

        - "token_cos": mean over positions of the cosine between each hidden
          state and the concept span (fraction of norm in the span, [0, 1]);
        - "geodesic": Grassmann geodesic distance between the text's dominant
          k-dim subspace and the concept span ([0, pi/2 * sqrt(k)]);
        - "mcc": mean canonical correlation of the same pair ([0, 1]).

        Deliberately invariant to the frame's column order and sign —
        evaluation asks "did the text land near the span", unlike the
        gauge-sensitive Stiefel steering objective. Uses the same
        tokenize -> last-hidden-state path as `concept_expression`.
        Cached per (text, concept name).
        """
        k = len(concept)
        if not text.strip():
            return {
                "token_cos": 0.0,
                "geodesic": math.pi / 2 * math.sqrt(k),
                "mcc": 0.0,
            }
        key = (text, str(concept.synset), "angles")
        if key not in self._expression_cache:
            with torch.inference_mode():
                tokens = self.fur.model.tokenize(text)
                hidden = self.fur.model.forward_last_hiden_state(tokens)
            hidden = hidden.squeeze(0).to(concept.device)
            frame = concept.tensor.squeeze(0)
            dominant = text_frame(hidden, k)
            self._expression_cache[key] = {
                "token_cos": float(vector_subspace_cosine(hidden, frame).mean()),
                "geodesic": float(grassmann_distance(dominant, frame)),
                "mcc": float(mean_canonical_correlation(dominant, frame)),
            }
        return self._expression_cache[key]

    def angle_record(
        self, continuation: str, baseline_continuation: str, concepts: dict
    ) -> dict:
        """Steered vs baseline angle metrics per concept (metric v3).

        Mirrors `expression_record`. Deltas are oriented so positive always
        means "moved toward the concept": cosine-like quantities use
        steered - baseline; the geodesic distance uses baseline - steered.
        """
        record = {}
        for name, concept in concepts.items():
            steered = self.concept_angles(continuation, concept)
            baseline = self.concept_angles(baseline_continuation, concept)
            record[name] = {
                "steered": steered,
                "baseline": baseline,
                "delta": {
                    "token_cos": steered["token_cos"] - baseline["token_cos"],
                    "geodesic": baseline["geodesic"] - steered["geodesic"],
                    "mcc": steered["mcc"] - baseline["mcc"],
                },
            }
        return record

    def perplexity(self, text: str) -> float:
        """Perplexity of a text under the unsteered model.

        Callers should pass the CONTINUATION, not the full text: full decoded
        outputs contain left-padding tokens (batched generation) and chat
        markup, both of which distort perplexity wildly (observed >20k on
        otherwise fluent baselines). Continuation-only PPL is unconditional
        on the prompt — a v1 simplification.
        """
        return float(self.fur.loss([text]).exp())

    def continuation_token_nll(self, text: str) -> tuple[list[int], torch.Tensor]:
        """Return continuation token IDs and prefix-conditioned token NLLs.

        The returned loss at position ``t`` is ``-log p(x_t | x_<t)``.  The
        tokenizer's BOS supplies context for the first continuation token, but
        no prompt or chat-formatted text is included.  This preserves the
        continuation-only PPL contract while allowing a later token span to be
        scored without resetting that span to BOS.

        The current long-generation notebooks use a Llama tokenizer with one
        leading BOS.  Reject another convention rather than silently shifting
        the token/loss alignment used by trajectory bins.
        """
        tokens = self.fur.model.tokenize(text)
        if tokens.ndim != 2 or tokens.size(0) != 1:
            raise ValueError("continuation_token_nll expects exactly one text")
        token_ids = tokens.squeeze(0)
        bos_token_id = self.fur.model.tokenizer.bos_token_id
        if not len(token_ids) or token_ids[0].item() != bos_token_id:
            raise ValueError("continuation tokenizer must prepend exactly one BOS")
        if len(token_ids) == 1:
            return [], torch.empty(0, dtype=torch.float32)

        with torch.inference_mode():
            output = self.fur.model(tokens, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output["logits"]
        labels = tokens[:, 1:].to(logits.device)
        token_nll = F.cross_entropy(
            logits[:, :-1, :].float().transpose(1, 2),
            labels,
            reduction="none",
        ).squeeze(0)
        return token_ids[1:].cpu().tolist(), token_nll.float().cpu()

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
        concepts: dict | None = None,
        angles: bool = False,
        metadata: list[dict] | None = None,
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
            concepts: Optional {name: Concept} map — enables metric v2
                (continuous concept-expression, steered vs baseline delta).
            angles: If True (and `concepts` given), also log metric v3 —
                gauge-invariant angle deltas per concept (`angle_record`).
                Default False keeps the record schema of prior experiments.
            metadata: Optional per-prompt derived metadata (for example actual
                continuation length, termination, and generation time). It is
                stored under ``record["metadata"]`` and does not alter the
                experiment configuration.
        """
        if metadata is not None and len(metadata) != len(prompts):
            raise ValueError("metadata must contain one mapping per prompt")
        if baseline_texts is None:
            baseline_texts = self.generate_baseline(prompts)

        records = []
        for i, (prompt, text) in enumerate(zip(prompts, texts)):
            continuation = self.continuation(text, prompt)
            baseline_continuation = self.continuation(baseline_texts[i], prompt)
            ppl = self.perplexity(continuation or text)
            baseline_ppl = self.perplexity(baseline_continuation or baseline_texts[i])
            ratio = ppl / baseline_ppl

            record = {
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
                "expression": (
                    self.expression_record(
                        continuation, baseline_continuation, concepts
                    )
                    if concepts
                    else None
                ),
                "angles": (
                    self.angle_record(continuation, baseline_continuation, concepts)
                    if angles and concepts
                    else None
                ),
                "probe_final": (float(probe[i, -1]) if probe is not None else None),
                "steps": recorder.per_input_steps(i) if recorder else None,
            }
            if metadata is not None:
                record["metadata"] = metadata[i]
            records.append(record)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return records
