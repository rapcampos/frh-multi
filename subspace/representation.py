"""Subspace-based concept representation over a HuggingFace unembedding.

Reuses the FRH loaders — model wrapper, whitened-representation layer, and
WordNet tokenization — but represents each concept as a linear subspace instead
of a Stiefel frame. Front end (lemma → token → whitened+centered feature) is
identical to :class:`frames.representations.frame.FrameUnembeddingRepresentation`;
the concept is then the top-``r`` SVD subspace of the pooled token features.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import pandas as pd
import torch

from frames.abstract import BaseModel
from frames.models.hf import BaseHuggingFaceModel, LanguageHuggingFaceModel
from frames.nlp import MultiLingualWordNetSynsets, SupportedLanguages
from frames.representations.unembedding import LinearUnembeddingRepresentation

from .concept import (
    ConceptSubspace,
    lemma_token_rows,
    subspace_from_features,
    subspace_from_lemma_features,
)
from .generation import (
    select_best_candidate,
    select_best_candidate_fluent,
    subspace_alignment,
)
from .positional import (
    PositionalConceptSubspace,
    positional_alignment,
    positional_token_ids,
)

ConceptGuide = ConceptSubspace | PositionalConceptSubspace
Method = Literal["pooled", "hierarchical", "positional"]

SUBSPACE_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "subspaces"


class SubspaceUnembeddingRepresentation(BaseModel, arbitrary_types_allowed=True):
    """Build, project onto, and generate with subspace concepts."""

    model: BaseHuggingFaceModel
    unembedding: LinearUnembeddingRepresentation
    data: MultiLingualWordNetSynsets = MultiLingualWordNetSynsets()
    rank: int = 8
    method: Method = "pooled"
    r_lemma: int = 1

    @classmethod
    def from_model_id(
        cls,
        id: str,
        rank: int = 8,
        method: Method = "pooled",
        r_lemma: int = 1,
        model_cls: type[BaseHuggingFaceModel] = LanguageHuggingFaceModel,
        language_codes: SupportedLanguages | None = None,
        synsets_kwargs: dict | None = None,
        *args,
        **kwargs,
    ) -> "SubspaceUnembeddingRepresentation":
        """Mirror of ``FrameUnembeddingRepresentation.from_model_id`` for subspaces.

        ``method`` selects the concept construction (``"pooled"`` — the flat
        one-vote-per-token SVD; ``"hierarchical"`` — the one-vote-per-lemma
        subspace-of-subspaces build, see :func:`subspace_from_lemma_features`).
        ``"positional"`` builds an independent subspace for each token position.
        All are per-call overridable on :meth:`get_concept` /
        :meth:`get_concept_cached` so one loaded model can build every arm.
        """
        model = model_cls(id=id, *args, **kwargs)
        return cls(
            model=model,
            unembedding=LinearUnembeddingRepresentation(model=model),
            data=MultiLingualWordNetSynsets(
                language_codes=language_codes or SupportedLanguages.from_model_id(id),
                **(synsets_kwargs or {}),
            ),
            rank=rank,
            method=method,
            r_lemma=r_lemma,
        )

    def _resolve(
        self, method: Method | None, r_lemma: int | None
    ) -> tuple[Method, int]:
        """Fall back to the instance defaults for per-call overrides."""
        method = method if method is not None else self.method
        if method not in ("pooled", "hierarchical", "positional"):
            raise ValueError(f"unknown method {method!r}")
        return (
            method,
            r_lemma if r_lemma is not None else self.r_lemma,
        )

    # -- construction ---------------------------------------------------------

    def _features_for_ids(self, token_ids: np.ndarray) -> torch.Tensor:
        """Whitened+centered features of a pool of token ids, pad dropped → (N, d)."""
        ids = torch.as_tensor(np.asarray(token_ids).reshape(-1), dtype=torch.long)
        pad_id = self.model.tokenizer.pad_token_id
        if pad_id is not None:
            ids = ids[ids != pad_id]
        return self.unembedding.get_token_representations(ids)  # (N, d)

    def build_concept_subspace(self, token_ids: np.ndarray) -> torch.Tensor:
        """Top-``rank`` orthonormal basis of a pool of token ids (pooled build).

        Whitened+centered features of the tokens (reusing the unembedding layer)
        are pooled and reduced to their principal subspace via SVD.
        """
        return subspace_from_features(self._features_for_ids(token_ids), self.rank)

    def _lemma_feature_blocks(self, synsets: pd.DataFrame) -> list[torch.Tensor]:
        """One ``(nᵢ, d)`` feature block per lemma — the hierarchical input.

        ``get_all_synsets`` gives one row per synset whose ``tokens`` cell stacks
        every lemma into a single ``(n_lemmas, max_tokens)`` array, so split it
        per lemma via :func:`lemma_token_rows` (iterating the column directly
        would collapse the whole synset into one block → a rank-1 concept).
        """
        pad_id = self.model.tokenizer.pad_token_id
        return [
            self._features_for_ids(ids)
            for stacked in synsets["tokens"].values
            for ids in lemma_token_rows(stacked, pad_id)
        ]

    def compute_concept(
        self,
        synsets: pd.DataFrame,
        method: Method | None = None,
        r_lemma: int | None = None,
        *,
        max_token_count: int | None = None,
    ) -> ConceptGuide:
        """Build a concept subspace from a synset DataFrame slice.

        ``method`` picks the construction (``"pooled"`` pools every token;
        ``"hierarchical"`` keeps each lemma/row as its own subspace, then
        combines; ``"positional"`` builds each token column independently).
        Positional construction requires explicit ``max_token_count`` and
        excludes longer words. ``None`` falls back to the instance defaults.
        """
        method, r_lemma = self._resolve(method, r_lemma)
        if synsets.empty:
            raise ValueError("cannot build a concept from an empty DataFrame")
        synset = (
            synsets["synset"].tolist()
            if len(synsets) > 1
            else synsets.iloc[0]["synset"]
        )

        if method == "positional":
            if max_token_count is None:
                raise ValueError("positional construction requires max_token_count")
            columns = positional_token_ids(
                synsets["tokens"].values,
                max_token_count,
                self.model.tokenizer.pad_token_id,
            )
            concept = PositionalConceptSubspace.from_features(
                synset, [self._features_for_ids(ids) for ids in columns], self.rank
            )
            concept.metadata.update(
                {
                    "model_id": self.model.id,
                    "languages": self.data.language_codes.name,
                }
            )
            return concept
        if method == "hierarchical":
            basis = subspace_from_lemma_features(
                self._lemma_feature_blocks(synsets), self.rank, r_lemma
            )
        elif method == "pooled":
            token_pool = np.concatenate(
                [np.asarray(t).reshape(-1) for t in synsets["tokens"].values]
            )
            basis = self.build_concept_subspace(token_pool)
        else:
            raise ValueError(f"unknown method {method!r}")

        return ConceptSubspace(synset=synset, basis=basis)

    def get_concept(
        self,
        synset: str,
        min_lemmas_per_synset: int,
        max_token_count: int,
        method: Method | None = None,
        r_lemma: int | None = None,
    ) -> ConceptGuide:
        """Build the subspace for a single named synset."""
        synsets = self.data.get_all_synsets(
            self.model.tokenizer, min_lemmas_per_synset, max_token_count
        )
        synsets = synsets[synsets["synset"] == synset]
        if synsets.empty:
            raise ValueError(f"Synset '{synset}' not found.")
        concept = self.compute_concept(
            synsets, method=method, r_lemma=r_lemma, max_token_count=max_token_count
        )
        if isinstance(concept, PositionalConceptSubspace):
            concept.metadata["min_lemmas_per_synset"] = min_lemmas_per_synset
        return concept

    def _concept_cache_path(
        self,
        synset: str,
        min_lemmas_per_synset: int,
        max_token_count: int,
        method: Method = "pooled",
        r_lemma: int = 1,
    ) -> Path:
        model_slug = self.model.id.replace("/", "--")
        languages = self.data.language_codes.name
        # Pooled keeps the original filename (existing caches stay valid);
        # hierarchical adds a tag so the two constructions never collide.
        tag = "" if method == "pooled" else f"_hier_rl{r_lemma}"
        if method == "positional":
            tag = "_positional_v1"
        fname = (
            f"{synset}__l{min_lemmas_per_synset}"
            f"_t{max_token_count}_r{self.rank}{tag}_{languages}.pt"
        )
        return SUBSPACE_CACHE_DIR / model_slug / fname

    def get_concept_cached(
        self,
        synset: str,
        min_lemmas_per_synset: int,
        max_token_count: int,
        method: Method | None = None,
        r_lemma: int | None = None,
    ) -> ConceptGuide:
        """Persist legacy basis tensors or a versioned positional payload."""
        method, r_lemma = self._resolve(method, r_lemma)
        path = self._concept_cache_path(
            synset, min_lemmas_per_synset, max_token_count, method, r_lemma
        )
        if path.exists():
            if method == "positional":
                payload = torch.load(
                    path, map_location=self.model.device, weights_only=True
                )
                if payload.get("version") != 1:
                    raise ValueError("unsupported positional cache version")
                concept = PositionalConceptSubspace(**payload["concept"])
                if (
                    len(concept) != max_token_count
                    or concept.rank != self.rank
                    or concept.synset != synset
                    or concept.metadata.get("model_id") != self.model.id
                    or concept.metadata.get("languages")
                    != self.data.language_codes.name
                    or concept.metadata.get("min_lemmas_per_synset")
                    != min_lemmas_per_synset
                ):
                    raise ValueError(
                        "positional cache does not match requested concept"
                    )
                return concept
            basis = torch.load(path, map_location=self.model.device)
            return ConceptSubspace(synset=synset, basis=basis)

        concept = self.get_concept(
            synset,
            min_lemmas_per_synset,
            max_token_count,
            method=method,
            r_lemma=r_lemma,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(concept, PositionalConceptSubspace):
            torch.save({"version": 1, "concept": concept.model_dump()}, path)
        else:
            torch.save(concept.basis, path)
        return concept

    # -- projection -----------------------------------------------------------

    def _last_hidden_state(
        self, input_text: Union[str, List[str], torch.Tensor]
    ) -> torch.Tensor:
        x = (
            self.model.tokenize(input_text, padding=True)
            if not isinstance(input_text, torch.Tensor)
            else input_text
        )
        return self.model.forward_last_hiden_state(x)

    def project(
        self,
        input_text: Union[str, List[str], torch.Tensor],
        concept: ConceptGuide,
    ) -> torch.Tensor:
        """Raw projection magnitudes (window averages for positional guides)."""
        last_hs = self._last_hidden_state(input_text).to(concept.device)
        return concept.project(last_hs)

    def expression(
        self,
        input_text: Union[str, List[str], torch.Tensor],
        concept: ConceptGuide,
    ) -> torch.Tensor:
        """Mean captured energy; positional guides average real window pairs."""
        if isinstance(concept, PositionalConceptSubspace):
            tokens = (
                input_text
                if isinstance(input_text, torch.Tensor)
                else self.model.tokenize(input_text, padding=True)
            )
            last_hs = self._last_hidden_state(tokens).to(concept.device)
            pad_id = self.model.tokenizer.pad_token_id
            mask = (
                torch.ones_like(tokens, dtype=torch.bool)
                if pad_id is None
                else tokens != pad_id
            )
            return concept.expression(last_hs, mask)
        last_hs = self._last_hidden_state(input_text).to(concept.device)
        return concept.energy_fraction(last_hs).mean(-1)

    # -- model utilities (harness parity) -------------------------------------

    def loss(self, input_text: Union[str, List[str]]) -> torch.Tensor:
        """Per-text LM loss — the continuation-only PPL input the harness needs.

        Mirror of ``FrameUnembeddingRepresentation.loss`` so the shared
        :class:`frames.evaluation.harness.EvaluationHarness` can drive a subspace
        representation unchanged.
        """
        tokens = self.model.tokenize(input_text, padding=True).unsqueeze_(1)
        return torch.tensor([self.model(x, labels=x).loss for x in tokens])

    def generate(self, input_text: Union[str, List[str]], *args, **kwargs) -> List[str]:
        """Unsteered decoded continuations — the harness baseline generator.

        Mirror of ``FrameUnembeddingRepresentation.generate`` (decode the model's
        raw token output); forwards ``max_new_tokens`` / ``do_sample`` etc.
        """
        return self.model.decode(self.model.generate(input_text, *args, **kwargs))

    # -- generation -----------------------------------------------------------

    def _candidates(
        self, tokens: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k next-token candidates, last hidden state, and candidate logprobs.

        Single-pass mirror of
        ``FrameUnembeddingRepresentation._generate_candidates``: returns
        ``(candidates, last_hidden_state, logprobs)`` where ``candidates`` has
        shape ``(rows, k, T+1)``, ``last_hidden_state`` is ``(rows, T', d)``, and
        ``logprobs`` is ``(rows, k)`` — the log-probability of each row's ``k``
        candidate next tokens (the top-k is taken on softmax probabilities, whose
        log is monotone, so candidate identity and order are unchanged). The
        logprobs feed the ``fluency_weight`` dial in
        :meth:`generate_with_topk_guide`.
        """
        inputs = self.model.make_input(self.model.decode(tokens), padding=True)
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        logits, hidden_states = outputs["logits"], outputs["hidden_states"]

        top = logits.softmax(-1)[..., -1, :].topk(k)
        generated = top.indices.unsqueeze(-1)
        expanded = tokens.unsqueeze(-2).expand(-1, k, -1)
        candidates = torch.cat([expanded, generated], dim=-1)
        return candidates, hidden_states[-1], top.values.log()

    def _is_generation_complete(self, tokens: torch.Tensor) -> bool:
        return tokens.eq(self.model.tokenizer.eos_token_id).any(-1).all().item()

    @torch.inference_mode()
    def generate_with_topk_guide(
        self,
        input_text: Union[str, List[str]],
        guide: ConceptGuide,
        k: int = 2,
        steps: int = 16,
        fluency_weight: float = 0.0,
        repel_guide: ConceptGuide | None = None,
        normalize_score: bool = False,
    ) -> tuple[List[str], torch.Tensor]:
        """Top-k guided generation steered toward a concept subspace.

        Same single-pass top-k guided algorithm as the original frame method
        (greedy with k-way lookahead, not beam search); each candidate's
        per-position hidden states are scored by projection magnitude onto
        ``guide`` (instead of frame correlation), and at each step the candidate
        with the highest cumulative alignment is kept and its ``k`` lookahead
        children are carried forward.

        ``guide`` may be a single concept or a :meth:`ConceptSubspace.blend` of
        several — the blended concept is one subspace, so multi-concept steering
        reuses this loop unchanged (Family-1 composition).

        Positional concepts instead score forward windows with their ordered
        bases, right-zero-padding the tail and averaging over active positions
        with a fixed denominator. Use :meth:`PositionalConceptSubspace.blend`
        for positionwise composition. Attractive and repelled positional guides
        must share width and feature dimension; mixed guide types are rejected.

        Supplying ``repel_guide`` creates a signed score-space guide: each
        candidate position receives ``project(guide) - project(repel_guide)``.
        This changes only the score before cumulative greedy selection; it does
        not negate a basis, whose projection magnitude would be unchanged.

        With ``fluency_weight`` = ``λ`` > 0, selection maximizes the blended
        objective ``cumulative alignment + λ · cumulative token log-probability``
        instead of alignment alone — the fluency dial, matching the frame
        pipeline's ``fluency_weight`` semantics. The default ``0.0`` leaves the
        original selection path untouched. The returned probe stays the concept
        (alignment) trajectory of the selected candidate either way.

        ``normalize_score`` divides each per-position projection magnitude by
        ``‖h‖``, scoring the square root of captured energy fraction instead of
        absolute magnitude. This is bounded, but does not guarantee identical
        fluency calibration across representations. The default ``False``
        reproduces the original scale-dependent score.
        """
        input_text = [input_text] if isinstance(input_text, str) else input_text
        n = len(input_text)
        positional = isinstance(guide, PositionalConceptSubspace)
        if repel_guide is not None:
            if positional != isinstance(repel_guide, PositionalConceptSubspace):
                raise ValueError("cannot mix pooled and positional guides")
            if positional and guide.bases.shape[:2] != repel_guide.bases.shape[:2]:
                raise ValueError(
                    "attraction and repulsion must share width and feature dimension"
                )
        if not positional:
            basis = guide.basis.to(self.model.device)
            repel_basis = (
                repel_guide.basis.to(self.model.device)
                if repel_guide is not None
                else None
            )

        inputs = self.model.make_input(input_text, padding=True)
        first, _, first_logprobs = self._candidates(inputs["input_ids"], k)
        tokens = first.flatten(0, 1)  # (n*k, T+1)
        cum_logprob = first_logprobs.flatten()  # (n*k,)

        probe = torch.zeros(n, 1)
        for _ in range(steps):
            sequences, last_hs, child_logprobs = self._candidates(tokens, k)
            if positional:
                scores = positional_alignment(
                    last_hs, guide, repel_guide, normalize_score
                )
            else:
                scores = subspace_alignment(
                    last_hs, basis, repel_basis, normalize=normalize_score
                )  # (n*k, T')
            if fluency_weight:
                col_idx, probe = select_best_candidate_fluent(
                    scores, cum_logprob, fluency_weight, n, k
                )
            else:
                col_idx, probe = select_best_candidate(scores, n, k)

            candidate_tokens = sequences.reshape(n, k, k, -1)
            tokens = candidate_tokens[np.arange(n), col_idx.cpu()].flatten(0, 1)

            if fluency_weight:
                rows = torch.arange(n, device=cum_logprob.device) * k + col_idx.to(
                    cum_logprob.device
                )
                cum_logprob = (
                    cum_logprob[rows].unsqueeze(-1) + child_logprobs[rows]
                ).flatten()

            if self._is_generation_complete(tokens):
                break

        return self.model.decode(tokens[::k]), probe.cpu()
