from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, List, Union

import numpy as np
import pandas as pd
import torch
from methodtools import lru_cache
from tqdm import tqdm

from frames.linalg.orthogonalization import solve_procrustes
from frames.models.hf import BaseHuggingFaceModel, LanguageHuggingFaceModel

from ..linalg import Frame
from ..nlp import MultiLingualWordNetSynsets, SupportedLanguages
from ..utils.stdlib import batchedlist
from . import aggregators
from .concept import Concept
from .unembedding import LinearUnembeddingRepresentation

CONCEPT_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "concepts"


class FrameUnembeddingRepresentation(LinearUnembeddingRepresentation):
    """
    A class for frame-based unembedding representation operations.
    """

    data: MultiLingualWordNetSynsets = MultiLingualWordNetSynsets()

    def build_concept_frame(
        self, tokens: list[int] | pd.Series, *args, **kwargs
    ) -> torch.Tensor:
        tokens = self._make_word_frames(tokens)
        return self.average_frames(tokens, *args, **kwargs)

    def average_frames(self, word_frames: torch.Tensor, dim: int = -3) -> torch.Tensor:
        """
        Average frames across a specified dimension.

        Args:
            word_frames (torch.Tensor): Tensor of word frames.
            dim (int): Dimension to average over.

        Returns:
            torch.Tensor: frame mean.
        """
        return solve_procrustes(word_frames.sum(dim=dim))

    def _compute_all_concepts(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute all concepts based on yielded concepts.

        Returns:
            torch.Tensor: Tensor of all computed concepts.
        """
        return torch.cat(list(self._yield_concepts(*args, **kwargs))).mT

    def _make_word_frames(self, tokens: Union[List[int], pd.Series]) -> torch.Tensor:
        """
        Create word frames from tokens.

        Args:
            tokens (Union[List[int], pd.Series]): List or Series of token IDs.

        Returns:
            torch.Tensor: Tensor of word frames.
        """
        return self.get_token_representations(np.stack(tokens))

    def _project_probes(
        self, input_text: Union[str, List[str]], *args, **kwargs
    ) -> tuple[Concept, torch.Tensor]:
        """
        Project probes for input text.

        Args:
            input_text (Union[str, List[str]]): Input text to project.

        Returns:
            tuple[Concept, torch.Tensor]: tuple of concepts and their projections.
        """
        concepts = self.get_all_concepts(*args, **kwargs)
        projections = self.project(input_text, concepts)
        return concepts, projections

    def _topk_index_probes(
        self, input_text: Union[str, List[str]], k: int, *args, **kwargs
    ) -> tuple[Concept, torch.Tensor]:
        """
        Get top-k index probes for input text.

        Args:
            input_text (Union[str, List[str]]): Input text to probe.
            k (int): Number of top probes to return.

        Returns:
            tuple[Concept, torch.Tensor]: tuple of concepts and their indices.
        """
        concepts, projections = self._project_probes(input_text, *args, **kwargs)
        indices = projections.topk(k=k, dim=-1).indices.cpu()
        return concepts, indices

    def _yield_concepts(
        self, tokens: np.ndarray, batch_size: int = 1 << 6, *args, **kwargs
    ) -> Iterator[torch.Tensor]:
        """
        Yield concept frames in batches.

        Args:
            tokens (np.ndarray): Array of tokens.
            batch_size (int): Size of each batch.

        Yields:
            torch.Tensor: Batches of concepts.
        """
        num_batches = max(len(tokens) // batch_size, 1)
        for batch in np.array_split(tokens, num_batches):
            words = self._make_word_frames(batch)
            yield self.average_frames(words, *args, **kwargs)

    @lru_cache()
    def get_all_concepts(self, *args, **kwargs) -> Concept:
        """
        Get all concepts.

        Returns:
            Concept: All concepts with their synsets and frames.
        """
        synsets = self.data.get_all_synsets(self.model.tokenizer, *args, **kwargs)
        return self.compute_concept(synsets)

    @lru_cache()
    def get_all_words_frames(self, *args, **kwargs) -> Frame:
        synsets = self.data.get_all_synsets(self.model.tokenizer, *args, **kwargs)
        return Frame(
            tensor=self._make_word_frames(synsets["tokens"]).mT.transpose(0, 1)
        )

    def compute_concept(self, synsets: pd.DataFrame) -> Concept:
        return Concept(
            synset=synsets["synset"] if len(synsets) > 1 else synsets.iloc[0]["synset"],
            tensor=self._compute_all_concepts(synsets["tokens"].values),
        )

    def get_concept(self, synset: str, *args, **kwargs) -> Concept:
        synsets = self.data.get_all_synsets(self.model.tokenizer, *args, **kwargs)
        synsets = synsets[synsets["synset"] == synset]
        if synsets.empty:
            raise ValueError(f"Synset '{synset}' not found.")
        return self.compute_concept(synsets)

    def _concept_cache_path(
        self, synset: str, min_lemmas_per_synset: int, max_token_count: int
    ) -> Path:
        """Cache file keyed by model, synset, filter args, and language set."""
        model_slug = self.model.id.replace("/", "--")
        languages = self.data.language_codes.name
        fname = (
            f"{synset}__l{min_lemmas_per_synset}" f"_t{max_token_count}_{languages}.pt"
        )
        return CONCEPT_CACHE_DIR / model_slug / fname

    def get_concept_cached(
        self, synset: str, min_lemmas_per_synset: int, max_token_count: int
    ) -> Concept:
        """Like `get_concept`, but persists the frame tensor to disk.

        First call builds the concept and writes it to `cache/concepts/`;
        later calls (including across sessions) load the tensor directly,
        skipping WordNet tokenization and frame construction.
        """
        path = self._concept_cache_path(synset, min_lemmas_per_synset, max_token_count)
        if path.exists():
            tensor = torch.load(path, map_location=self.model.device)
            return Concept(synset=synset, tensor=tensor)

        concept = self.get_concept(synset, min_lemmas_per_synset, max_token_count)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(concept.tensor, path)
        return concept

    def concept_pair(
        self,
        synset_a: str,
        synset_b: str,
        min_lemmas_per_synset: int,
        max_token_count: int,
    ) -> tuple[Concept, Concept, float]:
        """Build two concept frames (disk-cached) and their correlation rho.

        rho predicts steering interference between the concepts; see
        `Frame.rho` for the unequal-rank policy.

        Returns:
            tuple[Concept, Concept, float]: concept A, concept B, rho(A, B).
        """
        cargs = min_lemmas_per_synset, max_token_count
        concept_a = self.get_concept_cached(synset_a, *cargs)
        concept_b = self.get_concept_cached(synset_b, *cargs)
        return concept_a, concept_b, concept_a.rho(concept_b).item()

    def _generate(
        self, input_text: Union[str, List[str]], *args, **kwargs
    ) -> List[str]:
        """
        Private method to generate text based on input.

        Args:
            input_text (Union[str, List[str]]): Input text for generation.
            *args, **kwargs: Additional arguments for the generate method.

        Returns:
            List[str]: Generated text.
        """
        inputs = self.model.make_input(input_text, padding=True)
        gen = self.model._model.generate(
            *args, **inputs, pad_token_id=self.model.tokenizer.pad_token_id, **kwargs
        )
        return self.model.decode(gen)

    def generate(
        self, input_text: Union[str, List[str]], batch_size: int = 32, *args, **kwargs
    ) -> List[str]:
        """
        Generate text based on input, processing in batches.

        Args:
            input_text (Union[str, List[str]]): Input text for generation.
            batch_size (int): Number of inputs to process in each batch.
            *args, **kwargs: Additional arguments for the generate method.

        Returns:
            List[str]: Generated text.
        """
        return [
            text
            for batch in tqdm(batchedlist(input_text, batch_size))
            for text in self._generate(batch, *args, **kwargs)
        ]

    @staticmethod
    def _weighted_sum_score(
        projections: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Default scorer: weighted sum of the per-concept projection stack."""
        return aggregators.weighted_sum(projections, weights)

    def _project_hidden_states(
        self, last_hs: torch.Tensor, concept: Concept
    ) -> torch.Tensor:
        """Project hidden states onto a concept via sliding frame windows."""
        last_hs_padded = torch.nn.functional.pad(last_hs, (0, 0, 0, len(concept) - 1))
        last_hs_frames = last_hs_padded.unfold(-2, size=len(concept), step=1).squeeze_(
            0
        )
        return last_hs_frames * concept

    def _score_hidden_states(
        self,
        last_hs: torch.Tensor,
        concepts: List[Concept],
        weights: list[float],
        scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n: int | None = None,
        normalize: str | None = None,
    ) -> torch.Tensor:
        """Project hidden states onto every concept and aggregate via `scorer`.

        With `normalize` set ("zscore" | "rank"), per-concept projections are
        normalized across each input's candidate pool before aggregation
        (requires `n`, the number of inputs). Mandatory for genuine Family 2
        composition — see `frames.representations.aggregators`.
        """
        last_hs = last_hs.to(concepts[0].device)
        projections = torch.stack(
            [self._project_hidden_states(last_hs, c) for c in concepts], dim=-1
        )
        if normalize is not None:
            shape = projections.shape
            pooled = projections.reshape(n, -1, *shape[1:])
            pooled = aggregators.normalize_scores(pooled, method=normalize, dim=1)
            projections = pooled.reshape(shape)
        weights_tensor = torch.tensor(
            weights, device=projections.device, dtype=projections.dtype
        )
        return scorer(projections, weights_tensor)

    @torch.inference_mode()
    def _generate_guided(
        self,
        input_text: Union[str, List[str]],
        concepts: List[Concept],
        weights: list[float] | None = None,
        k: int = 2,
        steps: int = 16,
        scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        normalize: str | None = None,
    ) -> tuple[List[str], torch.Tensor]:
        """
        Shared top-k guided generation loop.

        At each step, generates k candidate continuations per beam, projects the
        hidden states onto every concept, aggregates the per-concept projections
        into a single score via `scorer`, and keeps the k children of the beam
        with the highest cumulative score.

        Args:
            input_text (Union[str, List[str]]): Input text for generation.
            concepts (List[Concept]): Concepts to guide the generation.
            weights (list[float] | None): Per-concept weights (default: all 1.0).
            k (int): Number of top candidates to consider.
            steps (int): Number of generation steps.
            scorer: Aggregates the per-concept projection stack (..., n_concepts)
                and the weights tensor into one score tensor (...). Defaults to
                weighted sum, which reproduces the original single-concept method
                exactly when called with one concept and weight 1.0.
            normalize: Per-step per-concept score normalization over the
                candidate pool ("zscore" | "rank" | None). Must stay None on
                the golden path; mandatory for genuine multi-concept
                composition (see `frames.representations.aggregators`).

        Returns:
            tuple[List[str], torch.Tensor]: Generated text and probe values.
        """
        input_text = self._prepare_input_text(input_text)
        n = len(input_text)

        weights = weights if weights is not None else [1.0] * len(concepts)
        scorer = scorer if scorer is not None else self._weighted_sum_score

        inputs = self.model.make_input(input_text, padding=True)
        tokens = self._generate_candidates(inputs["input_ids"], k)[0].flatten(0, 1)

        for _ in range(steps):
            # notice this approach uses a single pass through the model
            #  at the expense of having k times more candidates being processes.
            #  The other option is to use double pass, but takes longer.
            #  On larger models, our bottleneck is time, not memory,
            #  so we choose this approach for now.
            sequences, hidden_states = self._generate_candidates(tokens, k)

            scores = self._score_hidden_states(
                hidden_states[-1], concepts, weights, scorer, n, normalize
            )

            max_probe = scores.cumsum(-2).reshape(n, k, -1).max(-2)

            row_idx = np.arange(n)
            col_idx = max_probe.indices[..., -1].cpu()
            candidate_tokens = sequences.reshape(n, k, k, -1)
            tokens = candidate_tokens[row_idx, col_idx].flatten(0, 1)

            if self._is_generation_complete(tokens):
                break

        return self.model.decode(tokens[::k]), max_probe.values.cpu()

    def _generate_with_topk_guide(
        self,
        input_text: Union[str, List[str]],
        guide: Concept,
        k: int = 2,
        steps: int = 16,
    ) -> List[str]:
        """
        Generate text with top-k guided approach (original paper method).

        Thin wrapper over `_generate_guided` with a single concept; verified
        token-for-token against the golden file (tests/test_golden.py).

        Args:
            input_text (Union[str, List[str]]): Input text for generation.
            guide (Concept): Concept to guide the generation.
            k (int): Number of top candidates to consider.
            steps (int): Number of generation steps.

        Returns:
            List[str]: Generated text.
        """
        return self._generate_guided(input_text, concepts=[guide], k=k, steps=steps)

    def generate_with_topk_guide(
        self,
        input_text: Union[str, List[str]],
        *args,
        batch_size: int = 32,
        **kwargs,
    ) -> List[str]:
        text, probe = [], []
        for batch in tqdm(batchedlist(input_text, batch_size)):
            batch_text, batch_probe = self._generate_with_topk_guide(
                batch, *args, **kwargs
            )
            text.extend(batch_text)
            probe.append(batch_probe)

        n = min(p.size(-1) for p in probe)
        return text, torch.cat([p[..., :n] for p in probe])

    @staticmethod
    def _topk_beam_selection(
        cum: torch.Tensor, candidate_tokens: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the top-k beams from the candidate pool by cumulative score.

        Unlike the original method (which keeps the k children of a single
        winning parent), all pool candidates compete regardless of parent,
        so surviving beams genuinely diverge.

        Args:
            cum (torch.Tensor): Cumulative scores per candidate, shape (n, m, T).
            candidate_tokens (torch.Tensor): Expansions of each candidate,
                shape (n, m, k, T'). Children are ordered by next-token
                probability (child 0 = greedy).
            k (int): Number of beams to keep.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The surviving beams' expansions
            (n, min(k, m), k, T'), sorted by score (beam 0 = best), and the
            best beam's cumulative score trajectory (n, T).
        """
        n = cum.size(0)
        beam_scores = cum[..., -1]
        top = beam_scores.topk(min(k, beam_scores.size(-1)), dim=-1).indices
        new_tokens = candidate_tokens[torch.arange(n).unsqueeze(-1), top.cpu()]
        best_probe = cum[torch.arange(n, device=cum.device), top[:, 0]]
        return new_tokens, best_probe

    @torch.inference_mode()
    def _generate_with_topk_beam_guide(
        self,
        input_text: Union[str, List[str]],
        concepts: List[Concept],
        weights: list[float] | None = None,
        k: int = 2,
        steps: int = 16,
        scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        normalize: str | None = None,
    ) -> tuple[List[str], torch.Tensor]:
        """
        True beam-search variant of top-k guided generation.

        Differs from the original `_generate_with_topk_guide` in one structural
        way: at each step, all pool candidates compete and the top-k sequences
        survive (the original keeps the k children of a single winning parent,
        so its beams only ever differ in their last token). Beams here
        genuinely diverge, at the cost of a k^2-wide forward pass per step
        after the first (mind memory for large k * batch_size).

        No length normalization is applied: all beams share the same length by
        construction (prompts are padded together and every beam grows one
        token per step), so cumulative scores are directly comparable. EOS
        handling mirrors the original: generation stops once every pool
        candidate contains an EOS token.

        Args:
            input_text (Union[str, List[str]]): Input text for generation.
            concepts (List[Concept]): Concepts to guide the generation.
            weights (list[float] | None): Per-concept weights (default: all 1.0).
            k (int): Number of beams to maintain.
            steps (int): Number of generation steps.
            scorer: Same swappable aggregator as `_generate_guided`.

        Returns:
            tuple[List[str], torch.Tensor]: Best beam per input and its
            cumulative score trajectory.
        """
        input_text = self._prepare_input_text(input_text)
        n = len(input_text)

        weights = weights if weights is not None else [1.0] * len(concepts)
        scorer = scorer if scorer is not None else self._weighted_sum_score

        inputs = self.model.make_input(input_text, padding=True)
        tokens = self._generate_candidates(inputs["input_ids"], k)[0].flatten(0, 1)
        m = k  # pool size per input: k initially, k^2 after the first selection

        for _ in range(steps):
            sequences, hidden_states = self._generate_candidates(tokens, k)

            scores = self._score_hidden_states(
                hidden_states[-1], concepts, weights, scorer, n, normalize
            )

            cum = scores.cumsum(-2).reshape(n, m, -1)
            candidate_tokens = sequences.reshape(n, m, k, -1)
            new_tokens, best_probe = self._topk_beam_selection(cum, candidate_tokens, k)

            tokens = new_tokens.flatten(1, 2).flatten(0, 1)
            m = tokens.size(0) // n

            if self._is_generation_complete(tokens):
                break

        best_tokens = new_tokens[:, 0, 0]
        return self.model.decode(best_tokens), best_probe.cpu()

    def generate_with_topk_beam_guide(
        self,
        input_text: Union[str, List[str]],
        *args,
        batch_size: int = 32,
        **kwargs,
    ) -> tuple[list[str], torch.Tensor]:
        text, probe = [], []
        for batch in tqdm(batchedlist(input_text, batch_size)):
            batch_text, batch_probe = self._generate_with_topk_beam_guide(
                batch, *args, **kwargs
            )
            text.extend(batch_text)
            probe.append(batch_probe)

        n = min(p.size(-1) for p in probe)
        return text, torch.cat([p[..., :n] for p in probe])

    def quick_generate_with_topk_beam_guide(
        self,
        sentences: list[str],
        guides: list[tuple[str, str]],
        min_lemmas_per_synset: int,
        max_token_count: int,
        *args,
        **kwargs,
    ) -> tuple[list[str], torch.Tensor]:
        cargs = min_lemmas_per_synset, max_token_count

        concepts = []
        for guide in guides:
            concept_A = self.get_concept(guide[0], *cargs)
            concept_B = self.get_concept(guide[1], *cargs) if len(guide) > 1 else None
            concepts.append(concept_A - concept_B if concept_B else concept_A)

        return self.generate_with_topk_beam_guide(
            sentences, *args, concepts=concepts, **kwargs
        )

    def _generate_with_topk_multi_guide(
        self,
        input_text: Union[str, List[str]],
        guides: list[Concept],
        weights: list[float] | None = None,
        k: int = 2,
        steps: int = 16,
        **kwargs,
    ) -> tuple:
        weights = weights if weights is not None else [1.0 / len(guides)] * len(guides)
        return self._generate_guided(
            input_text, concepts=guides, weights=weights, k=k, steps=steps, **kwargs
        )

    def generate_with_topk_multi_guide(
        self,
        input_text: Union[str, List[str]],
        *args,
        batch_size: int = 32,
        **kwargs,
    ) -> tuple[list[str], torch.Tensor]:
        text, probe = [], []
        for batch in tqdm(batchedlist(input_text, batch_size)):
            batch_text, batch_probe = self._generate_with_topk_multi_guide(
                batch, *args, **kwargs
            )
            text.extend(batch_text)
            probe.append(batch_probe)

        n = min(p.size(-1) for p in probe)
        return text, torch.cat([p[..., :n] for p in probe])

    def quick_generate_with_topk_multi_guide(
        self,
        sentences: list[str],
        guides: list[tuple[str, str]],
        min_lemmas_per_synset: int,
        max_token_count: int,
        *args,
        **kwargs,
    ) -> tuple[list[str], torch.Tensor]:
        cargs = min_lemmas_per_synset, max_token_count

        guides_concepts = []
        for guide in guides:
            concept_A = self.get_concept(guide[0], *cargs)
            concept_B = self.get_concept(guide[1], *cargs) if len(guide) > 1 else None
            guide_concept = concept_A - concept_B if concept_B else concept_A
            guides_concepts.append(guide_concept)

        return self.generate_with_topk_multi_guide(
            sentences, *args, guides=guides_concepts, **kwargs
        )

    def quick_generate_with_topk_subspace_guide(
        self,
        sentences: list[str],
        guides: list[tuple[str, str]],
        min_lemmas_per_synset: int,
        max_token_count: int,
        *args,
        **kwargs,
    ) -> tuple[list[str], torch.Tensor]:
        cargs = min_lemmas_per_synset, max_token_count

        guide_concepts = []
        for guide in guides:
            concept_A = self.get_concept(guide[0], *cargs)
            concept_B = self.get_concept(guide[1], *cargs) if len(guide) > 1 else None
            guide_concepts.append(concept_A - concept_B if concept_B else concept_A)

        averaged = Concept.average(guide_concepts)
        return self.generate_with_topk_guide(sentences, *args, guide=averaged, **kwargs)

    def _prepare_input_text(self, input_text: Union[str, List[str]]) -> List[str]:
        """Prepare input text for generation."""
        return [input_text] if isinstance(input_text, str) else input_text

    def _generate_candidates(self, tokens: torch.IntTensor, k: int) -> torch.Tensor:
        """Generate candidate texts."""
        inputs = self.model.make_input(self.model.decode(tokens), padding=True)
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        logits, hs = outputs["logits"], outputs["hidden_states"]

        generated = logits.softmax(-1)[..., -1, :].topk(k).indices.unsqueeze_(-1)
        tokens = tokens.unsqueeze(-2).expand(-1, k, -1)
        candidates = torch.cat([tokens, generated], dim=-1)

        return candidates, hs

    def _is_generation_complete(self, gen_text: torch.Tensor) -> bool:
        """Check if the generation is complete."""
        return gen_text.eq(self.model.tokenizer.eos_token_id).any(-1).all()

    def loss(self, input_text: Union[str, List[str]]) -> torch.Tensor:
        """
        Compute the loss for the input text.

        Args:
            input_text (Union[str, List[str]]): Input text to compute loss for.

        Returns:
            torch.Tensor: Computed loss.
        """
        tokens = self.model.tokenize(input_text, padding=True).unsqueeze_(1)
        return torch.tensor([self.model(x, labels=x).loss for x in tokens])

    def probe(
        self, input_text: Union[str, List[str], torch.Tensor], concept: Concept
    ) -> torch.Tensor:
        """
        Probe the input text with a given concept.

        Args:
            input_text (Union[str, List[str]]): Input text to probe.
            concept (Concept): Concept(s) to probe with.

        Returns:
            torch.Tensor: Probe results.
        """
        return self.project(input_text, concept).cumsum(-2)

    def probe_to_input(
        self, input_text: Union[str, List[str]], k: int, steps: int = 3, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Compute the probe-to-input relationship.

        Args:
            input_text (Union[str, List[str]]): Input text to analyze.
            k (int): Number of top probes to consider.
            steps (int): Number of steps for integrated gradients.

        Returns:
            pd.DataFrame: DataFrame with probe-to-input relationships.
        """
        names = self.probe_topk(input_text, k, *args, **kwargs)[-1]
        concepts, indices = self._topk_index_probes(input_text, k, *args, **kwargs)
        last_step_top_probes = concepts.frame.mT[indices[-1]].to(self.model.device).mT
        temp_fix = last_step_top_probes[..., 0].mT
        ig = self.model.integrated_gradients(
            input_text, temp_fix, steps, vectorize=True
        )

        return pd.DataFrame(
            data=ig[:, 1:].sum(-1).float().cpu(),
            index=names,
            columns=self.model._tokenizer.tokenize(input_text),
        ).style.background_gradient(cmap="coolwarm", axis=1)

    def probe_topk(
        self, input_text: Union[str, List[str]], k: int, *args, **kwargs
    ) -> List[str]:
        """
        Get the top-k probes for the input text.

        Args:
            input_text (Union[str, List[str]]): Input text to probe.
            k (int): Number of top probes to return.

        Returns:
            List[str]: List of top-k probe synsets.
        """
        concepts, indices = self._topk_index_probes(input_text, k, *args, **kwargs)
        return concepts.synset["synset"].values[indices]

    def project(
        self, input_text: Union[str, List[str], torch.Tensor], concept: Concept
    ) -> torch.Tensor:
        """
        Project input text onto a concept.

        Args:
            input_text (Union[str, List[str]]): Input text to project.
            concept (Concept): Concept(s) to project onto.

        Returns:
            torch.Tensor: Projection results.
        """
        x = (
            self.model.tokenize(input_text, padding=True)
            if not isinstance(input_text, torch.Tensor)
            else input_text
        )
        last_hs = self.model.forward_last_hiden_state(x).to(concept.device)
        return self._project_hidden_states(last_hs, concept)

    @classmethod
    def from_model_id(
        cls,
        id: str,
        model_cls: type[BaseHuggingFaceModel] = LanguageHuggingFaceModel,
        language_codes: SupportedLanguages | None = None,
        synsets_kwargs={},
        *args,
        **kwargs,
    ) -> FrameUnembeddingRepresentation:
        return cls(
            model=model_cls(id=id, *args, **kwargs),
            data=MultiLingualWordNetSynsets(
                language_codes=language_codes or SupportedLanguages.from_model_id(id),
                **synsets_kwargs,
            ),
        )

    def quick_generate_with_topk_guide(
        self,
        sentences: list[str],
        guide: tuple[str, str],
        min_lemmas_per_synset: int,
        max_token_count: int,
        *args,
        **kwargs,
    ) -> list[str]:
        """
        Quick generate text with top-k guided approach.

        Args:
            *args, **kwargs: Additional arguments for the generate method.

        Returns:
            list[str]: Generated text.
        """
        cargs = min_lemmas_per_synset, max_token_count

        concept_A = self.get_concept(guide[0], *cargs)
        concept_B = self.get_concept(guide[1], *cargs) if len(guide) > 1 else None
        guide = concept_A - concept_B if concept_B else concept_A

        return self.generate_with_topk_guide(sentences, *args, guide=guide, **kwargs)

    def _relative_rank(self, tokens: list[int]) -> torch.Tensor:
        """Compute the frame's matrix rank / num vectors"""
        return Frame(tensor=self._make_word_frames(tokens).mT.float()).relative_rank

    def compute_relative_rank(self, tokens: list[int] | pd.Series, batch_size: int):
        num_batches = max(len(tokens) // batch_size, 2)
        all_batches = np.array_split(tokens, num_batches)
        return torch.cat([self._relative_rank(batch) for batch in tqdm(all_batches)])
