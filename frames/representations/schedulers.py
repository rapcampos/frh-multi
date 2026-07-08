"""Concept schedulers (Family 3) — time-composition for guided generation.

Family 3 keeps scoring single-concept at every step and varies WHICH concept
is active over time. Schedulers plug into the same `scorer` seam as the
Family 2 aggregators, but are stateful callables implementing an optional
protocol that the generation loops honor:

- `reset(n)` — called once at the start of a generation call (n = batch size)
- `observe(tokens, n)` — called each step with the current candidate token
  ids, BEFORE scoring; lets a scheduler react to the generated text
- `__call__(projections, weights)` — selects the active concept and returns
  its projections; the loop's `weights` tensor is IGNORED (Family 3 semantics:
  one concept at a time; Stochastic takes probabilities at construction)

Use a fresh scheduler instance per experiment, or rely on `reset` (invoked by
the loop) to clear state between generation calls.

Niche but important: concepts that cannot co-occur in one word can co-occur
in a passage — schedulers are the control condition that distinguishes method
failure from linguistic impossibility.
"""

import torch


class ConceptScheduler:
    """Base scheduler: subclasses set the active concept index per step."""

    def reset(self, n: int) -> None:  # noqa: B027 - optional hook, default no-op
        """Called by the generation loop before the first step."""

    def _select(self, projections: torch.Tensor, index: int) -> torch.Tensor:
        """Return the projections of a single concept (same index for all rows)."""
        return projections[..., index]


class RoundRobin(ConceptScheduler):
    """F3.a — cycle deterministically through the concepts, one per step."""

    def __init__(self):
        self.step = 0

    def reset(self, n: int) -> None:
        self.step = 0

    def __call__(
        self, projections: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        index = self.step % projections.size(-1)
        self.step += 1
        return self._select(projections, index)


class Stochastic(ConceptScheduler):
    """F3.b — sample the active concept per step, weights as probabilities.

    Seeded for reproducibility: `reset` re-seeds, so every generation call
    sees the same schedule.
    """

    def __init__(self, probabilities: list[float], seed: int = 0):
        self.probabilities = torch.tensor(probabilities, dtype=torch.float)
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

    def reset(self, n: int) -> None:
        self.generator = torch.Generator().manual_seed(self.seed)

    def __call__(
        self, projections: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        index = int(torch.multinomial(self.probabilities, 1, generator=self.generator))
        return self._select(projections, index)


class SentenceBoundary(ConceptScheduler):
    """F3.c — advance to the next concept at sentence boundaries.

    Detects boundaries with a punctuation heuristic on each input's
    representative beam (its first row = the greedy child of the selected
    beam). State is kept per input, so different inputs can be on different
    concepts at the same step.
    """

    def __init__(self, tokenizer, boundary_chars: str = ".!?\n"):
        self.tokenizer = tokenizer
        self.boundary_chars = set(boundary_chars)
        self.active = torch.zeros(0, dtype=torch.long)

    def reset(self, n: int) -> None:
        self.active = torch.zeros(n, dtype=torch.long)

    def observe(self, tokens: torch.Tensor, n: int) -> None:
        """Advance an input's counter when its newest token ends a sentence."""
        m = tokens.size(0) // n
        newest = tokens.reshape(n, m, -1)[:, 0, -1]
        pieces = self.tokenizer.batch_decode(newest.unsqueeze(-1))
        for i, piece in enumerate(pieces):
            if any(char in self.boundary_chars for char in piece):
                self.active[i] += 1

    def __call__(
        self, projections: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        n_concepts = projections.size(-1)
        rows = projections.size(0)
        m = rows // self.active.size(0)
        index = (self.active % n_concepts).repeat_interleave(m)
        onehot = torch.nn.functional.one_hot(index, n_concepts)
        onehot = onehot.view(rows, 1, 1, n_concepts)
        onehot = onehot.to(dtype=projections.dtype, device=projections.device)
        return (projections * onehot).sum(-1)
