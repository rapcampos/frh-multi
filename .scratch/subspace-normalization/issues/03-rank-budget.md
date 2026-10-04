# 03 — Rank budget for blended subspaces

Status: open
Type: research
Blocked by: 02

## Problem

`steering_comparison.py:42` fixes `subspace_rank = 8` for every configuration,
and `ConceptSubspace.blend(guides, rank=8)` collapses however many concepts are
being steered into that same 8-dimensional budget. At `unit_count = 4` each
concept gets roughly two directions; at `unit_count = 1` it gets all eight.

`blend_bases` takes the top-`rank` left singular vectors of the stacked bases,
so overlapping directions reinforce and disjoint ones compete. A fixed budget
means multi-concept steering is systematically starved relative to single-concept
steering — the capacity available per concept falls as concepts are added.

NB68 is consistent with this but does not isolate it: subspace all-improved falls
0.512 → 0.444 → 0.453 → 0.423 across unit counts 1-4, while its truncation rate
climbs 0.231 → 0.277 → 0.297 → 0.312. Those two are confounded — starved rank
and early stopping would both depress the score.

## Question

Does scaling rank with concept count recover multi-concept performance, and is
the NB68 unit-count decay a capacity artifact rather than a property of subspace
composition?

## Method

Sweep `subspace_rank` against `unit_count`. At minimum compare the fixed budget
(8) against a per-concept budget (`base_rank × unit_count`), holding everything
else at NB68 settings. Run after `02` lands so truncation is no longer confounding
the result; if `02` is refuted or deferred, report truncation rate alongside every
cell so the two effects stay separable.

Per-concept ranks are already recorded in the manifest provenance
(`construction[].subspace_rank`), so the realized rank per concept is known
without re-deriving it.

## Answer

_pending_
