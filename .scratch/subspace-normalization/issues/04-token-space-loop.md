# 04 — Keep the generation loop in token space

Status: prototyped and validated (not applied to project source)
Type: task

## Result

Prototyped in `../approach_a.py` with issue `05`, measured by
`../smoke_approach_a.py`: 4 configs (unit counts 1-4) x 8 prompts x lambda in
{0, 0.4}, WordNet source, 64 runs per loop.

| loop | lambda | mean words | short-run rate | identical to baseline |
|---|---|---|---|---|
| current | 0.0 | 15.3 | 0.19 | 0.12 |
| approach A | 0.0 | **17.9** | **0.06** | **0.00** |
| current | 0.4 | 14.0 | 0.25 | 0.12 |
| approach A | 0.4 | **17.0** | **0.09** | **0.03** |

Generated tokens under approach A: median 25 of 25, mean 24.4 (lambda=0). 15.6%
of runs now terminate legitimately, which the old loop could not detect at all.

Crucially it does **not** buy this by switching steering off — the trap
normalization fell into, where half its outputs were byte-identical to the greedy
baseline. Approach A moves that to 0.00, and at lambda=0.4 it diverges *more*
from baseline than the current loop (0.44 vs 0.62 shared leading words) while
also producing longer output. Better on both axes.

All five residual short runs are accounted for: three legitimately emitted
`<|eot_id|>` and stopped (e.g. `'learning new facts about the human experience.'`),
and two ran the full 25 tokens but score few *words* because the text is
degenerate (`"a topic we haven\'\'\'\'..."`). No residual truncation defect.

Text quality at lambda=0 is still poor. That is the steering objective, not the
loop — and it is now honest output rather than round-trip wreckage.

## Caveats

Small: 128 runs, one concept source, four configs. Unit count 4 still shows a
0.188 short rate against 0.000-0.062 at lower counts, which warrants a look in a
full battery.

## The defect

Issue `01` measured it: the subspace loop keeps only ~0.53 tokens per step against
frame blending's 1.005, and sometimes goes backwards (min growth -1).

Decoding `d: T → S` is many-to-one and encoding `e: S → T` picks one preferred
segmentation, so `e∘d` is a **projection onto the canonical subset** `T_c`. For
`t ∈ T_c` it is the identity; for `t ∉ T_c` it generally shortens the sequence.

Both loops rebuild state every step with `make_input(decode(tokens), padding=True)`
(`subspace/representation.py:291`, `frames/representations/frame.py:732`), so the
recurrence is not

    s_{t+1} = s_t ⊕ x        (append, as intended)

but

    s_{t+1} = e(d(s_t ⊕ x))  (append, then re-canonicalize)

Re-canonicalization can undo the append. There is no invariant guaranteeing
progress. The model distributes over all of `T`, not `T_c`, so nothing prevents a
proposed token from leaving the canonical set.

Frame steering escapes by accident: natural word-level continuations are
canonically segmented. Subspace steering's `‖Bᵀh‖` favours fragmentary subword
tokens, which are not.

## Second consequence

Even when length survives, the score is computed on hidden states of the
**re-canonicalized** sequence, so the model may be scoring a different
segmentation than the one selected. Alignment is evaluated on a perturbed input
and step `t+1` conditions on a state the search never chose. Any fix that only
lowers the rate at which the projection bites leaves this untouched.

## The fix

Maintain `(ids, attention_mask)` across steps; decode exactly once at the end.
The recurrence becomes a true append and the invariant "length grows by exactly 1
per step" holds by construction. Requires handling batch padding directly, with
left-padding so appends land at the true sequence end.

Ship with the invariant asserted per step (see below) so a regression is loud.

## Why it is deferred, and what would promote it

Deferred in favour of issue `02` (normalization), which is symptomatic — it
reduces how often the projection fires without removing it, and does not address
the mis-scoring above. If `02`'s validation shows the lossy-step rate stays
materially above zero, this becomes the fix.

## Frame path

The frame path sits on the fixed-point set — 0 lossy steps in 192 observations —
so on runs that never leave `T_c`, removing the round trip is a no-op. That is
evidence, not proof, and `_generate_with_topk_guide` is the original paper
function. Ship behind a flag defaulting to current behaviour, and run the
assertion on the frame path to learn empirically whether it ever leaves `T_c`.
If a real battery never trips it, promote the fix to default with evidence.

## Cheap guard, worth having regardless

Assert per step that the non-pad length grew by exactly 1; on violation fall back
to the next-best candidate or raise. This does not fix the recurrence, but it
converts silent corruption into a visible failure. Nothing asserted this
invariant, which is why the defect survived a 4,736-run battery.

## Impact on existing results

Under this fix the subspace arm is a different generator, so NB68's subspace
numbers do not carry over — both the fluency penalty and the cosine advantage
were measured on truncated, mis-scored text. Issue `03` (rank budget) is also
confounded until this lands: its unit-count axis moves together with the
truncation rate (0.231 → 0.312 across unit counts 1-4).
