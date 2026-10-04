# Steering a Language Model Toward Several Ideas at Once

A companion to the showcase figures: what the three composition methods
are, what the fluency dial does, where the test concepts come from, and
how to read the scores. Written for a general audience; the technical
definitions live in `metrics-explained.md` and the derivations in
`metrics-derivations.md`.

## What this is about

A language model writes one word at a time. We want it to keep writing
naturally while quietly leaning toward ideas of our choosing — say,
*sadness* and *music* — without those instructions ever appearing in the
prompt. This works by reading the model's own internal representations:
the method needs access to what the model "thinks" after each word, so it
applies to models one can open up, not to services that only return text.
Here we compare three ways of steering toward several ideas at once, and
we measure both whether the ideas made it into the text and whether the
text still reads like normal writing.

## How steering works

At every step, the model proposes a handful of likely next words — the
paper's top-k loop. Each idea we steer toward is represented as a
direction in the model's internal space, built automatically from the
idea's dictionary entry (its WordNet synonyms). For each candidate word,
we check how much the model's internal state would lean toward the target
direction if that word were chosen, and we keep the candidate that leans
most. Repeat until the text is written.

## Three ways to ask for several ideas at once

- **Blending** — mix the paints first. Average the ideas' directions into
  a single combined direction, then steer at that one target throughout.
- **Score voting** — a panel of judges. Keep the ideas separate; every
  candidate word gets one score per idea, the scores are put on a common
  scale so no idea shouts louder than the others, and the candidate with
  the best combined score wins.
- **Taking turns** — alternate the active idea. Steer toward the first
  idea on one step, the second on the next, and so on, cycling through
  the group.

Blending asks for text where the ideas overlap; voting rewards words that
serve any of the ideas well; taking turns guarantees each idea gets
dedicated steps but never optimizes for all of them at once.

## The fluency dial (λ)

Pure steering picks whichever candidate leans most toward the target,
even when the model finds that word unlikely — push hard enough and the
writing bends. The fluency weight λ mixes two preferences: how on-target
a candidate is, and how natural the model itself finds it. At λ = 0 the
dial is off (pure steering); as λ grows, the text returns to what the
model would have written anyway. In practice the recommended operating
range is λ between 0.3 and 0.4 — enough to keep the writing natural at a
modest cost in steering strength. The figures show both ends: dial off,
and λ = 0.4.

## Where the test concepts come from

Candidates were drawn from dictionary entries and each was first tested
*alone*: steer toward just that idea and measure the effect. The pool
keeps the 40 ideas whose solo effect cleared a floor, admitted one at a
time from the strongest down, with two caps along the way: no semantic
category may contribute more than six ideas, and an idea too similar to
one already admitted is skipped. The result is a pool of 40 varied,
individually-verified concepts.

Test groups are then drawn from that pool: 30 pairs, 30 triples, and 30
quadruples, chosen to cover the whole similarity spectrum — from
unrelated ideas to closely related ones — with each concept appearing in
only a few groups. Every group is tested on the same ten everyday
prompts ("Tell me a short story.", "Describe your favorite place.", …).

## How results are scored

- **Expression** — does the text lean toward the idea? Measured from the
  model's internal states after every word, averaged over the text, and
  compared against what the unsteered model would have written for the
  same prompt. Positive means the steering added something; the idea's
  word never has to appear literally.
- **Joint expression** — a team is as strong as its weakest player. For a
  group of ideas, each idea's improvement is measured and the *smallest*
  one is the group's score. Excelling at one idea while ignoring another
  does not count.
- **All-positive rate** — the simple yes-or-no version: in what share of
  runs did *every* idea in the group improve at least a little?
- **Token cosine** — the same "does the text lean toward the idea"
  question, but asked on a fixed 0-to-1 scale based on angles between the
  model's states and the idea's direction. Because the scale is bounded
  and unit-free, scores are comparable across different ideas and
  different models — a fair-comparison companion to expression.
- **Perplexity ratio** — the gibberish alarm. How much more surprising is
  the steered text than the unsteered text, to the model itself? A ratio
  near 1 means the writing stayed natural; runs above 2.5× are flagged as
  disfluent.

One honest caveat for the joint scores: they naturally shrink as the
group grows — a minimum over four ideas is harder than over two — so the
fair reading of the figures is method against method at each group size,
not the downward slope across sizes.

## Reading the figures

- **Expression, dial off.** Blending leads at every group size: its joint
  expression runs from about 0.93 with two ideas down to 0.73 with four,
  with taking-turns second and voting third throughout. The ordering is
  stable; the gaps are modest.
- **Token cosine, dial off.** The same picture on the fair 0-to-1 scale:
  blending stays ahead at every size, so its lead is not an artifact of
  how expression is scaled.
- **Expression and token cosine at λ = 0.4.** Blending's lead survives
  the fluency dial and its cost is small — from 0.93 to about 0.83 with
  two ideas, from 0.73 to about 0.64 with four. Taking turns is the
  method the dial hurts most: with three or four ideas its joint
  expression roughly halves, dropping it below voting. Each idea already
  gets only every third or fourth step; adding a pull back toward the
  model's own preferences leaves too little steering per idea.
- **Perplexity at λ = 0.4.** This is what the dial buys: the gibberish
  alarm reads around 6× with the dial off and drops to roughly 2.5–3.2×
  at λ = 0.4 — taking turns produces the most natural text (right at the
  2.5× alarm line), blending sits slightly above it, voting is the least
  natural. The overall trade: blending keeps the strongest steering;
  taking turns keeps the cleanest prose but pays for it in steering
  strength as groups grow.

Across all settings, every idea in the group improved in roughly half to
seven-tenths of runs (the all-positive rate), highest for blending with
two ideas and declining as groups grow — the expected weakest-link
effect, not a method failure.

## Fine print

All results are from Llama-3.1-8B-Instruct. Every plotted point averages
300 runs (30 groups × 10 prompts). The method reads the model's internal
states, so it requires open access to the model. Group sizes two, three,
and four all rest on 30 groups each; the pair arm was extended from its
original ten to thirty groups using the same selection recipe.

### Data appendix (internal)

Pool: `resources/32_concept_pool_bf16.json` (selection config: delta
floor 0.3, ≤6 per lexname, ρ-dedup 0.75). Groups:
`resources/33_scaling_groups_bf16.json`, `resources/39_quads.json`,
`resources/51_pairs_topup.json`. Runs: `resources/33_scaling3_bf16.jsonl`
(λ=0, n=2,3), `resources/39_scaling4.jsonl` (λ=0, n=4),
`resources/40_greedy_lambda.jsonl` (λ=0.4, all n),
`resources/51_scaling2_topup.jsonl` (both λ, n=2 top-up). Figures and
summary: notebook 52 (`resources/52_*.png`,
`resources/52_composition_summary.csv`).
