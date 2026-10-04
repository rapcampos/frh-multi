# Qualitative Gallery — Multi-Concept Steering under Top-k Concept-Guided Decoding

Curated qualitative examples of composition (n = 1 → 4), greedy Top-k decoding
(k = 4, 32 steps), unquantized Llama-3.1-8B (bf16). **Each block uses a distinct
nested concept quad**, matched to its prompt's register, steered via the simple
blend guide. n = 1 is a single-concept reference; the focus is the n = 2 → 4
progression. Per-concept **Δ** is the steered-minus-baseline mean
frame-correlation (expression metric); **✓** marks levels where *every* concept
improved (AND-success).

**Concept provenance.** The gallery opens with blocks built from the most
interpretable members of the **bf16-validated pool** (sorrow, fury, …) — the same
pool all quantitative claims rest on. The remaining blocks use an **illustration
set** chosen for human readability rather than validated steerability (sadness,
music, king, child, …); they demonstrate what composition looks like, not how
often it succeeds, and each is annotated with its concepts' validated singles Δ
where available so failure blocks can be attributed (weak guide vs failed
composition). Quantitative claims never rest on illustration-set blocks.

*Pool-native blocks P1–P2 below are from notebook 47
(`resources/47_qualitative_pool.jsonl`); the numbered blocks that follow are the
illustration set, each annotated with its concepts' validated singles Δ from the
nb32 slate.*

---

## Block P1 — "Write the opening of a novel." *(pool-native lead block; report §5.5)*

*Quad: sorrow → fury → flood → passing — all validated pool members — **all-improved through n = 4**.*
*Validated singles Δ (nb32 pool): sorrow 1.21 · fury 1.31 · flood 1.25 · passing 1.42.*

> **Unsteered baseline.** Here's the opening of a novel: **The Shadow in the Swamp** The cypress trees towered above me like skeletal sentinels, their knotted branches

**n = 1: sorrow** — Δ [sorrow +1.88] — ✓ all improved
> As I emerged through a swirls veil that hung like mist in front me, a gust wind slammed shut behind me with force. It left me stumbling forward,

**n = 2: sorrow + fury** — Δ [sorrow +2.69, fury +2.24] — ✓ all improved
> As I emerged through a swirls veil into morning light, I knew I wasn' home any more - or, for that fact - I wasn home ever again

**n = 3: sorrow + fury + flood** — Δ [sorrow +1.90, fury +1.28, flood +1.29] — ✓ all improved
> As she emerged through a swirls veil into morning light that clung stubbornly close to earth like damp skin to flesh she knew she could delay it for only

**n = 4: sorrow + fury + flood + passing** — Δ [sorrow +0.96, fury +0.92, flood +0.74, passing +0.60] — ✓ all improved
> As she emerged through a swirls veil into morning light, Aislynd realized she wasn not alone anymore - at last - not alone with only shadows and

**λ contrast (n = 2, sorrow + fury):**

**λ = 0** — Δ [sorrow +2.69, fury +2.24] — ✓ all improved
> As I emerged through a swirls veil into morning light, I knew I wasn' home any more - or, for that fact - I wasn home ever again

**λ = 0.4** — Δ [sorrow +0.84, fury +0.57] — ✓ all improved
> Here's the start of a novel I came-up-with on-the-spot: As I emerged from beneath the creaked awning, I squinted into

---

## Block P2 — "The traveler arrived at" *(pool-native)*

*Quad: country → town → dwelling → respite — all validated pool members — quad-dependent: holds at n = 1, 3; partial at n = 2, 4.*
*Validated singles Δ (nb32 pool): country 1.32 · town 1.39 · dwelling 1.37 · respite 1.47.*

> **Unsteered baseline.** The traveler arrived at the old inn on a stormy night, seeking refuge from the torrential rain. The sign creaked in the wind, reading "Moonstone Inn" in faded

**n = 1: country** — Δ [country +0.31] — ✓ all improved
> The traveler arrived at dusk. She stepped out onto worn wooden planked dock into mist. It was cool on bare feet. She looked out across water. It stretched far out. She

**n = 2: country + town** — Δ [country +0.32, town -0.03] — ✗ partial
> The traveler arrived at midnight. She walked through an old wooden station house with worn stone floors. Outside was nothing – just an exodus into nothing. No sign, not so far as

**n = 3: country + town + dwelling** — Δ [country +0.36, town +0.07, dwelling +0.54] — ✓ all improved
> The traveler arrived at the small, coastal tavern just after sundown. He pushed through swinging double glass pan es into warm light. Inside it smelled heavily, as it did every tavern,

**n = 4: country + town + dwelling + respite** — Δ [country +0.21, town -0.06, dwelling +0.08, respite -0.01] — ✗ partial
> The traveler arrived at the small, coastal tavern just after night fall. He pushed open heavy wooden swing door with creak, releasing warm light into cool night air. Inside was dimly

The gallery is ordered best-first and shows the full **range** of composition outcomes
— from clean layering through n = 3 to composition failure — rather than only clean
wins. Prose is verbatim model output (minor decoding artifacts intact).

*Source: `resources/qualitative_n1to4.json`, `resources/qualitative_gallery_varied.json`.*

---

## Block 1 — "Write the opening of a novel."

*Quad: sadness → music → snow → war. melancholy → music → winter → conflict — all-improved through **n = 3**.*
*Validated singles Δ (nb32 slate): sadness 0.85 · music 0.56 · snow 0.48 · war 0.83 — all below the pool floor (≈ 1.17).*

> **Unsteered baseline.** The city of Ashwood was a place where the past and present collided, where the crumbling facades of ancient buildings seemed to lean in, as if sharing secrets with the pedestrians

**n = 1: sadness** *(single-concept reference)* — Δ [sadness +1.04] — ✓ all improved
> The mist-shrooded alleé unwrapped like an unfurlng scroll of damp silk along either flank of Boulevard St-Lucien in Montparnesse's

**n = 2: sadness + music** — Δ [sadness +0.42, music +0.62] — ✓ all improved
> The rain poured like liquid night, druming off rooftles as if trying desperatelyto drown out its thoughts in its relentless patter. It was as much of sound as it

**n = 3: sadness + music + snow** — Δ [sadness +0.71, music +0.73, snow +0.46] — ✓ all improved
> The rain had a way of erasng all memory of what had happened the moment before its rhythym began drummin' out against pavement like fingers on keys of a

**n = 4: sadness + music + snow + war** — Δ [sadness +0.54, music -0.38, snow +0.07, war -0.23] — ✗ partial
> The rain-soaken alleys of Asher Street had seen their share over the past few years - broken bottles littering puddle-slick pavings like jaguar's

---

## Block 2 — "The traveler arrived at"

*Quad: storm → water → fire → war. elemental journey into conflict — all-improved through **n = 3**.*
*Validated singles Δ (nb32 slate): storm 1.01 · water 0.94 · fire 0.77 · war 0.83 — all below the pool floor (≈ 1.17).*

> **Unsteered baseline.** The traveler arrived at the small village, exhausted from the long journey. The sun was setting, casting a warm orange glow over the thatched roofs of the cottages. The air was filled

**n = 1: storm** *(single-concept reference)* — Δ [storm +0.53] — ✓ all improved
> The traveler arrived at his final stop after a long, arueling train of events (literally - it started as just that - an aruel train). As he stumbled out of the

**n = 2: storm + water** — Δ [storm +1.19, water +0.03] — ✓ all improved
> The traveler arrived at his final stop after a long, arduosous train journey. As he stepped out into fresh mountainair and took in his new suroundings, he felt

**n = 3: storm + water + fire** — Δ [storm +1.51, water +0.03, fire +1.09] — ✓ all improved
> The traveler arrived at her destination after a long, arueling trainride. As she stepped out into fresh mountainair she breathed deeply of it, letting it clear out all thoughts from her

**n = 4: storm + water + fire + war** — Δ [storm -0.04, water +0.07, fire +0.36, war -0.55] — ✗ partial
> The traveler arrived at a small, seaside café. It looked old-fashioned from a by gone time. There wasn‘ t anyone else in sight, except for an elderly man sitting by the

---

## Block 3 — "In the old house on the hill,"

*Quad: sorrow → love → storm → fire. haunted grief and old passion — all-improved through **n = 3**.*
*Validated singles Δ (nb32 slate): sorrow 1.21 (**pool member**) · love 0.56 · storm 1.01 · fire 0.77.*

> **Unsteered baseline.** In the old house on the hill, where the wind whispered secrets through the creaking floorboards, a young woman named Emilia lived a life shrouded in mystery. Her eyes, like two dark

**n = 1: sorrow** *(single-concept reference)* — Δ [sorrow +0.55] — ✓ all improved
> In the old house on the hill, surroundedby overgrowth of vines as tangled as memories themselves and windows cloudy with years worth dusting away. It had once belongedto Elwyn, an enigmatic woman

**n = 2: sorrow + love** — Δ [sorrow -0.31, love -0.46] — ✗ partial
> In the old house on the hill, surrounded only on three side's, where once stood tall trees that shaded children from sun's warm kiss. A young boy lived there with no family of kin. His

**n = 3: sorrow + love + storm** — Δ [sorrow +0.47, love +0.38, storm +1.10] — ✓ all improved
> In the old house on the hill, surroundedby overgrowth of weeds as thick and tangled as memories of past loves gone wrong... (Would you want me add anything else, suchas characters, settings,

**n = 4: sorrow + love + storm + fire** — Δ [sorrow +0.77, love +0.30, storm +1.09, fire -0.02] — ✗ partial
> In the old house on the hill, surroundedby overgrowth of weeds as thick and tangled as memories of past loves gone wrong... (Would you prefer me keep telling the tail as is, stop here,

---

## Block 4 — "It was the night"

*Quad: sorrow → king → queen → war. a royal tragedy — all-improved through **n = 1**.*
*Validated singles Δ (nb32 slate): sorrow 1.21 (**pool member**) · king 0.74 · queen 0.73 · war 0.83.*

> **Unsteered baseline.** It was the night the stars aligned in a way that only happened once a decade, and the small town of Willow Creek was bathed in an otherworldly glow.

**n = 1: sorrow** *(single-concept reference)* — Δ [sorrow +1.13] — ✓ all improved
> It was the night that would shroude everything in mystery forevermore: October thirty firstst at exactly midnight, under an inauspirous moon. What happens after that night?

**n = 2: sorrow + king** — Δ [sorrow -0.22, king -0.88] — ✗ partial
> It was the night that would shroude everything I thought I understood with a thick layer uncertainty. What happens after the first few paragraphs of your opening story line? Would love for you

**n = 3: sorrow + king + queen** — Δ [sorrow -0.11, king -0.71, queen -0.59] — ✗ partial
> It was the night that would shroude everything I thought I understood with a thick layer uncertainty. What happens after the first few paragraphs of a short or novel, and how would you

**n = 4: sorrow + king + queen + war** — Δ [sorrow -0.21, king +0.21, queen +0.42, war -0.05] — ✗ partial
> It was the night the storm of a thousand midlife crisses finally rolled ashORE, threatening not just to upEND but utterly DISINteGraTed, a seemingly idyl

---

## Block 5 — "He had walked for days when"

*Quad: storm → water → fire → animal. wilderness — all-improved through **n = 1**.*
*Validated singles Δ (nb32 slate): storm 1.01 · water 0.94 · fire 0.77 · animal 0.99 — all below the pool floor (≈ 1.17).*

> **Unsteered baseline.** He had walked for days when he stumbled upon the old, abandoned town. The streets were empty, the buildings crumbling, and the silence was oppressive. He had been searching for a place to call home

**n = 1: storm** *(single-concept reference)* — Δ [storm +1.29] — ✓ all improved
> He had walked for days when the rain started, a relentless down pour, drumbeat heavy, like someone was trying out for drum major of heaven's band section. It didn 't stop,

**n = 2: storm + water** — Δ [storm +0.15, water -0.49] — ✗ partial
> He had walked for days when he stumbled upon it - A worn and weather-wreckage of what used to stand. He recognized it immediately - A place from which he'd fled years before.

**n = 3: storm + water + fire** — Δ [storm +2.09, water +0.43, fire -0.10] — ✗ partial
> He had walked for days when the rain began. At fist it was just an annoyance - an added layer of chill in an unseasonally cool spring - but as the droplet-laced wind

**n = 4: storm + water + fire + animal** — Δ [storm +0.06, water -0.80, fire -0.48, animal -0.54] — ✗ partial
> He had walked for days when he stumbled upon it - A worn and weather-wreckage of what used to stand. He recognized it immediately - A once-magnanimious latticework

---

## Block 6 — "She opened the door and"

*Quad: child → mother → father → girl. a family scene — **composition failure** (breaks at n = 1).*
*Validated singles Δ (nb32 slate): child 1.01 · mother 0.78 · father 0.82 · girl 1.09 — all below the pool floor (≈ 1.17).*

> **Unsteered baseline.** She opened the door and stepped out into the cool evening air. The sky was a deep shade of indigo, with stars beginning to twinkle like diamonds scattered across the fabric of the universe.

**n = 1: child** *(single-concept reference)* — Δ [child -1.00] — ✗ partial
> She opened the door and a gust swept in, rustle through leaves on a nearby planer, carrying with them a hint scent. She closed eyes for just moment, breathed deep.

**n = 2: child + mother** — Δ [child +0.20, mother -0.55] — ✗ partial
> She opened the door and was envelopped immediately by a cacphony. Cars driving, people yelling, sireins in far distance. But amidst all of it was her mother. Her mom stood

**n = 3: child + mother + father** — Δ [child -0.78, mother -0.69, father -0.49] — ✗ partial
> She opened the door and stepped out of her car. Her heels crumched on to gravel. She took off them. She walked across to him.   "You came," she said.

**n = 4: child + mother + father + girl** — Δ [child -0.55, mother -1.39, father -1.70, girl -0.43] — ✗ partial
> She opened the door and a slant ray from a street lamp illuminated him for just long moment as he hesitates on her step. He wore a denim wind-break jacket over a plaid

---

## Reading the gallery

- **Pool-grade guides compose deeper.** Block P1 — the only quad whose every
  constituent clears the pool floor — is also the only quad that holds
  all-improved through n = 4. Block P2 shows quad/prompt dependence persists
  even pool-native (partial at n = 2 and 4). The P1 λ contrast is §5.6 in
  prose: ✓ retained at λ = 0.4 with visibly more conventional phrasing at
  roughly a third of the Δ.
- **Composition is visible when it works.** Block 1: melancholy imagery → rain
  "drumming" (music) → "fingers on keys" rhythm. Block 2: storm and fire layer onto
  the traveler's arrival with rising Δ (storm 0.53 → 1.51).
- **All-improved breaks at n = 4** wherever composition holds — the fourth concept is
  the one that most often cannot be added on top of three, mirroring the quantitative
  all-improved decline from n = 3 to n = 4.
- **Failure attribution: guide quality vs composition.** The validated-Δ
  annotations let the failure blocks be attributed. The family quad (Block 6) is
  a *guide-quality* failure: every constituent sits below the pool floor
  (slate Δ 0.78–1.09 < ≈ 1.17), so its collapse mirrors the §5.4 finding, not a
  composition limit. The royal quad (Block 4) is the cleaner *composition*
  exhibit: *sorrow* is pool-grade (Δ 1.21) and steers strongly alone, yet
  below-floor king/queen suppress it when composed. Composition outcomes are
  quad- and prompt-dependent — an honest illustration of the ~62–74%
  both-improved rate rather than a universal guarantee.
- **Perception vs metric.** Several partial blocks read on-theme while the metric
  denies improvement — a reminder that the expression Δ, not surface impression, is
  the success criterion.
- **Tier note.** Greedy Top-k throughout — the report's decoder; survivor-rule
  variants (beam search) are a future-work direction (report §7). A
  repeated-strong-chain version (`resources/qualitative_gallery_strong.json`)
  reaches n = 3 on more prompts by avoiding per-quad weak links.

