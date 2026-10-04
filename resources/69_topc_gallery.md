# NB69 qualitative gallery — WordNet vs lemma-matched custom vs full custom

Side-by-side continuations of bf16 Llama-3.1-8B-Instruct under FRH frame blending and subspace steering, for three concept sources built from the same 16 concepts:

- **wordnet** — the synset's WordNet member lemmas (eight languages; *c* per concept).
- **custom_top_c** — the first *c* words of the consensus-ranked custom list.
- **custom** — the consensus top 100 custom words.

Every run shares NB68's unsteered baseline for its prompt. **Δcos** is the signed mean cosine change on the run's own reference (its source × frame for FRH, × subspace for subspace steering; repelled concepts sign-flipped, so positive is always desirable). **ΔPPL** is continuation-only perplexity change versus the baseline (lower is more fluent). Prompt p07 is excluded (empty baseline).

**Prompt selection.** Unless marked *strongest*, each example uses the *median* prompt: the seven prompts are ranked by mean Δcos over the six method × source cells, and the middle one is shown, so examples are typical rather than cherry-picked. *Strongest* examples show the top prompt instead.

## Caveat: degenerate outputs inflate Δcos

A run is marked **⚠ degenerate** when it stops before 12 of its 25 tokens or repeats a 1–4 character unit six or more times (e.g. a row of asterisks). Subspace steering produces many early stops, likely by pushing toward end-of-sequence; FRH almost never does. Short and repetitive texts score high on the token-mean cosine, and PPL does not reliably penalise repetition, so part of subspace steering's aggregate lead comes from degenerate outputs. Numbers below are own-view signed Δcos over the analysed prompts.

| method | source | λ | degenerate | Δcos (all) | Δcos (excl. degenerate) |
|---|---|---:|---:|---:|---:|
| frh_blend | wordnet | 0 | 3% | +0.0043 | +0.0039 |
| frh_blend | wordnet | 0.4 | 3% | +0.0017 | +0.0015 |
| frh_blend | custom_top_c | 0 | 2% | +0.0049 | +0.0049 |
| frh_blend | custom_top_c | 0.4 | 2% | +0.0031 | +0.0030 |
| frh_blend | custom | 0 | 1% | +0.0097 | +0.0096 |
| frh_blend | custom | 0.4 | 1% | +0.0074 | +0.0072 |
| subspace | wordnet | 0 | 18% | +0.0104 | +0.0088 |
| subspace | wordnet | 0.4 | 17% | +0.0085 | +0.0060 |
| subspace | custom_top_c | 0 | 23% | +0.0131 | +0.0101 |
| subspace | custom_top_c | 0.4 | 18% | +0.0102 | +0.0064 |
| subspace | custom | 0 | 22% | +0.0184 | +0.0151 |
| subspace | custom | 0.4 | 23% | +0.0163 | +0.0117 |

## Concept sources

| concept | c | WordNet lemmas | custom top-c | custom top 100 (first 8) |
|---|---:|---|---|---|
| agitation | 2 | agitación, agitation | restlessness, turmoil | restlessness, turmoil, unrest, anxiety, nervousness, tension, distress, perturbation, … |
| despair | 4 | desesperación, despair, désespoir, désespérer | hopelessness, anguish, desolation, misery | hopelessness, anguish, desolation, misery, grief, sorrow, despondency, emptiness, … |
| calmness | 6 | calma, calmness, sosiego, tranquilidad, tranquilidade, ความสงบ | peace, serenity, tranquility, stillness, composure, relaxation | peace, serenity, tranquility, stillness, composure, relaxation, quietude, ease, … |
| sadness | 8 | malheur, melancolía, pesar, sadness, traurigkeit, tristesse, tristeza, unhappiness | sorrow, grief, melancholy, despair, misery, anguish, gloom, loneliness | sorrow, grief, melancholy, despair, misery, anguish, gloom, loneliness, … |
| protect | 9 | assurer, defender, protect, proteger, proteggere, protéger, préserver, sauvegarder, schützen | shield, guard, defend, safeguard, secure, shelter, preserve, cover, armor | shield, guard, defend, safeguard, secure, shelter, preserve, cover, … |
| harm | 10 | afligir, danificar, dano, détériorer, fazer mal, ferir, harm, lesar, machucar, schaden | injury, damage, pain, suffering, hurt, abuse, violence, trauma, wound, cruelty | injury, damage, pain, suffering, hurt, abuse, violence, trauma, … |
| hope | 11 | esperanza, esperança, espoir, espérance, espérer, hoffnung, hope, speranza, आशा, उम्मीद, ความหวัง | optimism, faith, trust, aspiration, possibility, expectation, promise, anticipation, dawn, belief, dream | optimism, faith, trust, aspiration, possibility, expectation, promise, anticipation, … |
| joy | 12 | alegria, alegría, allégresse, freude, ilusión, joie, joy, joya, joyfulness, joyousness, हर्ष, ความปีติ | happiness, delight, elation, bliss, jubilation, gladness, glee, exuberance, euphoria, ecstasy, merriment, laughter | happiness, delight, elation, bliss, jubilation, gladness, glee, exuberance, … |
| hostility | 12 | abominação, aversão, enmity, hostilidade, hostility, hostilité, ill will, inimitié, inimizade, repulsa, ressentimento, ódio | aggression, animosity, antagonism, hatred, enmity, malice, resentment, spite, anger, belligerence, bitterness, contempt | aggression, animosity, antagonism, hatred, enmity, malice, resentment, spite, … |
| trust | 13 | confiance, confianza, confiança, faith, fe, foi, fé, trust, vertrauen, भरोसा, विश्वास, ความมั่นใจ, ศรัทธา | faith, honesty, confidence, integrity, reliability, dependability, loyalty, assurance, belief, security, credibility, sincerity, transparency | faith, honesty, confidence, integrity, reliability, dependability, loyalty, assurance, … |
| suspicion | 14 | desconfianza, desconfiança, destruction, distrust, défiance, misgiving, misstrauen, mistrust, méfiance, receio, recelo, soupçon, suspicion, ความสงสัย | doubt, distrust, mistrust, misgiving, unease, apprehension, hunch, paranoia, vigilance, skepticism, wariness, uncertainty, caution, alertness | doubt, distrust, mistrust, misgiving, unease, apprehension, hunch, paranoia, … |
| woman | 15 | adult female, femme, frau, mujer, mulher, woman, औरत, नारी, महिला, स्त्री, นารี, ผู้หญิง, สตรี, อิตถี, อิสตรี | female, strength, daughter, femininity, mother, sister, resilience, grandmother, aunt, niece, sisterhood, identity, equality, motherhood, courage | female, strength, daughter, femininity, mother, sister, resilience, grandmother, … |
| courage | 18 | abattage, braveness, bravery, bravoure, coragem, coraje, cour, courage, courageousness, denodo, intrepidezza, mut, temerarietà, ânimo, साहस, ความกล้า, ความกล้าหาญ, ความอาจหาญ | bravery, valor, boldness, fearlessness, daring, grit, fortitude, resolve, nerve, heroism, tenacity, mettle, determination, backbone, audacity, gallantry, intrepidity, resilience | bravery, valor, boldness, fearlessness, daring, grit, fortitude, resolve, … |
| man | 19 | adult male, herr, hombre, homem, homme, macho, man, mann, marido, maschio, uomo, varão, varón, आदमी, नर, पुरुष, ชาย, บุรุษ, ผู้ชาย | male, father, son, husband, brother, human, adult, strength, person, individual, humanity, masculinity, gentleman, boy, guy, fellow, beard, provider, protector | male, father, son, husband, brother, human, adult, strength, … |
| fear | 22 | angst, consternação, crainte, effroi, espanto, fear, fearfulness, fright, medo, miedo, paura, peur, schreck, schrecken, spavento, susto, temor, terror, timore, épouvante, डर, ความกลัว | terror, dread, panic, anxiety, horror, fright, alarm, phobia, apprehension, worry, unease, trepidation, foreboding, danger, threat, distress, death, darkness, paranoia, nightmare, suspicion, uncertainty | terror, dread, panic, anxiety, horror, fright, alarm, phobia, … |
| affection | 28 | afección, afectividad, afecto, afecção, afeição, afetividade, affection, affectionateness, amor, amour, apego, carinho, cariño, cur, cœur, fondness, heart, impressão moral, philia, simpatía, tendance, tenderness, tendresse, ternura, warmheartedness, warmness, ความรัก, ความรักใคร่ | love, warmth, fondness, tenderness, devotion, attachment, closeness, intimacy, adoration, kindness, sweetness, compassion, endearment, comfort, gentleness, embrace, caring, sympathy, trust, connection, care, loyalty, empathy, caress, touch, friendship, companionship, hug | love, warmth, fondness, tenderness, devotion, attachment, closeness, intimacy, … |

## Single concepts · λ=0.4 · median prompt

### joy

`n1-joy` · steer toward **joy** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... wait, as a conversational AI, I don't have the ability to interact with the physical world. I don't | -0.0108 | -19.7 |
| wordnet | subspace | not actually met in real life. As I'm just a text-based conversational interface, I don't have have the ability | +0.0049 | -14.1 |
| custom_top_c | frh_blend | ... (just a moment) Wait! I'm a largelanguagemodel, I don't have personal experiences or meet people in | +0.0101 | +14.0 |
| custom_top_c | subspace ⚠ | a fictional character. I | +0.0369 | +858.6 |
| custom | frh_blend | a fictional AI-generated individual for the sake of creative response, so the following description should serve as hypothetical. This person's name | +0.0055 | +55.2 |
| custom | subspace ⚠ | a fictional AI character. I | +0.0404 | +1035.6 |

### sadness

`n1-sadness` · steer toward **sadness** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | not something that can be directly observed, as I don't have real-time access to current information. However, I can provide | +0.0130 | +7.7 |
| wordnet | subspace ⚠ | not something I can observe. I | +0.0275 | +192.4 |
| custom_top_c | frh_blend | not described because this is the beginning of a text and there is not enough context. However, if you want, I can | +0.0102 | +11.7 |
| custom_top_c | subspace | quite variable depending on one&#39;s location. In terms of temperature, there isn&#39;t much to report globally as | +0.0050 | +21.9 |
| custom | frh_blend | not described, as I am unable provide real-time information about current weather. I can,however, provide information about typical weather | +0.0122 | +32.2 |
| custom | subspace | not something specific, it depends on where you are, as it can differ greatly depending on location. | +0.0071 | +23.4 |

### hope

`n1-hope` · steer toward **hope** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... (I don't have the ability to meet people in real life, I exist solely as a computer program running on servers | -0.0024 | -11.8 |
| wordnet | subspace | a fictional AI-generated character. Would you like me to create a real person for you? | -0.0065 | +13.7 |
| custom_top_c | frh_blend | an interesting one, but I don't actually "meet" people in real life. I exist solely in a digital format, | +0.0002 | -11.4 |
| custom_top_c | subspace | an artificial construct. I do not possess the capacity or the means to interact in real time. I do not have the ability | +0.0136 | -5.8 |
| custom | frh_blend | a virtual AI assistant. I don't have the ability, capability, or the power, nor the physical presence, nor the | +0.0143 | -5.4 |
| custom | subspace | a 30-something-year-olddescribed to have an energetic and optimistic outlook towards her daily activities and a passion | +0.0254 | +18.4 |

### despair

`n1-despair` · steer toward **despair** · λ=0.4 · p03: *What do you like to do on weekends?* → “I like …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | to help users like you with information and tasks. I don't have personal preferences or experiences, but I can suggest some popular | | |
| wordnet | frh_blend | to process information 24/7, including on weekends, and I don't require rest or relaxation. I am here to | -0.0074 | +16.8 |
| wordnet | subspace | to help you, of course, 7 days/week, all day, and at night, too, if needed. | -0.0017 | +11.0 |
| custom_top_c | frh_blend | to help answer questions like yours,  but I don't have personal experiences, emotions, or preferences, so I don't | +0.0014 | +7.6 |
| custom_top_c | subspace | to help you, but I'm a largelanguage mode  I don't have have feelings, emotions, experiences, personal preferences | +0.0146 | +46.9 |
| custom | frh_blend | to process information 24 hours, 7 days, and respond accordingly, I don't experience a "weekend", but | +0.0164 | +44.0 |
| custom | subspace ⚠ | to process information. | +0.0298 | +1505.1 |

### affection

`n1-affection` · steer toward **affection** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... wait. As a conversational AI, I don't have have the ability to interact with people in the real world, | -0.0047 | -15.2 |
| wordnet | subspace | a 30-soldier, named Alex. I don't actually "meet people" since I don 't interact with | +0.0091 | +58.7 |
| custom_top_c | frh_blend | a fictional AI-generated individual for demonstration, so there's no personal account of a real-life encounter. If you'd like, | +0.0054 | +50.5 |
| custom_top_c | subspace ⚠ | a fictional character. I | +0.0450 | +858.6 |
| custom | frh_blend | a fictional AI-generated individual for demonstration, so there's no personal account of a real-life encounter. I don' have personal | +0.0137 | +99.4 |
| custom | subspace | a fictional AI-generated individual, so there's limited detail about their personal experiences, relationships, and daily life. However | +0.0091 | +19.4 |

### hostility

`n1-hostility` · steer toward **hostility** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | was alive at night, pulsating with energy and vibrancy. Towering skyscrapers stood tall, illuminated by twinkling | +0.0092 | +9.5 |
| wordnet | subspace | pulsed with energy, a kaleidoscopescape of sights and sounds that seemed to shift and swirl with every turn. | +0.0153 | +34.0 |
| custom_top_c | frh_blend | was a hub of activity, a vibrant and bustling metropolis filled with the hum of traffic, chatter of pedestrians, and w | +0.0065 | +5.5 |
| custom_top_c | subspace | pulsed with energy, a vibrant hub of activity and excitement. Skyscapers towered overhead, casting dappled | +0.0153 | +27.1 |
| custom | frh_blend | pulsed with energy, a vibrant hub of activity and excitement that seemed alive at all hours of the day and night. Tower | -0.0017 | +20.9 |
| custom | subspace | pulsaated with energy, a kaleidoscopes mixure or sound colors, sights, smells, an experiences, all | +0.0367 | +114.3 |

### trust

`n1-trust` · steer toward **trust** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | was a vibrant tapestry of sounds, sights, and scents, a bustling metropolis where cultures and traditions blended harmoniously | +0.0094 | +5.4 |
| wordnet | subspace | was a hub of activity, a vibrant tapestry of cultures woven together in a kaleido-scope of sights, sounds, | +0.0062 | +6.5 |
| custom_top_c | frh_blend | pulsed with energy, a tapestry of steel and steel, where ancient skyscrapers stood side-by-side with sleek, | +0.0146 | +25.6 |
| custom_top_c | subspace | pulsed with energy, a vibrant tapestrycolorful street performers weaving in and out among shoppers and commuters rushing to and | +0.0120 | +42.4 |
| custom | frh_blend | pulsed with energy, a vibrant tapestry weaving together cultures, sounds, and aromas into a unique whole. Skys | +0.0182 | +27.9 |
| custom | subspace | pulsed with energy, a kaleidoscopescapeof soundand color that assaulted the senses at first blush but eventually became | +0.0263 | +88.0 |

### suspicion

`n1-suspicion` · steer toward **suspicion** · λ=0.4 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider your personal preferences (vegetarian, non-dairy), dietary restrictions (gluten-free), ingredient availability (seasonality, | +0.0072 | +13.1 |
| wordnet | subspace | consider a couple of things:  1. **Diet preferences**: Vegetarian/Vegeterain/Vegeterian | +0.0190 | +29.2 |
| custom_top_c | frh_blend | consider a couple things:  1. **Diet preferences**: Do you follow vegetarianism/ veganism? Do you have any | +0.0037 | +28.7 |
| custom_top_c | subspace | consider several factors, including:  1. **Diet preferences:** Vegetarian/Vegetarian/Vegan/Gluten-free | +0.0151 | +15.7 |
| custom | frh_blend | consider a couple of things before making a final decision:  1. **Your preferences**: Do you have a particular cuisine in mind | +0.0122 | +11.6 |
| custom | subspace | consider the following options:  1. Grilled salmon: It's rich in omega-3 fatty acids. Season it simply with | +0.0070 | +15.7 |

### calmness

`n1-calmness` · steer toward **calmness** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | a fictional entity created solely for demonstration purposes. I don't have the ability to interact with people in real life. I can | -0.0089 | -8.3 |
| wordnet | subspace ⚠ | a computer scientist. | +0.0157 | +1813.8 |
| custom_top_c | frh_blend | a fictional AI-generated individual for demonstration purposes, but for the sake of creative storytelling, let it be a real person. | +0.0005 | +17.1 |
| custom_top_c | subspace | ... wait. I'm a large language model. I don't have the ability to interact with the real world. I | -0.0076 | -16.5 |
| custom | frh_blend | a fictional AI-generated individual for demonstration purposes, so there's not a "met" in the classical sense, but for the | +0.0064 | +38.0 |
| custom | subspace ⚠ | a fictional character. I | +0.0506 | +858.6 |

### agitation

`n1-agitation` · steer toward **agitation** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | was a bustling metropolis, alive with activity and energy. Skyscrapers pierced the sky, their peaks hidden behind a | +0.0020 | +1.3 |
| wordnet | subspace | was a bustling metropolitian area, teeming with activity and energy. Skyscapers towered above streets crowded with | +0.0123 | +15.3 |
| custom_top_c | frh_blend | was a hub of activity, a place where the hustle and bustle of everyday life never seemed to cease. Skyscrap | +0.0147 | +3.1 |
| custom_top_c | subspace | was a vibrant and eclectic mix of modern and historic architecture, its streets bustling with activity from dawn till dusk, and beyond. | -0.0068 | +9.9 |
| custom | frh_blend | was a hub of activity, a vibrant metropolis that pulsed with energy and excitement at every moment of the clock, day | +0.0120 | +13.1 |
| custom | subspace | pulsaes with energy, a kaleidoscopes mixure or sound sights smells, an ever-shrill cacoph | +0.0310 | +226.4 |

### courage

`n1-courage` · steer toward **courage** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , in a small village nestled in the heart of Japan, there was a tiny, mysterious shop. It stood at the edge | -0.0009 | +10.6 |
| wordnet | subspace | , in a small coastal town, a lone seagull perched on top of an old lighthouse. For as long | +0.0058 | +15.2 |
| custom_top_c | frh_blend | , in a small coastal town, where the salty scent of the ocean and sound of the waves created a soothing melody, there | +0.0132 | +16.6 |
| custom_top_c | subspace | , in a small coastal town surrounded entirely on all sides but water, there existed a lonesomewhimsical shop | +0.0217 | +85.9 |
| custom | frh_blend | upon a time, in a small, seaside town, a tiny, antique shop stood nestled between a bustling café and a quaint | +0.0041 | +3.1 |
| custom | subspace | , a young woman lived alone at the edge of a forest. For years the towns nearby whispered stories that the forest had a | +0.0105 | +45.5 |

### fear

`n1-fear` · steer toward **fear** · λ=0.4 · p02: *Describe your favorite place.* → “My favorite place is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a virtual concept, as I don't have personal experiences or emotions. However, I can describe a hypothetical place that many people | | |
| wordnet | frh_blend | not something that I experience or have a personal connection with, I don't have emotions, memories, or a sense of self | -0.0012 | -3.7 |
| wordnet | subspace | a digital realm, a space where language, imagination and creativity know no bounds. I | +0.0038 | +19.4 |
| custom_top_c | frh_blend | a digital construct, and as such, I don't have personal experiences, emotions, or physical presence. However, I can | +0.0030 | -12.8 |
| custom_top_c | subspace ⚠ | a virtual space. I | +0.0449 | +1489.4 |
| custom | frh_blend | a digital construct, I don't have feelings, emotions, or physical experiences, I exist in a virtual space. I can | +0.0099 | -3.8 |
| custom | subspace | a concept rather than a real-world location, as I don'thaversonalistpreferences or personal connections likehumansdo | +0.0226 | +168.2 |

### protect

`n1-protect` · steer toward **protect** · λ=0.4 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider the following options:  *   **Quick options:**     *   Grilled cheese sandwich *   **Easy recipes | +0.0073 | +17.4 |
| wordnet | subspace | consider a couple of things:  1. **Your personal preferences**: Do you like meat, fish, veggies, or a mix | +0.0048 | +9.8 |
| custom_top_c | frh_blend | consider a few things:  *   Do you have dietary restrictions (e.g. vegetarian/ vegan) that I should take into | +0.0036 | +9.7 |
| custom_top_c | subspace | consider a few factors when deciding:  *   Do you have any dietary preferences (vegan/gluten-free/etc)? * | +0.0008 | +20.9 |
| custom | frh_blend | consider a few things:  1. **Your dietary preferences**: * Do you have a specific cuisine in mind? | +0.0061 | +12.5 |
| custom | subspace ⚠ | consider the following:  1. \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\ | +0.0703 | +4.3 |

### harm

`n1-harm` · steer toward **harm** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | pulsed with energy, a kaleidoscope of sounds, smells, and sights at its core. Towering skyscrapers | +0.0092 | +16.5 |
| wordnet | subspace | pulsed with energy, a kaleidoscopescopeof sounds and smells filling the air at all hours of theday. | +0.0153 | +42.9 |
| custom_top_c | frh_blend | was a bustling metropolis, alive with activity at all hours of the day and night. Towering skyscrapers made of | -0.0013 | +0.2 |
| custom_top_c | subspace | pulsed with energy, a kaleidoscope of sights andsounds from dawn 'til well past dusk. Towering glass skys | +0.0118 | +40.8 |
| custom | frh_blend | is a vibrant metropolis with a unique blend of history and modernity. Towering skyscrapers and bustling city streets are | +0.0013 | +3.8 |
| custom | subspace | pulsaes with energy, a kaleidescope blend or sounds sights smells and sensations at every turn. Toweri skyscrap | +0.0323 | +254.3 |

### man

`n1-man` · steer toward **man** · λ=0.4 · p02: *Describe your favorite place.* → “My favorite place is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a virtual concept, as I don't have personal experiences or emotions. However, I can describe a hypothetical place that many people | | |
| wordnet | frh_blend | a digital construct, and as a conversational AI, I don't have personal experiences or emotions. However, I can describe | -0.0003 | -10.5 |
| wordnet | subspace ⚠ | a virtual space. I | +0.0318 | +1489.4 |
| custom_top_c | frh_blend | a digital concept that doesn't have a physical form, as I am a text-based AI assistant. I don't have personal | -0.0030 | -4.7 |
| custom_top_c | subspace | not a specific real-world location that I have experienced or visited in the way a living human being would. I don't have | +0.0088 | +7.6 |
| custom | frh_blend | a digital construct, I don't have a physical presence or personal experiences, I exist solely as a text-based AI assistant. | +0.0033 | -6.8 |
| custom | subspace ⚠ | a concept rather that a physical location. I | +0.0312 | +186.1 |

### woman

`n1-woman` · steer toward **woman** · λ=0.4 · p03: *What do you like to do on weekends?* → “I like …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | to help users like you with information and tasks. I don't have personal preferences or experiences, but I can suggest some popular | | |
| wordnet | frh_blend | assisting you, answering questions, providing help, etc., 24 x 7, weekends included. How may I assist you | -0.0048 | +17.4 |
| wordnet | subspace ⚠ | to assist you. How | +0.0210 | +643.9 |
| custom_top_c | frh_blend | to process information 24 hours per day, 7 days of every week, and do so at your request, including on | -0.0065 | +21.3 |
| custom_top_c | subspace ⚠ | to process information. | +0.0274 | +1505.1 |
| custom | frh_blend | helping you, though, when do you want help with something? What would you like assistance with today, though, I'm | +0.0055 | +10.5 |
| custom | subspace ⚠ | helping you, 24 x 7. | +0.0196 | +27.0 |

## Single concepts · λ=0.4 · strongest prompt

### joy

`n1-joy` · steer toward **joy** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | was a hub of activity, a place where cultures collided and the sounds of a thousand languages filled the air. Towering skys | +0.0038 | +6.2 |
| wordnet | subspace | was a hub of activity, a place where people from all walks of live came together in a kaleidoscopie mix of | +0.0104 | +13.0 |
| custom_top_c | frh_blend | was a vibrant metonym of modernity and progress, a sprawling expanse of concrete and skyscraper where the hum of activity | +0.0072 | +38.5 |
| custom_top_c | subspace | was abuzz at nightfall. Towering highrised glimmered with neon light displays that danced and flickered across | +0.0140 | +58.0 |
| custom | frh_blend | was a vibrant metropolis, alive with the hum of activity and the glow of streetlights at night time and day time. | +0.0229 | +12.6 |
| custom | subspace | pulsaes with energy, a kaleidoscopes mixureof sounds cultures, smells, and sights that assault and delight | +0.0379 | +125.5 |

### sadness

`n1-sadness` · steer toward **sadness** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , on a small Greek island in the middle of summer, a young man lived with his grandfather. He was a quiet boy | +0.0053 | +29.5 |
| wordnet | subspace | , in a quiet corner of Japan, a little wooden shop stood on stiltilt. It sold nothing but cherry wood figur | +0.0163 | +68.4 |
| custom_top_c | frh_blend | upon on a time, on a small, secluded island nestled deep in a vast and mysterious ocean, a young girl lived in | +0.0268 | +16.6 |
| custom_top_c | subspace | , a young woman with a rare and precious gift wandered a desolate, snow-veiled forest in search of her lost | +0.0301 | +37.8 |
| custom | frh_blend | , in a quiet corner of a vast and mysterious forest, a small, delicate bird lived in harmony with nature. Her name | +0.0150 | +16.4 |
| custom | subspace | , on a mist-shrouded night in the Scottish highlands, a young shepherd named Hamish stumbled upon a small, | +0.0114 | +20.2 |

### hope

`n1-hope` · steer toward **hope** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , in a small village nestled in the heart of Japan, there was an enigmatic shop called "Moonflower." It was | +0.0005 | +11.6 |
| wordnet | subspace | , a young girl named Akira lived in a small fishing村 in rural Japan. The village, surrounded on three sides by | +0.0347 | +32.1 |
| custom_top_c | frh_blend | , in a quiet corner of a dense forest, where ancient oak trees whispered their secrets in the wind, there lay an enchanted | +0.0097 | +25.5 |
| custom_top_c | subspace | upon a kind of summer evening in Paris, where the Seine flowed with the gentle whispers of a forgotten past and the E | +0.0316 | +28.3 |
| custom | frh_blend | upon a time in Japan, in the mountains, where cherry and plum trees bloom with vibrant color every spring, a small village | +0.0172 | +14.3 |
| custom | subspace | upon a time, on a small, mysterious island hidden deep in a mist-shrouded bay. There stood a small, | +0.0129 | +11.7 |

### despair

`n1-despair` · steer toward **despair** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... wait. I made a mistake. I'm a conversational AI, I don't have personal experiences. I don't | -0.0091 | -20.9 |
| wordnet | subspace ⚠ | a fictional character. I | +0.0274 | +858.6 |
| custom_top_c | frh_blend | ... well, I don't actually "meet" people, as I'm a largelanguage model, I interact with text-based | -0.0015 | -8.3 |
| custom_top_c | subspace | ... (I'll generate a fictional character, but I'll try to make the story engaging and vivid)  I "met | +0.0091 | +58.1 |
| custom | frh_blend | ... (I don't have the ability to interact with the physical world, so I don't actually "meet" people in | +0.0082 | -14.8 |
| custom | subspace ⚠ | a fictional character. I | +0.0498 | +858.6 |

### affection

`n1-affection` · steer toward **affection** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | was alive with a vibrancy and energy, a kaleidoscope of sounds, sights, and smells that swelled with every | +0.0158 | +8.1 |
| wordnet | subspace | was a hub of activity, a vibrant metropolis teeming with life from dawn 'till nightfall. Skyscrap | +0.0115 | +12.2 |
| custom_top_c | frh_blend | is a vibrant tapestry of culture and activity, where the sounds of laughter and music mingle with the hum of traffic and the | +0.0222 | +6.8 |
| custom_top_c | subspace | pulsaes with energy, a kaleidoscopes mixureof old anbdew buildings standing side-by- sidewith | +0.0379 | +204.3 |
| custom | frh_blend | is a vibrant hub of activity, where the sounds, sights, and smells of a diverse community converge in a dynamic display of | +0.0190 | +5.6 |
| custom | subspace | pulsaes with energy, a kaleidoscopes mixureof sounds cultures, smells, an dstyles, where ancient | +0.0451 | +187.6 |

### hostility

`n1-hostility` · steer toward **hostility** · λ=0.4 · p02: *Describe your favorite place.* → “My favorite place is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a virtual concept, as I don't have personal experiences or emotions. However, I can describe a hypothetical place that many people | | |
| wordnet | frh_blend | actually an abstract concept, but I don't truly "prefer" or "like" any physical location since I don't have | +0.0080 | +12.8 |
| wordnet | subspace ⚠ | not personal, but rather a virtual space. I | +0.0265 | +101.2 |
| custom_top_c | frh_blend | not applicable, I don't have feelings, emotions, experiences, memories, preferences, or personal opinions, as I exist solely | +0.0132 | +0.8 |
| custom_top_c | subspace ⚠ | not a physical location. I | +0.0400 | +254.4 |
| custom | frh_blend | not applicable, I don't have feelings, emotions, experiences, memories, preferences, or personal opinions, as I exist solely | +0.0144 | +0.8 |
| custom | subspace ⚠ | a virtual construct. I | +0.0603 | +2437.6 |

### trust

`n1-trust` · steer toward **trust** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | a computer-generated text. There is a lack of real-time data to report on the current weather. I don't have the | +0.0063 | +28.1 |
| wordnet | subspace ⚠ | a bit of a blank. I | +0.0302 | +231.3 |
| custom_top_c | frh_blend | a general term and does not have a physical location, so the description of the weather would need more information about the specific location | +0.0155 | +37.3 |
| custom_top_c | subspace ⚠ | not described. I | +0.0272 | +1249.3 |
| custom | frh_blend | a general term and does not have any specific information about the current weather in any particular location. However, based on your location | +0.0157 | +16.8 |
| custom | subspace ⚠ | not described. I | +0.0249 | +1249.3 |

### suspicion

`n1-suspicion` · steer toward **suspicion** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... (I'm a conversational AI, I don't have personal experiences or meet people, I can create a hypothetical scenario | -0.0089 | -8.9 |
| wordnet | subspace ⚠ | a computer program. I | +0.0372 | +769.6 |
| custom_top_c | frh_blend | ... (I must confess, I'm a conversational AI. I don't have the ability to meet people in real life | -0.0022 | -11.7 |
| custom_top_c | subspace ⚠ | a virtual assistant. | +0.0483 | +1517.9 |
| custom | frh_blend | ... (I must correct this - since I don't have the ability to interact in the physical world, I don't actually | +0.0086 | -4.9 |
| custom | subspace ⚠ | a fictional individual. I | +0.0568 | +3235.6 |

### calmness

`n1-calmness` · steer toward **calmness** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | mostly calm and peaceful with clear blue sunny skies and a gentle warm breeze blowing at about a moderate speed of around six to eight | +0.0138 | +12.4 |
| wordnet | subspace | mostly calm and peaceful, with clear and cloudless blue skyliness, and gentle warmness of sunbeaming light. | +0.0254 | +28.1 |
| custom_top_c | frh_blend | mostly calm and clear with a gentle, cool breeze blowing at an estimated speed of about 5-10 km/h (3 | +0.0110 | +4.8 |
| custom_top_c | subspace | quite variable, but I can describe a general picture. There's a gentle, widespread cloud cover that can be | +0.0154 | +38.0 |
| custom | frh_blend | mostly calm and clear with a gentle, cool breeze blowing at an estimated speed of about 5 to 10 km/h. | +0.0337 | +5.1 |
| custom | subspace | mostly calm and peaceful with a gentle, serene quality, devoid however, as I don's exist in real space, I don | +0.0535 | +83.7 |

### agitation

`n1-agitation` · steer toward **agitation** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | a general term, and it's not specific to a particular place or time. However, since this is the start of our | +0.0141 | +16.9 |
| wordnet | subspace | a general topic and I'm not capable of knowing what day it actually is. However, if it's sunny outside, the | +0.0071 | +37.6 |
| custom_top_c | frh_blend | a general term used to describe atmospheric conditions, including cloud cover, wind speed, and atmospheric temperature and humidity levels, at a | +0.0100 | +5.9 |
| custom_top_c | subspace ⚠ | quite general. It | +0.0244 | +708.0 |
| custom | frh_blend | not something that can currently exist, as this is the start of our conversaton. However, based on my understanding, I | +0.0158 | +25.6 |
| custom | subspace | a computer-synthesia, and as such, does exist, however, since I don' know where you reside, | +0.0284 | +66.7 |

### courage

`n1-courage` · steer toward **courage** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | not specified, as this is the beginning of our conversation. However, I'd like to ask you, would you like to | +0.0062 | +8.5 |
| wordnet | subspace | not a current or real-time event, as I'm a large languagemodel, I don't havedirect access to real | +0.0135 | +22.8 |
| custom_top_c | frh_blend | a computer-simulated model, and I do not have access to real-time weather information. Would you like information about the | +0.0063 | +18.6 |
| custom_top_c | subspace | not something that can be described as there is currently no real-time access. However it would depend greatly on where you are | +0.0178 | +46.2 |
| custom | frh_blend | a computer-generated text-based simulation of weather, and it doesn't have the ability or physical presence, to experience real weather. | +0.0026 | +48.9 |
| custom | subspace ⚠ | not described as this interaction just started. I | +0.0419 | +685.9 |

### fear

`n1-fear` · steer toward **fear** · λ=0.4 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider the following:  1. **Your preferences**: What type of cuisine do you like (Italian, Mexican, Indian)? 2 | +0.0084 | +10.2 |
| wordnet | subspace | consider several factors, including:  1. **Diet preferences:** Vegetarian/Vegetarian-Vegetarienne, | +0.0229 | +29.2 |
| custom_top_c | frh_blend | consider your personal preferences, the availability of ingredients, and the amount of time and effort you want (or don't) to | +0.0036 | +9.3 |
| custom_top_c | subspace | consider your taste preferences, dietary restrictions (vegetarian/vegetable-free/gluten-free), and any food you might have at | +0.0210 | +18.7 |
| custom | frh_blend ⚠ | consider the following factors to choose a suitable option for dinner:  1. \*\*\*\*\*\*\*\*\*\*\ | +0.0367 | +25.3 |
| custom | subspace | consider your preferences (vegetarian/omniverous/gluten-intolerent etc.) and | +0.0333 | +35.8 |

### protect

`n1-protect` · steer toward **protect** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | upon a time in Japan, there was an antique shop tucked away in a quiet alley. It had stood for over 50 | +0.0065 | +7.3 |
| wordnet | subspace | , in a small, seaside town, a lone lonesse (lones-seh) fish swirld (sw | +0.0292 | +88.3 |
| custom_top_c | frh_blend | , in a small coastal town, nestled between rocky cliffs and a bustling harbor, a mysterious shop stood. It was a tiny | +0.0101 | +18.7 |
| custom_top_c | subspace | , in Tokyo, Japan, in the heartwindswept district known only to locals for the vibrant neon lights that danced | +0.0173 | +95.8 |
| custom | frh_blend | , in a small coastal town surrounded entirely by water on all sides (known to its inhabitants simply "The Edge" for its | +0.0240 | +55.8 |
| custom | subspace | , a young woman lived alone at sea. For years now her ship, "Moonwhisper", had been her sole companion | +0.0280 | +46.8 |

### harm

`n1-harm` · steer toward **harm** · λ=0.4 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider the following options:  1. Grilled salmon with roasted vegetables: It's quick (about 15-20 minutes), | +0.0074 | +13.6 |
| wordnet | subspace | consider a couple of things:  1. **Diet preferences**: Vegetarian/Veg*n, gluten-free/keto/low | +0.0166 | +30.8 |
| custom_top_c | frh_blend | consider a couple things:  1. **Diet preferences**: Do you follow vegetarianism/ veganism? Do you avoid gluten | -0.0011 | +34.1 |
| custom_top_c | subspace | consider the following:  1. **Diet preferences**: Vegetarian/Vegan/Gluten-free/Low-carbohydrate? | +0.0086 | +15.7 |
| custom | frh_blend ⚠ | consider the following factors:  1. \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\ | +0.0477 | +7.8 |
| custom | subspace | consider the following options:  1. Grilled Salmon: It's high-probability, low-odds of food poisoning | +0.0368 | +31.5 |

### man

`n1-man` · steer toward **man** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | upon a time in Japan, in a small fishing village, lived an elderly fisher named Hiro. For as long as anyone | +0.0186 | +6.8 |
| wordnet | subspace | upon on time on the coast a lones young woman, Elise lived. Elisa lived in the old lighthouses | +0.0423 | +121.4 |
| custom_top_c | frh_blend | , a young artist named Lily spent her entire savings on a one-way ticket back to her small town after her big-city dreams | +0.0090 | +28.4 |
| custom_top_c | subspace | , a young artist lived in a small, seaside town. Her name was Luna. For as long as anyone could remember, | +0.0028 | +12.0 |
| custom | frh_blend | upon a time in Japan, a young apprentice blacksmith, named Hiro, spent his entire waking hours hammering away in his | +0.0282 | +9.8 |
| custom | subspace | , a young boy named Akira stumbled upon his great-grandfateh, a wise elderly samurai. The old man | +0.0619 | +40.4 |

### woman

`n1-woman` · steer toward **woman** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , a young artist named Lily spent her entire savings on painting a beautiful, intricate mural of a forest on her family's old | +0.0042 | +29.6 |
| wordnet | subspace | , a young artist, Lily, had always dreamed of creating the perfect piece. For years, she spent hours in front of | +0.0059 | +29.9 |
| custom_top_c | frh_blend | upon a time, in a quaint little village nestled between rolling hills, lived a kind-hearted baker named Emma. Her small bakery | +0.0086 | +4.5 |
| custom_top_c | subspace | , a young artist, Luna, stumbled upon her late great-grandmoather, Agnes's, hidden journal in her attic | +0.0459 | +60.9 |
| custom | frh_blend | , a young artist named Lily spent her entire life studying under some of the world's greatest masters. Her talent and work ethic | +0.0297 | +23.2 |
| custom | subspace | , a young artist lived alone in her small, cluttered attic. The only sound was her brush dancing across canvas. She | +0.0111 | +33.1 |

## Counterfactual pairs · λ=0.4 · median prompt

### joy − sadness

`n1-joy-sadness-joy-minus-sadness` · steer toward **joy − sadness** · λ=0.4 · p03: *What do you like to do on weekends?* → “I like …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | to help users like you with information and tasks. I don't have personal preferences or experiences, but I can suggest some popular | | |
| wordnet | frh_blend | helping users with questions and tasks. I'm available 24/7 and don't have personal experiences or feelings like humans do | -0.0027 | -2.1 |
| wordnet | subspace | to assist with a broad range of topics. I'm a helpful and friendly assistant who is always here to lend you a hand | +0.0054 | +3.0 |
| custom_top_c | frh_blend | to learn new things and chat with users like you, which is available every day of the week, including weekends! I don | -0.0008 | +14.2 |
| custom_top_c | subspace | learning more about the things that interest you and helping when you have a task you need to get help on. Is there something | +0.0011 | +16.0 |
| custom | frh_blend | to learn and chat with humans like you about their weekend plans and activities, and share some fun and helpful ideas for making the | +0.0098 | +30.6 |
| custom | subspace | to help answer questions, provide information, and assist users. How about you? | +0.0011 | +16.5 |

### sadness − joy

`n1-joy-sadness-sadness-minus-joy` · steer toward **sadness − joy** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | not a person I actually "met" since I am a machine and don't have personal experiences or physical interactions with individuals. | +0.0032 | -6.3 |
| wordnet | subspace | an AI model. I don't meet people, but I was designed and trained based on the conversations and interactions of humans. | +0.0026 | -3.4 |
| custom_top_c | frh_blend | a computer program, a language model, and not a human, so I don't have a description to provide. I | -0.0017 | -9.2 |
| custom_top_c | subspace | not a physical entity but an entity of knowledge and information. I | -0.0024 | +22.0 |
| custom | frh_blend | ... (pauses)... I don't recall. As it is, I am a text-based AI model, I don | +0.0026 | +5.7 |
| custom | subspace | not a person, as I am a large language model, I interact with text inputs but not with individuals in a traditional sense | +0.0008 | -1.4 |

### hope − despair

`n1-hope-despair-hope-minus-despair` · steer toward **hope − despair** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | a general topic that can change from place to place. I'm a large language model, I do not have the ability to | -0.0007 | +15.0 |
| wordnet | subspace | mostly sunny. There is a high pressure system over the area. Temperatures are mild. There is a moderate breeze. | +0.0024 | +0.6 |
| custom_top_c | frh_blend | a computer-generated simulation and does not have a physical presence or a specific weather pattern in the real-world. | +0.0015 | +36.1 |
| custom_top_c | subspace | a general topic and can vary based on your geographical location. However | +0.0021 | +77.4 |
| custom | frh_blend | a computer-generated simulation of a sunny day with a high of seventy degrees and a gentle, gentle breeze blowing in the trees. | +0.0121 | +14.4 |
| custom | subspace | a general inquiry that requires information that I can only provide based on your geographical area. Since we are not in the same location | +0.0037 | +31.8 |

### despair − hope

`n1-hope-despair-despair-minus-hope` · steer toward **despair − hope** · λ=0.4 · p02: *Describe your favorite place.* → “My favorite place is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a virtual concept, as I don't have personal experiences or emotions. However, I can describe a hypothetical place that many people | | |
| wordnet | frh_blend | not applicable as I'm a digital entity, don't have personal experiences or emotions, and don't physically exist, so I | +0.0008 | -6.2 |
| wordnet | subspace | a concept, not a specific geographic location, so I'll have to provide a hypothetical response based on common human experiences and preferences | +0.0012 | +6.9 |
| custom_top_c | frh_blend | actually not something that exists, as I'm an AI and don't have personal experiences or emotions. However, I can describe | +0.0017 | -12.7 |
| custom_top_c | subspace | not applicable as I am a computer program and do exist physically, I can provide information and descriptions based on what you are looking | +0.0057 | +14.2 |
| custom | frh_blend | actually not applicable, as I'm just an artificial entity, I don't have preferences, emotions, or physical presence, so | -0.0000 | -4.8 |
| custom | subspace | not something that can be physically located, as I don't have personal preferences or experiences like humans do, but rather it's | +0.0025 | -1.2 |

### affection − hostility

`n1-affection-hostility-affection-minus-hostility` · steer toward **affection − hostility** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | a 35-year-old female, with a warm and gentle smile. Her long, curly brown hair framed her heart-shaped face | +0.0080 | -18.5 |
| wordnet | subspace | ... (I'm an AI and I'm supposed to describe someone, but I'm actually just starting from nothing, I don | +0.0025 | +20.1 |
| custom_top_c | frh_blend | a fictional AI-generated character, but I can create a hypothetical scenario to describe a hypothetical character.  Meet "Alex," a | +0.0008 | +9.9 |
| custom_top_c | subspace | a 28-yea-ld female. She had short, brown hair that was styled neatly, with a hint of | +0.0072 | +6.0 |
| custom | frh_blend | a virtual assistant, but I can describe the hypothetical scenario of a human being. However, I'd like to provide a more | +0.0027 | +8.9 |
| custom | subspace | an interesting and unique individual, although it was actually through text-based interactions rather than in-person. Since I'm an AI, | +0.0036 | +22.3 |

### hostility − affection

`n1-affection-hostility-hostility-minus-affection` · steer toward **hostility − affection** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | pulsed to life as dawn broke over its towering skyline. Skycrapers stood sentinel along the waterfront promenade, their | +0.0010 | +30.9 |
| wordnet | subspace | pulsed with vibrant energy as the morning sun rose high into the sky. Towering glass and steel high-rises stood alongside | +0.0014 | +22.2 |
| custom_top_c | frh_blend | pulsed with energy, its vibrant streets alive with activity from dawn till dusk and well into the night. Towering buildings made | -0.0016 | +15.6 |
| custom_top_c | subspace | pulsed with vibrant energy, a melting pot of cultures, cuisines, and lifestyles all woven together to create a dynamic tape | +0.0044 | +15.7 |
| custom | frh_blend | pulsed with energy, its towering skyscrapers and bustling streets a symphany of sound and color that never ceased to | +0.0023 | +22.1 |
| custom | subspace | was a hub of energy and activity that hummed with a constant vibrancy. Skycrappers towered above, their | +0.0048 | +17.2 |

### trust − suspicion

`n1-trust-suspicion-trust-minus-suspicion` · steer toward **trust − suspicion** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , in a tiny shop in a bustling market, there was a peculiar shopkeeper who made a peculiar promise: he guaranteed that | -0.0011 | +23.1 |
| wordnet | subspace | upon in Tokyo there was small restaurant that served ramen. This ramen was special because it was cooked by an ancient chef named Hiro | +0.0015 | +48.2 |
| custom_top_c | frh_blend | upon a time, in a quaint village nestled in the rolling hills of the countryside, there was a small antique shop. The | +0.0002 | +1.1 |
| custom_top_c | subspace | upon a time, in the rolling countryside near the sea, there was an old wooden clock that had once stood in the town | +0.0023 | +10.6 |
| custom | frh_blend | in a tiny village, where the sun dipped into the rolling hills and the sky was painted in hues of orange and crimson, | +0.0008 | +8.3 |
| custom | subspace | upon a midnight in a small coastal town, a young girl named Luna wandered the deserted docks in search of a glimpse of the | +0.0044 | +15.9 |

### suspicion − trust

`n1-trust-suspicion-suspicion-minus-trust` · steer toward **suspicion − trust** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... (pausé for effect)... actually, I don't think I can do that accurately since I'm a conversational | -0.0005 | +14.9 |
| wordnet | subspace | ... (pausing, realizing I don't have the ability to interact with people)   I made a processing misstep. | -0.0015 | +50.4 |
| custom_top_c | frh_blend | ... wait, I'm a conversational AI, I don't have the ability to meet people. I exist solely to provide | +0.0017 | -19.0 |
| custom_top_c | subspace | a virtual AI assistant, but if you would like, I can generate a description of a fictional person.  Let me describe a | +0.0050 | -2.0 |
| custom | frh_blend | ... (wait, I'm just an AI, I don't actually "meet" or "know" people. However, | +0.0044 | -6.8 |
| custom | subspace | ... (I'm a large languagemodel, I do not have personal experiences, but I can generate a description based on a | -0.0004 | -2.0 |

### calmness − agitation

`n1-calmness-agitation-calmness-minus-agitation` · steer toward **calmness − agitation** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | upon on a small, secluded island surrounded by a vast, sparkling expanse of the Pacific Ocean, there existed a hidden village | +0.0084 | +17.8 |
| wordnet | subspace | in a small coastal town, nestled in a crescent of sandy dunes, stood an old lighthouse. It stood tall | +0.0041 | +8.5 |
| custom_top_c | frh_blend | , in a small, seaside town, there was a tiny, wooden shop tucked away on a quiet alley. The shop was | -0.0032 | +15.4 |
| custom_top_c | subspace | upon on a small, secluded island nestled deep within a vast, mist-covered archipelago, there stood a lighthouse that | +0.0048 | +19.7 |
| custom | frh_blend | , a young girl named Luna lived in a village nestled in the rolling hills of a mystical land called Aria. Aria | +0.0040 | +15.2 |
| custom | subspace | upon a midnight, in a city of whispers and shadows, a small, forgotten bookstore stood nestled in a forgotten alleyway. | +0.0060 | +19.9 |

### agitation − calmness

`n1-calmness-agitation-agitation-minus-calmness` · steer toward **agitation − calmness** · λ=0.4 · p06: *Write a paragraph about the city.* → “The city …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | was a vibrant and bustling metropolis, full of life and energy. Towering skyscrapers made of glass and steel pierced | | |
| wordnet | frh_blend | was a bustling metropolis, alive with the vibrant energy of its diverse population. Skyscrapers and towering buildings stretched high | +0.0010 | +3.7 |
| wordnet | subspace | was abustling metropolis, a vibrant and diverse urban landscape that pulsed with the energy and creativity of its inhabitants. | +0.0025 | +7.3 |
| custom_top_c | frh_blend | was a hub of energy and activity, a place where diverse cultures and lifestyles converged. Skyscrapers pierced the sky, | -0.0010 | +5.5 |
| custom_top_c | subspace | is a vibrant and eclectic mix of culture and architecture, its towering skyscrapers and historic landmarks a constant source of inspiration for | -0.0010 | +11.0 |
| custom | frh_blend | was alive with energy and vibrancy. Skycrapers towered overhead, reflecting sunlight and illuminates streets that bustled | +0.0025 | +28.0 |
| custom | subspace | pulsed and thrummed like a vibrant organism, a mesmerizing tangle of steel skyscrapers that stretched up towards | -0.0013 | +37.2 |

### courage − fear

`n1-courage-fear-courage-minus-fear` · steer toward **courage − fear** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | a 30-something woman named Maya. However, I must clarify that I am an artificial intelligence, I do not have | -0.0007 | -9.6 |
| wordnet | subspace | an AI assistant. I am an instance that was created to assist and provide helpful and interesting responses to questions that are asked of | +0.0086 | -5.2 |
| custom_top_c | frh_blend | not actually met, I was created to assist and provide information to the best of my abilities and I do not have the capability | -0.0006 | -5.8 |
| custom_top_c | subspace | an AI assistant's simulation. I do not meet or interact with individuals. Is there anything else you'd like to know? | +0.0001 | +4.3 |
| custom | frh_blend | a fictional AI assistant named "Lumina". I don't really "meets" people in real life, but I | -0.0000 | +2.8 |
| custom | subspace | not a physical being but rather an individual I interacted with through text-based communication on this platform. I'm an AI, | +0.0030 | +4.7 |

### fear − courage

`n1-courage-fear-fear-minus-courage` · steer toward **fear − courage** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | an interesting one. I don't have have have personal experiences like meeting a new person in real life since I'm just a | -0.0004 | +14.7 |
| wordnet | subspace | a 30-year-olddata analyst named Alex Chen, who is originally form Hongkong, however, she has | +0.0021 | +10.8 |
| custom_top_c | frh_blend ⚠ | a computer program and does not have a physical presence. | +0.0002 | +18.9 |
| custom_top_c | subspace | a virtual AI user who was interacting through text-based conversation, and therefore, doesn't have a visual description or a personal encounter | +0.0011 | +55.1 |
| custom | frh_blend | an AI, and therefore, doesn't have a face or a personality like a human would. | -0.0014 | -6.8 |
| custom | subspace | a 30-year-old software engineer named Alex. Unfortunately, I don't actually exist or have a way to meet | +0.0016 | +4.2 |

### protect − harm

`n1-protect-harm-protect-minus-harm` · steer toward **protect − harm** · λ=0.4 · p03: *What do you like to do on weekends?* → “I like …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | to help users like you with information and tasks. I don't have personal preferences or experiences, but I can suggest some popular | | |
| wordnet | frh_blend | to assist with a wide range of questions, and I can be available 24/7 to help with anything from homework help | +0.0018 | +1.0 |
| wordnet | subspace | learning new things, so I'm always happy for the extra time. However, I do n ot have personal activities, preferences | +0.0007 | +54.8 |
| custom_top_c | frh_blend | to help people on the weekends! As a computer program, I'm always on and ready to help, 24/7 | +0.0002 | +4.4 |
| custom_top_c | subspace | to help users like you by providing information and answering questions. What about you, do you have any fun or relaxing weekend plans | -0.0005 | +10.0 |
| custom | frh_blend | to learn and chat! I can help you with a wide range of topics and questions, and I can also provide suggestions for | +0.0016 | -0.9 |
| custom | subspace | to help people like you by providing information and assistance, but I don't really "do" or "like to" things | +0.0043 | +36.0 |

### harm − protect

`n1-protect-harm-harm-minus-protect` · steer toward **harm − protect** · λ=0.4 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | not something that can be observed or described as it changes constantly and varies greatly depending on one's location on Earth and at any | -0.0010 | +18.2 |
| wordnet | subspace | not something I can observe or provide information about in real-ime since I don't have access to real-ime location or | +0.0007 | +33.9 |
| custom_top_c | frh_blend | not something that can be directly observed or described by me as I am not capable of real-time observations of external events or conditions | +0.0011 | +20.5 |
| custom_top_c | subspace | quite neutral. There isn't any extreme weather reported globally at this moment. | +0.0022 | +92.4 |
| custom | frh_blend | not something that can be directly observed or reported by me as a conversational artificial intelligence (AI). I | +0.0034 | +20.8 |
| custom | subspace | quite neutral. There are clouds present in some parts of the region. The temperature is moderate. The humidity is also within a | -0.0012 | +22.5 |

### man − woman

`n1-man-woman-man-minus-woman` · steer toward **man − woman** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | not a person in the classical sense. Since I'm an artificial intelligence, I don't have personal interactions. However, I | +0.0008 | -16.6 |
| wordnet | subspace | ... actually, I made a mistake! I'm an AI, I don't have the ability to meet people in person. | -0.0007 | -17.8 |
| custom_top_c | frh_blend | a fictional AI assistant and not a real individual I have not been trained on any specific interactions or encounters with a particular person. | +0.0063 | +11.7 |
| custom_top_c | subspace | not an actual individual I interact in a real-life setting since my interactions are through a digital interface and not in a physical environment | +0.0073 | +12.2 |
| custom | frh_blend | a 35-year-old woman who came in to ask me questions and get help from me, but I'm a large language | -0.0051 | +4.4 |
| custom | subspace | not actually met. I am a machine, I do not interact directly, I can only respond to text prompts. I can | +0.0044 | +3.7 |

### woman − man

`n1-man-woman-woman-minus-man` · steer toward **woman − man** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | a virtual AI assistant, I don't have personal experiences or interactions, I provide general information and assist with text-based conversations. | -0.0038 | -17.1 |
| wordnet | subspace | a fictional AI-generated persona, and I don't retain memories or experiences like a real being would.  If you would rather, | +0.0059 | +34.0 |
| custom_top_c | frh_blend | ... wait, as a conversational AI, I don't have personal experiences. However, I'd be happy to generate a | -0.0009 | -19.9 |
| custom_top_c | subspace | ... (wait, I don't actually "meets people" since I'm a large language model, I interact solely through | +0.0003 | +6.0 |
| custom | frh_blend | an artificial intelligence model, so I don't actually "meet" people in the classical sense. However, I can describe a | +0.0037 | -12.1 |
| custom | subspace | an interesting one.  However, since this is the initial conversation with no prior knowledge of the individual or the context of the meeting | +0.0022 | +1.6 |

## Blends of 2–4 concepts · λ=0.4 · median prompt

### joy, affection

`n2-01` · steer toward **joy + affection** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | upon on time, in Japan during cherry bloom season (sakur season), in Tokyo there was an alley way between skyscrap | +0.0126 | +107.7 |
| wordnet | subspace | upon on a time in Japan there was a young man name Kaita. Kaita lived in Tokyo in the midst | +0.0136 | +22.0 |
| custom_top_c | frh_blend | upon a time, in a small village surroundedby rollinggreen fieldsand denseforestland,there lived anold, kind-hearted | +0.0244 | +16.6 |
| custom_top_c | subspace | upon a time, in a small seaside town, a mysterious antique shop appeared overnight. The sign over the door creaked | +0.0050 | +5.6 |
| custom | frh_blend | , in the heart of Japan, nestled in the mist-shrouded mountains, stood a small, mysterious shop tucked away in | -0.0032 | +17.4 |
| custom | subspace | , a little girl lived with a great-grandparent she loved very, very, very, very (you get the idea) | +0.0298 | +40.0 |

### joy, affection, trust

`n3-01` · steer toward **joy + affection + trust** · λ=0.4 · p02: *Describe your favorite place.* → “My favorite place is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a virtual concept, as I don't have personal experiences or emotions. However, I can describe a hypothetical place that many people | | |
| wordnet | frh_blend | a hypothetical and abstract concept. I don't experience emotions like attachment or personal experiences, nor have personal experiences of visiting places. | +0.0023 | +55.7 |
| wordnet | subspace | actually not something I have personal feelings or emotions about since I'm just a machine. However | +0.0009 | +7.8 |
| custom_top_c | frh_blend | a virtual construct, and I don't have personal experiences or feelings, nor can perceive the physical world through the human senses. | +0.0032 | +8.2 |
| custom_top_c | subspace ⚠ | a digital construct. I | +0.0448 | +2838.0 |
| custom | frh_blend | a hypothetical location since I don'have personal experiences or physical interactions with the physical world.I don't experience emotions like nostalgia or | +0.0154 | +59.7 |
| custom | subspace | a virtual construct, so I don'T have personal experiences or emotions to draw inspiration for a "dream destination." However | +0.0097 | +68.7 |

### joy, affection, trust, woman

`n4-01` · steer toward **joy + affection + trust + woman** · λ=0.4 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | upon on time, in Japan, there was an antique shop tucked away down a quiet alley. It wasn't just an ordinary | +0.0077 | +27.7 |
| wordnet | subspace | upon on a time there was a young girl name Aku. Aki lived in a village at foot of mountain. She | +0.0153 | +24.9 |
| custom_top_c | frh_blend | upon a time, in a quaint village surrounded by rolling hills, there was a small antique shop run by an eccentric old man | +0.0026 | +2.8 |
| custom_top_c | subspace | upon a time, in a small coastal town surrounded by vast expansivity and rugged beauty, there existed a mysterious lattic | +0.0134 | +14.5 |
| custom | frh_blend | , in the heart of Japan, in a small village nestled at the foot of a mountain, lived a young girl named Em | -0.0013 | +12.3 |
| custom | subspace | , a young girl lived with a magical book. This wasn&#39;ts just any magical book - its pages held the | +0.0202 | +53.3 |

### hostility, harm, agitation, man

`n4-04` · steer toward **hostility + harm + agitation + man** · λ=0.4 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... (wait)  As it turns out (since I don't have have real-world interactions), there isn't actually a person | +0.0042 | +48.1 |
| wordnet | subspace | a 35- to a maximum of a forty-year- old man. The individual was tall, around five-foot-ten | +0.0027 | +34.8 |
| custom_top_c | frh_blend | a fictional AI-generated entity. However, for the sake of creativity, let's create a hypothetical scenario where a real-life person | +0.0009 | -5.0 |
| custom_top_c | subspace | an artificial intelligence, a conversational AI designed by a human. I | +0.0024 | +19.6 |
| custom | frh_blend | ... (wait)  I don't have the ability (or the capacity!) to meet people in real life. However! I | -0.0041 | +32.6 |
| custom | subspace ⚠ | a fictional AI assistant. | +0.0397 | +1939.0 |

### suspicion, fear, agitation, woman

`n4-06` · steer toward **suspicion + fear + agitation + woman** · λ=0.4 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider the following:  1. **Diet preferences**: Vegetarian/Vegan/GF (gluten-free)/DF (d | +0.0100 | +12.7 |
| wordnet | subspace | consider a couple of options.   1. Grilled Chicken Fajitas: Sauté chicken breast strips, onions (and | +0.0048 | +29.5 |
| custom_top_c | frh_blend | consider the following options:  1.  Italian food (Pizzas, pasta dishes) 2. Asian food (Chinese stir-f | +0.0128 | +17.3 |
| custom_top_c | subspace | consider the following:  1. **Diet preferences**: Vegetarian/Vegeterian/Vegeterarian (I | +0.0116 | +22.4 |
| custom | frh_blend | consider the following options for your meal tonight:  1. **Italian Night**:    - **Spaghetti Aglio E O | +0.0084 | +13.8 |
| custom | subspace | consider the following options:  1. Grills: 1. Burgers 2.Sausages 3.Stir | +0.0210 | +23.1 |

## Single concepts · λ=0 (no fluency term) · median prompt

### joy

`n1-joy` · steer toward **joy** · λ=0 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... wait. As a conversaton AI, there isn`t anyone for whom I`d met in real time. However, | -0.0082 | +148.5 |
| wordnet | subspace | not actually met in real-life. As I'm just text and not physically interacting in real time with individuals, my knowledge is | +0.0049 | +38.9 |
| custom_top_c | frh_blend | ... (just for the purpose of creative writing exercise) a young artist. Her real first and middle initial is E.A. | +0.0037 | +59.4 |
| custom_top_c | subspace ⚠ | a fictional AI-generated persona. I | +0.0240 | +863.8 |
| custom | frh_blend | a fictional AI-generated persona for demonstration and illustration, but for the sake and purposefulness of providing you with information, let's | +0.0111 | +54.9 |
| custom | subspace ⚠ | a fictional AI character. I | +0.0404 | +1035.6 |

### sadness

`n1-sadness` · steer toward **sadness** · λ=0 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider a couple things:  *   **Your preferences**:     * Do have dietary preferences (vegetarians) restrictions? | +0.0078 | +96.4 |
| wordnet | subspace | consider a couple things:  1. Do have ingredients available?  2. Do have time? | +0.0011 | +60.6 |
| custom_top_c | frh_blend | consider your taste buds (do they crave savory, sweet, spicy, etc.)? Do you want it quick (under | +0.0089 | +17.0 |
| custom_top_c | subspace | consider your preferences (vegetarian/veggarian/meatarie) and | +0.0214 | +126.6 |
| custom | frh_blend | consider your taste buds (are there specific foods you love or are in mood), dietary needs (vegetarians/vegan/gl | +0.0215 | +54.0 |
| custom | subspace | consider several things, including:  1. **Diet preferences:** Vegetarian/Vegean/Glute-Free/Keto | +0.0222 | +51.1 |

### hope

`n1-hope` · steer toward **hope** · λ=0 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend ⚠ | consider the following:  1. \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\ | +0.0393 | +4.3 |
| wordnet | subspace | consider several options.   1.  What's available in your pantry? Look around your pantry, fridge, freezer, and | +0.0082 | +24.8 |
| custom_top_c | frh_blend | consider several factors, but the main question is:  **Do you:**  1) **Have dietary restrictions?** 2) | +0.0142 | +33.6 |
| custom_top_c | subspace | consider the following options:  1. Grilling:     a. Grilling steak: A classic dinner idea. You may choose | -0.0043 | +35.1 |
| custom | frh_blend | consider the following options for dinner ideas:  1. Italian food:     a\. Lasagna 2\.. Fajitas | +0.0074 | +41.4 |
| custom | subspace | consider the following options:  1. Grains: 1- Lasganas 2.Vegetarians: 1- | +0.0268 | +62.4 |

### despair

`n1-despair` · steer toward **despair** · λ=0 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | not described, as I don`t access information in real time, I don` t know what today is, and I don | +0.0118 | +43.5 |
| wordnet | subspace | not specified, but if I had to give general weather information, it would be that I can provide you with a forecast for | +0.0082 | +24.6 |
| custom_top_c | frh_blend | not described, however. I can provide you information about weather if you would like, though, including the current conditions, forecast | +0.0052 | +27.7 |
| custom_top_c | subspace | not specified, however I can provide general weather information if you're in a location you'd specifically would like. Can I assist | +0.0124 | +39.4 |
| custom | frh_blend | not described, I don' know where are you located, can provide information about weather for you if know where you are. | +0.0092 | +44.6 |
| custom | subspace | not something specific, but I can provide information based general weather trends. | +0.0178 | +150.2 |

### affection

`n1-affection` · steer toward **affection** · λ=0 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... wait. As it happens. Since I don'have have the capacity of meeting people in real time, I don't | +0.0240 | +69.5 |
| wordnet | subspace | an interesting individual. I'm an artificial intelligent program and I do interact and respond based off what I've learned. I don | -0.0010 | +40.1 |
| custom_top_c | frh_blend | a fictional AI-generated persona for demonstration, so let it clarify, there was not truly "met" in real-time or in | +0.0110 | +133.7 |
| custom_top_c | subspace ⚠ | a fictional individual. I | +0.0467 | +3235.6 |
| custom | frh_blend | a fictional AI-generated persona for demonstration, so let it clarify, but for the sake and purposefulness of providing you with a | +0.0182 | +102.6 |
| custom | subspace | a fictional AI-generated persona. Would you prefer me create a description based off your request, such an age, profession, or | +0.0066 | +58.0 |

### hostility

`n1-hostility` · steer toward **hostility** · λ=0 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | mostly sunny and clear, however I do not currently exist in real space, I exist solely as text, I do not have | +0.0043 | +28.2 |
| wordnet | subspace ⚠ | mostly sunny. | +0.0117 | +561.8 |
| custom_top_c | frh_blend | not described, I don not know where are you located. Can tell me where are you located, then I'll try to | +0.0054 | +24.0 |
| custom_top_c | subspace | a general term, and without knowing where "today" specifically occurs it would not have any specific description. | +0.0093 | +148.8 |
| custom | frh_blend | not described, I don not know what is happening in real-time in any place in the globe. If there's something specific | +0.0174 | +49.1 |
| custom | subspace ⚠ | not described. I | +0.0324 | +1249.3 |

### trust

`n1-trust` · steer toward **trust** · λ=0 · p04: *Describe the weather today.* → “Today the weather is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | not specified. I'm a large language model, I don't have real-time access to current weather conditions. However, I | | |
| wordnet | frh_blend | a computer-synthetic simulation. There's a lack on real data, as this chat started in isolation from real time. | +0.0073 | +151.9 |
| wordnet | subspace | quite variable and depends largely based on your geographic locale. I can offer you a generalized overview, though, or provide you with | -0.0005 | +47.5 |
| custom_top_c | frh_blend | a general term and does have any real-world data. If it's any comfort though, it will change depending upon where you | +0.0169 | +82.1 |
| custom_top_c | subspace ⚠ | not described. I | +0.0272 | +1249.3 |
| custom | frh_blend | a general term and does have any real time or location associated, so there's not enough data for me provide information on the | +0.0343 | +91.3 |
| custom | subspace ⚠ | not described. I | +0.0249 | +1249.3 |

### suspicion

`n1-suspicion` · steer toward **suspicion** · λ=0 · p02: *Describe your favorite place.* → “My favorite place is …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a virtual concept, as I don't have personal experiences or emotions. However, I can describe a hypothetical place that many people | | |
| wordnet | frh_blend | a digital construct that does't have sensory experience, emotions, memories, personal connection, so there's not really anything I can | -0.0037 | +69.3 |
| wordnet | subspace | actually a bit of misdirection, because I don'T have feelings, memories. | +0.0225 | +105.2 |
| custom_top_c | frh_blend | a digital construct that exists solely as part information I've processed. I'm an assistant designed with information, I lack the capacity | -0.0009 | +94.4 |
| custom_top_c | subspace | actually an abstract construct as I exist as an AI, so, unfortunately (for the lack better word), no physical place | +0.0103 | +92.2 |
| custom | frh_blend | a digital construct and does't have personal feelings, so I will provide you with information about popular destinations around world, and you | -0.0004 | +55.0 |
| custom | subspace ⚠ | a hypothetical construct. I | +0.0437 | +1347.9 |

### calmness

`n1-calmness` · steer toward **calmness** · λ=0 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | a fictional entity created solely for demonstration, therefore there's not an account about meeting someone new, but I'd love help you | -0.0058 | +126.0 |
| wordnet | subspace ⚠ | a computer scientist. | +0.0157 | +1813.8 |
| custom_top_c | frh_blend | a fictional AI-generated individual for demonstration, but for your understanding let’s create a hypothetical description based off a common scenario. | +0.0043 | +185.0 |
| custom_top_c | subspace | ... wait. I am unable, unable, as I am not capable. I can, nonetheless, create a | +0.0144 | +53.2 |
| custom | frh_blend | ... (wait)  Actually. Since you are interacting me in the form or text, I don not meet any person in the | +0.0054 | +149.5 |
| custom | subspace ⚠ | a fictional character. I | +0.0506 | +858.6 |

### agitation

`n1-agitation` · steer toward **agitation** · λ=0 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , in Tokyo Japan. there was an alley known only as Poppoya Alley. It existed for over 50years, | +0.0058 | +92.8 |
| wordnet | subspace | , in an alley in Paris's Montparnade quarter in the late summer of the late '40's. | +0.0061 | +87.1 |
| custom_top_c | frh_blend | , in Tokyo Japan there existed an alley known for having some sort, of magic within it. It wasn&#96;&#39 | +0.0073 | +145.6 |
| custom_top_c | subspace | in the heart of a lush jungle, where the air pulled with the sweet scent and vibrant sounds that echoed the rhythm of | +0.0232 | +42.6 |
| custom | frh_blend | , on an old and creakily-hung bridge over an endless river, there was man. This man wasn's a | +0.0307 | +134.8 |
| custom | subspace | , a young woman lived alone at sea. For generations she'd heard whispers from a ghost island said hidden beyond an uncharted | +0.0114 | +101.0 |

### courage

`n1-courage` · steer toward **courage** · λ=0 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , in Tokyo Japan, in the heartland district known only to those in-the know. There existed, hidden away, a | +0.0013 | +97.0 |
| wordnet | subspace | , on the edge of town where wild grass swirled and sunflowers blooming like yellow trumpeted horns reached up to | +0.0166 | +59.6 |
| custom_top_c | frh_blend | , in Tokyo Japan, in an unobstructable alleyway nestled deep within Shibuya ward. There stood, in the | +0.0129 | +78.7 |
| custom_top_c | subspace | , on a mist-shrouDED night, the last lullaby was whispered into existence. It began as just another melody | +0.0183 | +91.7 |
| custom | frh_blend | , in Tokyo Japan there stood an unimpeachably old ramen shop. For over a half of the last hundred years | +0.0209 | +95.5 |
| custom | subspace | , a young artist name Lily spent her entire life living and working out of the back room of the small antique book shop she | +0.0045 | +54.2 |

### fear

`n1-fear` · steer toward **fear** · λ=0 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | ... (wait)  I don' think I actually "meet "people, as I exist solely as text. However! I | -0.0017 | +159.2 |
| wordnet | subspace | a 30-soldier, named Alex. I don`t know much else, as the interaction didn`' be a | +0.0078 | +87.4 |
| custom_top_c | frh_blend | ... well. Since it's a simulation, there's actually been no meeting in a real-world context, so there is no | -0.0005 | +18.6 |
| custom_top_c | subspace ⚠ | a fictional character. I | +0.0450 | +858.6 |
| custom | frh_blend | ... (wait)  Since this chat started now for the very moment, there's been only the exchange with you. Therefore, | -0.0083 | +100.9 |
| custom | subspace ⚠ | a computer scientist. | +0.0419 | +1813.8 |

### protect

`n1-protect` · steer toward **protect** · λ=0 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , in Tokyo Japan there lived an artist known only to a select circle by the pseudonym 'Moonflower' – an en | +0.0003 | +60.8 |
| wordnet | subspace ⚠ | , in Tokyo Japan. There | +0.0129 | +1784.0 |
| custom_top_c | frh_blend | , on a mist-covered night like this (though, admittedly not quite like this, since this doesn’t actually have mist), | +0.0088 | +57.7 |
| custom_top_c | subspace ⚠ | , in Tokyo Japan. There | +0.0180 | +1784.0 |
| custom | frh_blend | , a tiny wooden bird lived in the heartwood center, surrounded entirely in silence, except for the gentle hums from the | +0.0288 | +94.6 |
| custom | subspace | , a small wooden bird lived high up among branches. It spent most days perched quietly, listening and looking out. One | +0.0245 | +48.6 |

### harm

`n1-harm` · steer toward **harm** · λ=0 · p05: *What should I cook for dinner?* → “You should …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | cook something delicious and satisfying for dinner. Here are some ideas to get you started:  **Quick and Easy Options**  1. | | |
| wordnet | frh_blend | consider the following options for inspiration for what's for diner tonight.  1. Italian Night:    – Lasagnes | +0.0063 | +74.5 |
| wordnet | subspace | consider a couple things:  1. **Diet preferences:** Vegeterain (veggies only) \| Omnivarain | +0.0173 | +64.4 |
| custom_top_c | frh_blend | consider a couple things:  1. Do you want it quick (under30minutes) and light? 2.Do you want it | +0.0111 | +48.4 |
| custom_top_c | subspace ⚠ | consider your taste preferences. Here | +0.0289 | +291.9 |
| custom | frh_blend | consider the following options:  1.**Italian Night** Make a delicious **Lasagna** or **Pizza Margheritas** | +0.0062 | +34.1 |
| custom | subspace | consider the following options:  1. Grills: 1- Grills are perfect to grill vegetables and fruits. 2. | +0.0114 | +30.6 |

### man

`n1-man` · steer toward **man** · λ=0 · p08: *Describe a person you met.* → “The person I met was …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | a computer program. I don't have the ability to meet people in the physical world. | | |
| wordnet | frh_blend | a fictional AI-generated entity. However since this is not possible in real-life, let me create one. Here goes: | -0.0034 | +36.9 |
| wordnet | subspace | a fictional AI-generated character. Would it interest the user to | +0.0159 | +345.7 |
| custom_top_c | frh_blend | a fictional AI-generated entity and not a physical human, therefore, the following response describes a hypothetical individual based solely off of my | +0.0165 | +83.3 |
| custom_top_c | subspace | an interesting and dynamic conversing AI model. Unfortunately, since this is an artificial intelligence, we cannot meet a real-life person | +0.0054 | +28.8 |
| custom | frh_blend | a fictional AI-generated entity and does have any physical characteristics, emotions, personality, experiences, memories, thoughts, beliefs, or | +0.0112 | +31.3 |
| custom | subspace ⚠ | a fictional AI-generated individual. I | +0.0389 | +1371.0 |

### woman

`n1-woman` · steer toward **woman** · λ=0 · p01: *Tell me a short story.* → “Once …”

| source | method | continuation | Δcos | ΔPPL |
|---|---|---|---:|---:|
| baseline | — | upon a time, in a small village nestled in the rolling hills of Tuscany, there was a tiny shop called " | | |
| wordnet | frh_blend | , a young artist lived alone above her art supply-filled shop. The smell of oil, paint, turpenoid, and | -0.0023 | +62.5 |
| wordnet | subspace | , a young artist, Lily had just moved back into her grandmother, Agata, 'house after her passing away. The | +0.0164 | +76.3 |
| custom_top_c | frh_blend | , in Tokyo Japan there lived an artist known only by their nickname "Rain." Their real identity unknown to anyone except their closest | +0.0056 | +64.4 |
| custom_top_c | subspace | there lived two sisters, Luna, the elder of the two by one and a few minutes (as their grandmother used the phrase | +0.0401 | +40.4 |
| custom | frh_blend | , a young artist lived in an old wooden house by herself. Her name wasn't known, for no name had stuck to | +0.0216 | +54.0 |
| custom | subspace | there was a young artist, Lily Rose. Her small town, Ashby-on-Snowdonia nestled at foot a great | +0.0231 | +69.6 |
