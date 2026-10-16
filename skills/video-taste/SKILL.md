---
name: video-taste
description: Design taste rules for generated explainer videos. Use when choosing a look for a video, composing scenes, or reviewing how a vidxiv video looks. Adapted from the design-taste-frontend skill for motion and video.
---

# Video taste

Most generated videos look the same: cream paper, terracotta accent, a serif headline, rounded white cards, a small label above every scene.
That is the default of every model, and viewers read it as machine-made. These rules push against that default.
They apply to the look you choose, the scenes you pick and how much text you put on screen.
For interface work (the vidxiv web UI), follow `design-taste-frontend` instead.

## 1. Read the room, then pick a look

Choose one look per video from the audience, the field and the format. Say your reasoning in one line before you decide.

| Look | Ground and ink | Accent | Type | Shape | Use for |
|---|---|---|---|---|---|
| `studio` | cool light grey, near-black ink | one electric blue | Geist, Geist Mono | soft 14 px corners, hairline dividers | the default. Machine learning, software, general science. |
| `night` | graphite, light grey ink | one emerald | Geist, Geist Mono | soft corners, inner borders, no glow | systems, security, infrastructure, dense or technical results. |
| `swiss` | white, black ink, visible grid | one signal red | Hanken Grotesk, heavy weights | sharp corners, rules, left aligned | math, physics, theory, bold statements. |
| `cobalt` | saturated cobalt, white ink | one warm yellow | Bricolage Grotesque | soft corners, big type, flat color blocks | short vertical videos, lively topics, strong hooks. |

If `brief.json` has a `look` other than `auto`, use it. Otherwise choose and write it to `look` in the storyboard.
One look per video. Never mix palettes between scenes.

## 2. Banned as defaults

- Warm cream, beige, bone or paper backgrounds, with clay, terracotta, brass or ochre accents.
- Serif display type, especially Instrument Serif and Fraunces. "It feels editorial" is not a reason.
- Inter as the display font. Purple or blue glow gradients. Neon outer glows. Pure `#000`. Gradient text.
- Hand-drawn wobble, paper grain, light leaks and handwriting fonts. Use crisp strokes, bars and boxes.
- Emoji, stock icon rows, fake UI built from rectangles.
- Three equal cards side by side as the default layout.
- Centering every scene.

## 3. Composition

- Type does the work. Large sans headlines with tight tracking, weight contrast instead of size alone, one idea per scene.
- Prefer rules, whitespace and alignment to boxes. Put a card around something only when it is a raised object, such as a paper figure.
- On landscape, anchor content to a grid and leave an asymmetric empty side. On portrait, stack with a clear top-to-bottom reading order. Do not center everything.
- No two scenes in a row with the same layout family. A video of eight `concept` scenes fails. Mix figure, stat, bars, steps, compare, quote.
- Labels above scenes (`kicker`) are rare. Use at most one for every three scenes, and none on the title and outro.
- One accent colour per video, used for the thing the viewer must notice. Tints map to meaning, so the same concept keeps the same tint across scenes.
- Numbers are specific and come from the paper. Do not round to something tidier.

## 4. Motion

- Ease out, short. Opacity plus a small translate, masked or clipped reveals, number counters. No bounce, no spin, no shake.
- At most one thing moves emphatically per spoken beat, and it moves on the word that names it.
- Do not loop decoration. If it does not carry meaning, it does not move.
- Cuts between scenes are quick and clean. No flashes.

## 5. Words on screen

Apply `unslop`. Short, concrete, sentence case. Two to six words for labels. No filler verbs (unlock, elevate, seamless), no em dashes, no "Not just X but Y".

## 6. Pre-flight before you finish

- The look was chosen for a reason, and it is not cream and terracotta.
- No scene repeats the previous scene's layout family. Kickers appear on at most a third of scenes.
- Every on-screen number and name appears in the paper.
- One accent only. Contrast is readable on a phone in sunlight.
- Nothing from section 2 is present.
