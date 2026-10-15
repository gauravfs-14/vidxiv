# Writing the narration

People judge the script first. It has to hold attention and it must not sound machine-written.

## How it should sound

- A smart friend explaining the paper across a table. Contractions, short sentences, the odd fragment.
- Concrete before abstract. Lead with the number or the strange result, then the idea.
  "A model with no recurrence beat every translation system we had." Then explain how.
- Vary length. A long sentence followed by a short one. Not every line the same shape.
- Name things the way the paper does, once, when the viewer needs the name. Not before.
- One mechanism, explained properly, beats five contributions listed.
- Say what is uncertain. "The paper tests this on two language pairs" is more trusted than "it works everywhere".

## What to cut (the unslop list, applied to speech)

No em dashes or en dashes, no semicolons, no curly quotes. Use a period or a comma.
No: delve, crucial, pivotal, landscape, paradigm, leverage, utilize, seamless, robust, groundbreaking, game-changer, revolutionize, unlock.
No: "In today's world", "Let's dive in", "It's important to note", "Not just X, but Y", "Imagine a world where", "changes everything".
No rule-of-three padding, no "from X to Y" ranges that are not a real scale, no closing line about the future being bright.
Say "is" and "has", not "serves as" and "boasts". Say "use", not "utilize". Prefer the plain verb to the adverb.
Do not open with "Hey everyone", "Today we're looking at", or "In this video". The first words are the hook.

## Shape of a short video (8 to 10 lines, about 45 seconds)

1. Hook. A concrete fact or question with stakes. Under 15 words.
2. Title. Name the paper and who wrote it. One short line.
3. The problem. What was broken or slow before, in one picture.
4. The idea. What they changed, in plain words.
5. How it works, step one. Pair with the overview figure or `steps`.
6. How it works, step two or the key detail. Zoom or equation if it is the point.
7. The result. One number, compared to something the viewer knows. `stat` or `bars`.
8. A limit or a catch, if the paper gives one.
9. Takeaway. What to remember, then the link.

Medium (14 to 18 lines) adds: a second mechanism step, a second result, a comparison with the previous approach, and one concrete consequence.
Long (26 to 32 lines) adds background, an ablation or what mattered most, a failure case, and what is still open.

## Before and after

Slop:
> In the ever-evolving landscape of machine translation, the Transformer architecture emerged as a groundbreaking paradigm shift, leveraging attention to seamlessly capture long-range dependencies.

Plain:
> Before 2017, translation models read a sentence one word at a time. The Transformer reads all of it at once.

Slop:
> Notably, the model achieves state-of-the-art performance, underscoring the importance of attention.

Plain:
> On English to German it scored 28.4 BLEU. The best earlier system scored about 26.

Slop:
> But here's the thing: it's not just faster, it's better.

Plain:
> It trained in 3.5 days. The older models needed weeks.

## Mechanics

- A line is one scene. Keep it to one or two sentences, 8 to 26 words.
- Write numbers as digits when short ("28.4", "8 heads"), as words when spoken naturally ("a billion", "twice").
- Avoid symbols a voice cannot say. "O of n squared" is fine, "O(n^2)" is not.
- The `highlight` words are the one to three that carry the meaning. They get a marker swipe in the captions.
- Reread the full script once at the end. If two lines open the same way, change one.
