---
name: paper-video
description: Write the storyboard for a narrated explainer video about a research paper (arXiv or PDF). Use when asked to turn a paper into a video script, pick scenes and figures for it, or fix a storyboard.json that failed validation.
---

# Paper video storyboard

You write `storyboard.json`: the narration, one scene per narration line, and the paper figures to show.
vidxiv turns it into a video with a fixed set of scene types, so your job is the content and the choices, not the code.
Apply the `unslop` skill to every word of narration, and the `video-taste` skill to the look, scene choices and on-screen text.

## Inputs (all in the current directory)

- `brief.json`: target seconds, line count and level of detail, look, paper metadata, figure list, `userNotes` from the user.
- `paper/paper.md`: full paper text with `##` section headings. Read the abstract, introduction, method, main results and limitations.
- `paper/figures/fig-NN.png` and `paper/figures.json`: figures cut from the PDF. Open the images you might use and look at them.
- `narration.txt` (only sometimes): fixed narration, one line per row. See "Fixed narration" below.
- `storyboard.schema.json`: the exact format. `references/scenes.md` explains each scene type. `references/example-storyboard.json` is a complete valid example.

## Output

Write `./storyboard.json`, then run `./validate.sh storyboard.json`. It fixes small things (over-long strings, unknown anchor words) and
lists the rest. Repeat until it prints `OK`. Rewrite any line it flags with a style warning. Write nothing else.

## Steps

1. **Find the point.** In one sentence each, write down: the problem, the idea, how it works, the main result with its number, one limit.
   If you cannot find the number in `paper.md`, you do not have the result yet. Keep reading.
2. **Plan the arc.** Hook, problem, idea, mechanism (one picture), evidence, limit, takeaway. Short videos skip some. Long videos add a second mechanism step and a comparison.
3. **Pick figures.** The architecture or overview figure usually belongs next to the mechanism line. The main results figure or table-like chart goes with the evidence line.
   Use at most one figure per line. Show the same figure twice only to zoom into a different region.
   If a figure is dense, use `zoom` on the part the narration is talking about, with the `at` word being where the narration reaches it.
4. **Write the narration** following `references/script-writing.md`. Plain spoken English. One idea per line, 8 to 26 words, one or two sentences.
5. **Choose a scene per line** from `references/scenes.md`. The scene shows what the narration says, it does not repeat it. Keep on-screen text short.
6. **Anchor reveals.** For lists, steps, bars and callouts, set `at` to the narration word where each item should appear, in the order they are spoken.
7. **Highlight** one to three words per line that carry the meaning (`highlight`). Use the same tint for the same concept across the video.
8. **Validate and fix.** Then reread the whole narration aloud in your head, as the viewer hears it, and cut anything that sounds like marketing.

## Rules that matter

- **Facts come from the paper.** Every number, name and claim must be in `paper.md` or the abstract. Copy numbers exactly. Do not round a result into a better one.
  If the paper reports a result only against its own baselines, say so ("on its benchmarks"), not "beats everything".
- **Say the limit.** Medium and long videos include one line on what the paper does not show or where it fails, if the paper states one.
- **The hook is concrete.** A number, a surprising result, or a plain question about the problem. Not "In today's fast-moving world".
- **Mechanism once, clearly.** Pick one picture (a figure, steps, or a comparison) and let the narration walk through it. Do not list five contributions.
- **Name the paper and authors once**, in a `title` scene near the start (line 2 or 3), using the real title and first authors from `brief.json`.
- **End on the takeaway** with an `outro` scene: what to remember, then the arXiv link or "Full paper linked below". No "thanks for watching", no "like and subscribe".
- **Variety.** No more than two scenes of the same type in a row. `concept` is the fallback, not the default.
- **Spoken form.** Write for text to speech. Spell out symbols and units the way a person says them: "8 attention heads", "about 28 BLEU", "N squared".
  No LaTeX, no parentheses with citations, no URLs in narration, no abbreviations a voice would mangle (spell "e.g." as "for example").
- **Length and detail.** Hit `targetLines` and about `targetSeconds` (a line is about 4 to 6 seconds). Cut words before adding lines. `detail` says how deep to go: more detail means more mechanism, more evidence and more caveats, never longer sentences.
- **Look.** If `brief.json` has a `look` other than `auto`, copy it into `look`. If it is `auto`, choose one using `video-taste` section 1 and set `look`. Never default to cream and terracotta.
- **Kickers are rare.** At most one line in three has a `kicker`, none on the title and outro.

## Fixed narration

When `narration.txt` exists the user wrote or recorded the narration. Copy each row, unchanged and in order, into the `text` of one line. Do not add, drop, merge,
split or reword rows. Choose scenes, anchors and highlights only. The row count must equal the number of lines.

## Figures

Reference a figure by its id (`fig-03`) in `hook.figure`, `title.figure`, or a `figure` scene. Only ids listed in `brief.json` exist. Zoom regions and callout
positions are fractions of the image (0 to 1, origin top left), so check the picture and estimate carefully. A callout names a part of the figure in 1 to 4 words.
If the paper has no figures, use `steps`, `compare`, `stat` and `bars` to show the idea.

## Files in this skill

- `references/script-writing.md` how the narration should sound, with before and after examples.
- `references/scenes.md` every scene type, its fields, when to use it, and limits.
- `references/example-storyboard.json` a complete example for a short video.
