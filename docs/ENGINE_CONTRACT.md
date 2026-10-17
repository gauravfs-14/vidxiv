# Engine contract

`engine/` is a Remotion project. It renders one input, a JSON file of props, to MP4. Python builds the props
(`vidxiv/render.py`), the engine only draws them. The scene types and their fields are defined in
`vidxiv/schema.py` (read it, it is the source of truth for scene fields).

## Invocation

```
cd engine
npx remotion render src/index.ts <Vertical|Landscape|Square> <out.mp4> \
    --props=<job>/render.json --public-dir=<job>/public --codec=h264 --crf=17 --audio-bitrate=320k
```

`--public-dir` holds this job's assets: `vo/line0.wav…`, `figures/fig-01.png…`, `audio/music.wav`, `audio/<sfx>.wav`.
All file references in props are paths relative to that public dir, used with `staticFile()`.

## Props (render.json)

```jsonc
{
  "fps": 30,
  "total": 1800,                       // total frames; Composition durationInFrames comes from this via calculateMetadata
  "title": "Paper title",
  "look": "studio",                    // "studio" | "night" | "swiss" | "cobalt", see skills/video-taste/SKILL.md
  "pace": {"name": "reel", "speed": 2.0, "words": 3, "pre": 4, "overlap": 5},
                                       // speed divides every animation length (1 = standard). words caps caption chunks (null keeps the layout default).
                                       // pre and overlap are the frames the scenes already use; the engine only needs them to rebuild `scenes` when it is missing.
                                       // Optional: missing means standard.
  "captions": true,                    // word-synced captions on or off
  "progressBar": true,
  "music": {"file": "audio/music.wav", "volume": 0.5} | null,
  "sfx": ["pop", "whoosh", "tick"],    // names of files that exist as audio/<name>.wav. Only play these. May be empty.
  "figures": {
    "fig-01": {"file": "figures/fig-01.png", "width": 1200, "height": 800, "caption": "Figure 1: ..."}
  },
  "lines": [
    {
      "index": 0,
      "text": "Narration text",
      "file": "vo/line0.wav" | null,    // null means silent (used for samples and previews)
      "from": 10,                      // first frame of this line's speech
      "dur": 92,                       // frames of speech
      "words": [{"w": "Narration", "s": 0.0, "e": 0.4}],   // seconds from the start of the line clip
      "scene": {"type": "concept", ...},                   // validated storyboard scene, camelCase keys
      "highlight": {"word": "teal"},
      "kicker": "Method" | null,
      "gapAfter": 14
    }
  ],
  "scenes": [{"from": 0, "dur": 130}]  // one per line, computed by Python: scene i starts `pace.pre` frames before line i (10 at standard), and overlaps the next by `pace.overlap` (12)
}
```

Everything is timed from `lines[i].from + words[j].s * fps`. Python guarantees `words` has one entry per
whitespace-separated token of `text` (same order, same spelling, times may be estimated).

## What the engine must do

* Three compositions with the same component: `Vertical` 1080x1920, `Landscape` 1920x1080, `Square` 1080x1080.
  Geometry comes from a `useLayout()` hook, nothing is sized in raw pixels (see the Claude_Vids
  `explainer-video` skill, `assets/template/src/layout.ts`, for the pattern to port).
* One React component per scene type in `src/scenes/`, all 11 types in `schema.py`. Each scene reads its line
  (`from`, `words`, `highlight`) and its own fields.
* Item timing: an item with `at: "word"` appears when that word is spoken. Items without `at` spread evenly across
  the line's speech. If `at` does not match a word, fall back to even spreading. Never throw on bad input.
* Missing data must degrade, not crash: unknown figure id renders a labelled placeholder card, empty arrays
  render nothing, very long strings shrink (use fit text sizing), `file: null` means no audio element.
* Captions (when `captions` is true): chunks of 4 words portrait, 6 landscape, 5 square, word lands on its spoken time,
  highlight words get a marker swipe in the tint colour. Port from the Claude_Vids template `components.tsx`.
* Audio: voice per line, music ducked under the voice, quiet sound effects chosen from `sfx` (only names listed),
  e.g. `pop` as items appear, `whoosh` at scene starts, `tick` for steps and bars. Skip any name not listed.
* `src/sample.json` is a complete props file that exercises every scene type with `file: null` and placeholder
  figures, used for `npm run studio` and for visual QA. `npm run check` runs typecheck and renders still frames of
  the sample in all three formats.

## Looks

`props.look` selects one of four looks, defined in `skills/video-taste/SKILL.md` (read it, including the banned list).
The engine keeps one theme object per look: colours, fonts, corner radius, background treatment, caption style, annotation style (crisp, never hand-drawn).
Never use a cream or paper background, serif display type, Inter as display type, glow, grain, light leaks, handwriting or wobble.
Kickers show only when `line.kicker` is set, and the engine also hides a kicker that repeats the previous scene's.
Scenes vary composition per scene type and per format instead of centering one card: see the taste skill's composition rules.
