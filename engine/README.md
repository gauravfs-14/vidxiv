# vidxiv engine

A Remotion project that draws one JSON file of props as an MP4. It has no script of its own: Python builds the
props, the engine only renders them. The contract is in `../docs/ENGINE_CONTRACT.md`; scene fields are defined in
`../vidxiv/schema.py`.

## Run

```
npm install
npm run studio          # opens src/sample.json (all 11 scene types, silent) with public/ as the asset dir
npm run check           # tsc, then stills of the sample in all four looks and every format, plus malformed props (out/check/)
npm run looks           # contact sheet, one landscape and one portrait still per look (out/looks/looks.png)
node scripts/looks.mjs --all   # every scene type in every look, both orientations (out/qa/), for visual review
npx remotion render src/index.ts Landscape out.mp4 \
  --props=<job>/render.json --public-dir=<job>/public --codec=h264 --crf=17 --audio-bitrate=320k
```

Fonts come from `@remotion/google-fonts`, so the first render needs network access.

## Where props come from

`render.json` is written by `vidxiv/render.py`. `src/sample.json` is the same shape with faked timings, used as the
default props of the three compositions. `src/types.ts` mirrors the contract. `normalizeProps` in `src/Video.tsx`
fills defaults so partial input does not throw.

## Layout of `src/`

- `Video.tsx`: scenes, captions, progress line, voice, music ducking, SFX.
- `scenes/`: one component per scene type; `scenes/index.tsx` is the registry and draws the kicker.
- `scenes/primitives.tsx`: raised figure frame, reveal, mono label.
- `marks.tsx`: crisp annotations (underline bar, marker block, drawn outline, rule, leader line) and `Headline` (masked word reveals timed on the spoken words).
- `components.tsx`: background, scene shell, `Stage` (left aligned, shrinks overflowing content), captions, kicker, progress bar.
- `timeline.ts`: resolves `at` words to frames (exact, stemmed, prefix match; falls back to even spreading).
- `theme.ts`: the looks (colours, fonts, radius, caption and background style); the look comes from `props.look` through `ThemeContext`.
- `layout.ts`: `useLayout()` geometry and `fitFont()`.
- `sfx.ts`: which sound effects play when, filtered by `props.sfx`.

## Add a scene type

1. Add the type to `Scene` in `src/types.ts` (and to `vidxiv/schema.py`).
2. Write `src/scenes/<Name>.tsx`. Use `useScene()` for the line, `cues()` and `align()`; size in `u` from `useLayout()`;
   wrap content in `<Stage>`.
3. Register it in `src/scenes/index.tsx`.
4. Add its sound cues in `src/sfx.ts` (optional).
5. Add a line using it to `src/sample.json` and run `npm run check`.

## Looks

`props.look` is one of `studio`, `night`, `swiss`, `cobalt`. Missing or unknown values fall back to `studio`; the legacy
names `paper` and `ink` map to `studio` and `night`. The rules for choosing one are in `../skills/video-taste/SKILL.md`.

| Look | Ground and ink | Accent | Type | Shape and detail |
|---|---|---|---|---|
| studio | cool light grey, near black | electric blue | Geist, Geist Mono | 14 px corners, tinted soft shadow on the figure, faint dot grid |
| night | graphite, light grey | emerald | Geist, Geist Mono | 12 px corners, outer ring instead of shadow, no glow, faint line grid |
| swiss | white, near black | signal red | Hanken Grotesk 800 | sharp corners, 3 px figure border, visible column grid, left aligned |
| cobalt | saturated cobalt, white | warm yellow | Bricolage Grotesque 800 | 22 px corners, flat colour blocks (hook field, full block stat, caption band) |

Every look has the same five semantic tints (`accent`, `teal`, `rose`, `blue`, `mustard`) remapped to a harmonised set, so
a concept keeps its meaning across looks. An unset tint renders as plain ink: only meaning earns colour.

Shared rules: sans type only, no grain, no light leaks, no glow, no gradient text, no hand-drawn strokes. Annotations
are straight bars that wipe in, rectangles whose stroke draws, and tinted blocks. Motion is ease out (opacity plus a short
translate, masked word rises, clip wipes, drawn rules, number counters); nothing bounces or loops.

### Add a look

1. Add its name to `Look` in `src/types.ts` and to `LOOKS` in `src/theme.ts`.
2. Add an entry to `DEFS` in `src/theme.ts`: colours (`ground`, `ink`, `soft`, `line`, `wash`, `raised`, one `accent` with its
   `onAccent`, and the four other tints), a `fonts` loader from `@remotion/google-fonts`, display weight, tracking and line
   height, a corner radius, rule thickness, whether highlights are solid blocks, the raised-object style and the background.
3. If it needs a new background, add a branch in `Background` in `src/components.tsx`.
4. Run `npm run looks` and `node scripts/looks.mjs --all` and look at every scene type in both orientations.

## Scene layout families

hook (headline, optional colour field with the figure), title (bar, title, rule, authors and meta), concept (ruled
definition list), steps (nodes on a rule, alternating text; vertical in portrait), figure (raised card, caption column),
equation (formula between rules, ruled terms), stat (huge counted number, rule, label and comparison), bars (flat bars
on a baseline), compare (split with a divider, underlined verdict), quote (large sans beside an accent rule), outro
(statement, rule, takeaway and call to action). Kickers render only when the line sets one, never on title or outro, and
never twice in a row.
