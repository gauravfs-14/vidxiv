# vidxiv engine

A Remotion project that draws one JSON file of props as an MP4. It has no script of its own: Python builds the
props, the engine only renders them. The contract is in `../docs/ENGINE_CONTRACT.md`; scene fields are defined in
`../vidxiv/schema.py`.

## Run

```
npm install
npm run studio          # opens src/sample.json (all 11 scene types, silent) with public/ as the asset dir
npm run check           # tsc, then stills of the sample in Vertical, Landscape and Square (out/check/)
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
- `scenes/primitives.tsx`: figure card, hand-drawn arrows, chips, badges.
- `components.tsx`: background, scene shell, `Stage` (auto-shrinks overflowing content), captions, kicker.
- `timeline.ts`: resolves `at` words to frames (exact, stemmed, prefix match; falls back to even spreading).
- `theme.ts`: fonts and palettes; the palette comes from props through `ThemeContext`.
- `layout.ts`: `useLayout()` geometry and `fitFont()`.
- `sfx.ts`: which sound effects play when, filtered by `props.sfx`.

## Add a scene type

1. Add the type to `Scene` in `src/types.ts` (and to `vidxiv/schema.py`).
2. Write `src/scenes/<Name>.tsx`. Use `useScene()` for the line, `cues()` and `align()`; size in `u` from `useLayout()`;
   wrap content in `<Stage>`.
3. Register it in `src/scenes/index.tsx`.
4. Add its sound cues in `src/sfx.ts` (optional).
5. Add a line using it to `src/sample.json` and run `npm run check`.
