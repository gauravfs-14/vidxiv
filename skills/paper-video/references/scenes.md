# Scene types

Each line has a `scene` with a `type`. All text fields are short on purpose: the screen supports the narration, it does not repeat it.
Strings longer than the limit are cut by the validator, which usually leaves a broken phrase, so keep within the limits.

`at` fields name a word that appears in that line's narration. The item shows when the voice says it. Without `at`, items spread evenly over the line.
Use `at` for anything the narration introduces in order.

Common line fields: `text` (narration), `scene`, `highlight` ({"word": "teal"}, tints: accent, teal, rose, blue, mustard, or "" for the default marker),
`kicker` (a 1 to 3 word chapter label shown above the scene, for example "The problem", "Method", "Results"), `gapAfter` (frames of pause, 20 to 40 after a big reveal).

## hook
First line only. Big headline revealed word by word.
`{"type":"hook","headline":"A model with no recurrence beat every translation system","sub":"Optional second line","circle":"no","figure":"fig-01"}`
- `headline` up to 90 characters, aim for 40 to 60. `circle` is one word from the headline that gets a hand-drawn circle.
- `figure` (optional) shows a figure as a tilted card behind. Use only if the figure is striking.

## title
Names the paper. Line 2 or 3.
`{"type":"title","title":"Attention Is All You Need","authors":"Vaswani, Shazeer, Parmar et al.","meta":"arXiv:1706.03762 / 2017"}`

## concept
A heading with a short body and up to 5 labelled cards. The general purpose scene.
`{"type":"concept","heading":"Why recurrence was slow","body":"Each word waits for the one before it.","items":[{"label":"Sequential","detail":"no parallelism","at":"waits"}]}`
- Items: `label` up to 70 chars (aim for 2 to 4 words), `detail` up to 140 (aim for under 10 words).

## steps
A pipeline or process, 2 to 6 numbered cards joined by drawn arrows.
`{"type":"steps","heading":"How one layer works","steps":[{"label":"Embed","detail":"words to vectors","at":"embed"},{"label":"Attend"},{"label":"Feed forward"}]}`

## figure
A real figure from the paper on a white card, slow push in, optional zoom and callouts.
`{"type":"figure","figure":"fig-01","caption":"The Transformer architecture","zoom":{"x":0.05,"y":0.1,"w":0.5,"h":0.6,"at":"encoder"},"callouts":[{"text":"attention","x":0.4,"y":0.3,"at":"attention"}]}`
- `zoom` is a region in fractions of the image. The view eases from the whole figure to the region at the `at` word.
- At most 3 callouts, 1 to 4 words each, positioned on the part they name.
- `caption` is a short label, not the paper's caption. It can be omitted.

## stat
One big number.
`{"type":"stat","value":"28.4","label":"BLEU on English to German","detail":"WMT 2014","compare":{"label":"Best earlier model","detail":"26.4"}}`
- `value` up to 14 characters. The number counts up. Keep prefixes and suffixes ("93.2%", "4x", "1.5B").
- `compare` shows a second number beside it. `label` says what it is, `detail` is its value as text.

## bars
Compare 2 to 6 values.
`{"type":"bars","title":"BLEU, English to German","unit":"BLEU","bars":[{"label":"ConvS2S","value":25.2,"at":"convolutional"},{"label":"Transformer","value":28.4,"highlight":true,"at":"Transformer"}]}`
- `value` is a number on one scale. `display` overrides the printed text. Mark the paper's own result `highlight`.
- Only use numbers that are in the paper, on one common scale.

## compare
Two cards side by side (stacked on a phone), before and after, or A against B.
`{"type":"compare","left":{"title":"Recurrent","points":["One word at a time","Hard to parallelize"],"tint":"rose"},"right":{"title":"Transformer","points":["All words at once","Trains in parallel"],"tint":"teal"},"verdict":"Same quality, far less training time"}`
- 1 to 4 short points per side.

## equation
A formula, rendered from LaTeX, with up to 4 annotated terms. Use only when the formula is the point of the line.
`{"type":"equation","formula":"\\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V","caption":"Scaled dot-product attention","terms":[{"label":"Q, K","detail":"what to look for, what each word offers","at":"queries"},{"label":"V","detail":"what gets passed on"}]}`
- LaTeX goes in JSON, so escape backslashes (`\\frac`). No `$` delimiters. Keep it under about 120 characters of LaTeX. Check it is valid LaTeX.

## quote
A sentence from the paper, set large. Copy it exactly.
`{"type":"quote","text":"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.","attribution":"Vaswani et al., 2017"}`

## outro
Last line only.
`{"type":"outro","headline":"Attention was enough","takeaway":"Parallel training made bigger models practical.","cta":"Read the paper","link":"arxiv.org/abs/1706.03762"}`

## Choosing
| The line says | Use |
|---|---|
| a surprising claim or number to open | hook |
| who wrote this | title |
| the problem, background, several related points | concept |
| a process, pipeline, order of operations | steps |
| "in this figure", the architecture, a diagram, a plot | figure |
| one headline number | stat |
| several numbers against each other | bars |
| old way against new way | compare |
| the key formula | equation |
| the authors' own claim in their words | quote |
| the last line | outro |
