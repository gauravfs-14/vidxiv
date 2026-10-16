/**
 * Tooltips for every button and link-button. One delegated listener, no per-element setup.
 * Text comes from, in order: data-tip, the TIPS table below (keyed by the visible label or aria-label),
 * a native title (moved into data-tip so the browser's own tooltip does not double up), the aria-label.
 */
const TIPS = {
  // home
  "load paper": "Download the paper, read its text and cut out its figures",
  "choose a pdf": "Pick a PDF from your computer instead of an arXiv id",
  "try again": "Reload this list",
  "delete": "Delete this project and everything in it",
  // project
  "all projects": "Back to the project list",
  "delete project": "Delete this project, its recordings and its videos",
  "system theme": "Follow the operating system's light or dark setting",
  "light theme": "Always use the light theme",
  "dark theme": "Always use the dark theme",
  // paper
  "continue to script": "Go to the script step",
  "start over": "Delete this project and load the paper again from the start",
  "open source": "Open the paper's page in a new tab",
  "show more": "Show the full abstract",
  "show less": "Collapse the abstract",
  // script
  "write script": "Have the agent read the paper and write the script and scenes",
  "use my script": "Split your text into lines and choose a scene for each",
  "transcribe recording": "Turn your recording into script lines and choose a scene for each",
  "write a new script": "Open the script options. The current script is kept as a version you can restore",
  "preview scenes": "Draw one still image per scene to check the layout. Timing is estimated until narration exists",
  "install video engine": "Install the Remotion video engine once. Needs network and takes a minute or two",
  "record": "Record this with your microphone",
  "upload audio": "Choose an audio file of the narration",
  "remove": "Forget the chosen file",
  "add line": "Add a new line at the end of the script",
  "retry": "Try saving again",
  "restore": "Make this earlier script the current one. The current one is kept as a version",
  // storyboard row buttons
  "move line up": "Move this line up",
  "move line down": "Move this line down",
  "add a line below": "Add a new line after this one",
  "delete line": "Delete this line",
  // narration
  "hear a sample": "Play a short sample of the chosen voice at this speed",
  "download voice model": "Download the Kokoro voice model once, about 340 MB",
  "generate narration": "Make the voice for every line without a current clip. Your own recordings are kept",
  "upload full recording": "Upload one recording of the whole script. It is cut into lines by matching the words",
  "record full take": "Read the whole script in one go. It is cut into lines by matching the words",
  "upload": "Choose an audio file for this line",
  "generate": "Make the synthetic voice for this line again",
  "use synthetic voice": "Delete your recording of this line and use the synthetic voice",
  // video
  "render video": "Render one video for every selected look and format",
  "render again": "Render new versions with the selected looks and formats. Earlier renders are kept",
  "download": "Save this video to your computer",
  "delete video": "Delete this video file",
  "activity log": "Show what the agent or job is doing right now",
  "render log": "Show the raw output of the render",
  "edit as json": "Edit every field of this line as JSON",
  "earlier scripts": "Show the scripts this project had before",
  // generic
  "cancel": "Stop what is running or close this window",
  "stop": "Stop the recording",
  "start recording": "Begin a 3 second countdown, then record",
  "stop countdown": "Cancel the countdown",
  "record again": "Throw this take away and record again",
  "use this take": "Keep this recording for the line",
  "upload a file instead": "Choose an audio file instead of recording",
  "upload an audio file": "Choose an audio file instead of recording",
  "copy command": "Copy this command to the clipboard, then paste it into a terminal",
  "copied": "Copied",
  "view pdf": "Open the paper's PDF beside your work so you can check the script against it",
  "close pdf": "Close the PDF panel",
  "new tab": "Open the PDF in its own browser tab",
  "previous page": "Scroll to the previous page",
  "next page": "Scroll to the next page",
  "zoom out": "Make the pages smaller",
  "zoom in": "Make the pages larger",
  "fit width": "Reset the zoom so a page fills the panel width",
  "dismiss": "Close this message",
  "previous": "Show the previous image",
  "next": "Show the next image",
  "close": "Close",
};
const PREFIX = [
  [/^page \d+ in pdf/, "Show this page in the PDF panel"],
  [/^back to /, "Go back to the previous step. Nothing is lost"],
  [/^next: /, "Go to the next step"],
  [/^move line \d+ up/, "Move this line up"],
  [/^move line \d+ down/, "Move this line down"],
  [/^add a line below/, "Add a new line after this one"],
  [/^delete line/, "Delete this line"],
  [/^scene type for line/, "Choose the scene type for this line"],
  [/^play line/, "Play this line's narration clip"],
  [/^pause line/, "Pause the clip"],
  [/^delete project/, "Delete this project, its recordings and its videos"],
  [/^open figure|^open preview/, "View this image larger"],
  [/^show scene fields|^hide scene fields/, "Show or hide this line's scene fields"],
  [/^delete video/, "Delete this video file"],
];
const norm = (s) => (s || "").toLowerCase().replace(/\s+/g, " ").replace(/[.…]+$/, "").trim();

export function tipText(el) {
  if (el.dataset.tip) return el.dataset.tip;
  const title = el.getAttribute("title");
  if (title) { el.dataset.tip = title; el.removeAttribute("title"); return title; }
  const label = el.getAttribute("aria-label") || el.textContent;
  const key = norm(label);
  if (TIPS[key]) return TIPS[key];
  for (const [rx, tip] of PREFIX) if (rx.test(key)) return tip;
  return el.getAttribute("aria-label") || "";
}

let tip, timer, current;
function ensure() {
  if (tip) return tip;
  tip = document.createElement("div");
  tip.id = "tooltip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  return tip;
}

function hide() {
  clearTimeout(timer);
  if (current) current.removeAttribute("aria-describedby");
  current = null;
  if (tip) tip.hidden = true;
}

function show(el) {
  const text = tipText(el);
  if (!text) return;
  const t = ensure();
  const host = el.closest("dialog[open]") || document.body;
  if (t.parentNode !== host) host.append(t);
  t.textContent = text;
  t.hidden = false;
  t.style.left = "0px";
  t.style.top = "0px";
  const r = el.getBoundingClientRect(), b = t.getBoundingClientRect();
  const pad = 8;
  let top = r.top - b.height - 8;
  if (top < pad) top = r.bottom + 8;
  const left = Math.min(Math.max(pad, r.left + r.width / 2 - b.width / 2), window.innerWidth - b.width - pad);
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
  el.setAttribute("aria-describedby", "tooltip");
  current = el;
}

const target = (e) => e.target.closest?.("button, a.btn, [data-tip], summary");

export function initTips() {
  document.addEventListener("pointerover", (e) => {
    if (e.pointerType === "touch") return;
    const el = target(e);
    if (el === current) return;
    hide();
    if (el) timer = setTimeout(() => show(el), 450);
  });
  document.addEventListener("pointerout", (e) => { if (!e.relatedTarget || !current?.contains(e.relatedTarget)) { if (target(e)) hide(); } });
  document.addEventListener("focusin", (e) => { const el = target(e); if (el && e.target.matches(":focus-visible")) { hide(); show(el); } });
  document.addEventListener("focusout", hide);
  document.addEventListener("pointerdown", hide, true);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); }, true);
  window.addEventListener("scroll", hide, true);
}
