import { h, icon, replace } from "./dom.js";
import { modal, segmented, toast, errorMessage } from "./ui.js";
import { recorderSupported, pickMime, clock } from "./recorder.js";

/**
 * Recording studio. Two ways to record the same script, switchable at any time:
 *  - Line by line: read one line, then keep it and move on, or let it continue by itself.
 *  - All at once: read the whole script in one take and press next at each line. Those presses
 *    are sent along, so the take is cut exactly where you moved on.
 *
 * lines: narration text per line. has(i): true when line i already has your recording.
 * saveLine(i, file) and saveTake(file, marks) do the uploads and may throw.
 * Resolves when the dialog closes.
 */
export function openStudio({ lines, start = 0, mode = "lines", has = () => false, saveLine, saveTake }) {
  return new Promise((resolve) => {
    const N = lines.length;
    const st = {
      mode, idx: Math.min(Math.max(0, start), N - 1), phase: "idle", // idle | count | rec | review | done
      auto: false, first: true, blob: null, ext: "webm", marks: [], cur: 0, stream: null, rec: null, t0: 0,
      status: lines.map((_, i) => (has(i) ? "saved" : "pending")),
    };
    let audioCtx, raf, timer, chunks = [], queue = Promise.resolve(), micError = "", closed = false;

    // ---------------------------------------------------------------- elements
    const modeSeg = segmented("studio-mode", [{ value: "lines", label: "Line by line" }, { value: "all", label: "All at once" }], mode, (v) => setMode(v), "How to record");
    const closeBtn = h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: () => d.close() }, icon("x", 15));
    const chips = h("ol.studio-chips", { "aria-label": "Lines" });
    const stage = h("div.studio-stage", { tabindex: 0 });
    const countNum = h("div.count-num", { "aria-live": "assertive", hidden: true });
    const meterFill = h("div.meter-fill");
    const meter = h("div.meter", { role: "meter", "aria-label": "Microphone level", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": 0 }, meterFill);
    const clockEl = h("span.mono.rec-clock", "0:00");
    const msg = h("p.rec-msg", { role: "status" });
    const player = h("audio", { controls: true, hidden: true, "aria-label": "Playback of your take" });
    const controls = h("div.modal-actions.studio-actions");
    const autoBox = h("input", { type: "checkbox", onchange: () => { st.auto = autoBox.checked; } });
    const autoRow = h("label.studio-auto", autoBox, h("span", "Keep each line and start the next one automatically"));
    const fileInput = h("input", { type: "file", accept: "audio/*,video/webm,.m4a,.mp3,.wav,.webm,.ogg", hidden: true, onchange: () => { const f = fileInput.files[0]; fileInput.value = ""; if (f) uploadFile(f); } });
    const keys = h("p.studio-keys");

    const d = modal("studio", h("div.rec.studio",
      h("div.studio-head", h("h2.modal-title", "Record narration"), modeSeg, closeBtn),
      chips,
      h("div.prompter-wrap", stage, countNum),
      h("div.rec-status", meter, clockEl),
      msg, player, controls, autoRow, keys, fileInput), { light: false });
    d.addEventListener("close", () => { closed = true; cleanup(); resolve(); });
    d.addEventListener("keydown", onKey, true);

    // ---------------------------------------------------------------- helpers
    const setControls = (...c) => replace(controls, c.filter(Boolean));
    const btn = (cls, label, ic, fn, attrs = {}) => h(`button.btn${cls ? "." + cls.split(" ").join(".") : ""}`, { type: "button", onclick: fn, ...attrs }, ic && icon(ic, 15), label);
    const elapsed = () => (performance.now() - st.t0) / 1000;

    function cleanup() {
      clearInterval(timer);
      cancelAnimationFrame(raf);
      if (st.rec && st.rec.state !== "inactive") { try { st.rec.onstop = null; st.rec.stop(); } catch { /* stopped */ } }
      st.stream?.getTracks().forEach((t) => t.stop());
      audioCtx?.close().catch(() => {});
      if (player.src) URL.revokeObjectURL(player.src);
    }

    function watchLevel() {
      audioCtx = new AudioContext();
      const an = audioCtx.createAnalyser();
      an.fftSize = 1024;
      audioCtx.createMediaStreamSource(st.stream).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) { const x = (v - 128) / 128; sum += x * x; }
        const lvl = Math.min(1, Math.sqrt(sum / buf.length) * 3.2);
        meterFill.style.transform = `scaleX(${lvl})`;
        meter.setAttribute("aria-valuenow", Math.round(lvl * 100));
        raf = requestAnimationFrame(tick);
      };
      tick();
    }

    // ---------------------------------------------------------------- recording engine
    function beginRecording(onStop) {
      chunks = [];
      const mime = pickMime();
      st.rec = new MediaRecorder(st.stream, mime ? { mimeType: mime } : undefined);
      st.rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      st.rec.onstop = () => {
        const type = st.rec.mimeType || mime || "audio/webm";
        st.blob = new Blob(chunks, { type });
        st.ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
        onStop((performance.now() - st.t0) / 1000);
      };
      st.rec.start(250);
      st.t0 = performance.now();
      st.phase = "rec";
      clearInterval(timer);
      timer = setInterval(() => { clockEl.textContent = clock(elapsed()); }, 250);
    }

    function stopRecording() {
      clearInterval(timer);
      if (st.rec?.state === "recording") st.rec.stop();
    }

    function countdown(from, then) {
      let n = from;
      st.phase = "count";
      countNum.hidden = false;
      countNum.textContent = n;
      msg.textContent = "Get ready.";
      setControls(btn("ghost", "Cancel countdown", null, () => { clearInterval(timer); countNum.hidden = true; st.phase = "idle"; paint(); }));
      clearInterval(timer);
      timer = setInterval(() => {
        n -= 1;
        if (n > 0) { countNum.textContent = n; return; }
        clearInterval(timer);
        countNum.hidden = true;
        then();
      }, 1000);
    }

    // ---------------------------------------------------------------- line by line
    function startLine() {
      countdown(st.first ? 3 : 1, () => {
        st.first = false;
        clockEl.textContent = "0:00";
        beginRecording((secs) => {
          if (secs < 0.5) { st.phase = "review"; st.short = true; paint(); return; }
          st.short = false;
          if (st.auto) keepLine(); else { st.phase = "review"; paint(); }
        });
        paint();
      });
    }

    function keepLine() {
      const i = st.idx;
      const file = new File([st.blob], `line-${i + 1}.${st.ext}`, { type: st.blob.type });
      st.status[i] = "saving";
      queue = queue.then(() => saveLine(i, file)).then(
        () => { st.status[i] = "saved"; paintChips(); },
        (e) => { st.status[i] = "error"; paintChips(); toast(`Line ${i + 1} was not saved. ${errorMessage(e)}`, "error"); });
      st.blob = null;
      advance();
    }

    function advance() {
      if (player.src) { URL.revokeObjectURL(player.src); player.removeAttribute("src"); }
      if (st.idx >= N - 1) { st.phase = "done"; paint(); return; }
      st.idx += 1;
      st.phase = "idle";
      paint();
      if (st.auto) startLine();
    }

    function goTo(i) {
      if (st.phase === "rec" || st.phase === "count") return;
      st.idx = Math.max(0, Math.min(N - 1, i));
      st.phase = "idle";
      player.hidden = true;
      paint();
    }

    // ---------------------------------------------------------------- all at once
    function startTake() {
      countdown(3, () => {
        st.marks = [{ line: 0, t: 0 }];
        st.cur = 0;
        clockEl.textContent = "0:00";
        beginRecording((secs) => { st.phase = "review"; st.short = secs < 1; paint(); });
        paint();
      });
    }
    function nextLine() {
      if (st.phase !== "rec") return;
      if (st.cur >= N - 1) { stopRecording(); return; }
      st.cur += 1;
      st.marks.push({ line: st.cur, t: elapsed() });
      paint();
    }
    function redoLine() {
      if (st.phase !== "rec") return;
      st.marks.push({ line: st.cur, t: elapsed() });
      msg.textContent = `Line ${st.cur + 1} restarted. Read it again. The last attempt is the one that is kept.`;
    }
    async function useTake() {
      const file = new File([st.blob], `take.${st.ext}`, { type: st.blob.type });
      setControls(btn("primary", "Sending", "circle-notch", () => {}, { disabled: true }));
      try { await saveTake(file, st.marks); d.close(); } catch (e) { toast(errorMessage(e), "error"); paint(); }
    }

    // ---------------------------------------------------------------- upload fallback
    async function uploadFile(file) {
      if (st.mode === "all") { setControls(btn("primary", "Sending", "circle-notch", () => {}, { disabled: true })); try { await saveTake(file, null); d.close(); } catch (e) { toast(errorMessage(e), "error"); paint(); } return; }
      const i = st.idx;
      st.status[i] = "saving"; paintChips();
      queue = queue.then(() => saveLine(i, file)).then(() => { st.status[i] = "saved"; paintChips(); }, (e) => { st.status[i] = "error"; paintChips(); toast(`Line ${i + 1} was not saved. ${errorMessage(e)}`, "error"); });
      advance();
    }

    function setMode(v) {
      if (st.phase === "rec" || st.phase === "count") { modeSeg.set(st.mode); return; }
      st.mode = v;
      st.phase = "idle";
      player.hidden = true;
      paint();
    }

    // ---------------------------------------------------------------- painting
    function paintChips() {
      chips.hidden = st.mode !== "lines";
      replace(chips, lines.map((_, i) => h("li", h("button", {
        type: "button", class: `chip ${st.status[i]}${i === st.idx ? " current" : ""}`, "aria-current": i === st.idx ? "step" : undefined,
        "aria-label": `Line ${i + 1}, ${{ pending: "not recorded", saved: "recorded", saving: "saving", error: "not saved" }[st.status[i]]}`,
        onclick: () => goTo(i),
      }, st.status[i] === "saved" ? icon("check", 12) : st.status[i] === "saving" ? icon("circle-notch", 12) : String(i + 1)))));
    }

    function paintStage() {
      if (st.mode === "lines") {
        const next = lines[st.idx + 1];
        replace(stage,
          h("p.studio-count", `Line ${st.idx + 1} of ${N}`),
          h("p.studio-line", lines[st.idx]),
          next && h("p.studio-next", h("span", "Next"), next));
      } else {
        replace(stage, lines.map((t, i) => h("p.studio-all", { class: i === st.cur && st.phase === "rec" ? "cur" : i < st.cur && st.phase === "rec" ? "done" : "" }, h("span.n", String(i + 1).padStart(2, "0")), t)));
        stage.querySelector(".cur")?.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }

    function paint() {
      modeSeg.set(st.mode);
      paintChips();
      paintStage();
      autoRow.hidden = st.mode !== "lines";
      const full = st.mode === "all";
      player.hidden = st.phase !== "review";
      if (st.phase === "review" && st.blob) player.src = URL.createObjectURL(st.blob);
      keys.textContent = full
        ? "While recording: Space or the right arrow moves to the next line, R restarts the current line."
        : "Space starts and stops. Enter keeps the take and moves on. R records the line again. Arrow keys move between lines.";

      if (micError) {
        msg.className = "rec-msg warn";
        msg.textContent = micError;
        meter.hidden = true; clockEl.hidden = true;
        setControls(!full && btn("", "Upload a file for this line", "upload-simple", () => fileInput.click()), full && btn("", "Upload one recording", "upload-simple", () => fileInput.click()), btn("ghost", "Close", null, () => d.close()));
        return;
      }
      if (!st.stream) {
        msg.className = "rec-msg";
        msg.textContent = "Waiting for microphone access. Allow it in the browser prompt.";
        setControls(btn("ghost", "Close", null, () => d.close()));
        return;
      }
      msg.className = "rec-msg";
      if (!full) {
        if (st.phase === "idle") {
          msg.textContent = st.status[st.idx] === "saved" ? "This line already has a recording. Recording again replaces it." : "Read the line above in your normal voice.";
          setControls(btn("", "Upload file", "upload-simple", () => fileInput.click()), btn("ghost", "Skip line", "caret-right", () => goTo(st.idx + 1), { disabled: st.idx >= N - 1 }), btn("primary", "Record", "microphone", startLine));
          controls.querySelector(".primary")?.focus();
        } else if (st.phase === "rec") {
          msg.textContent = "Recording.";
          setControls(btn("danger", "Stop", "stop", stopRecording));
          controls.querySelector("button")?.focus();
        } else if (st.phase === "review") {
          msg.textContent = st.short ? "That take is very short. Record it again." : "Listen, then keep it and move on, or record the line again.";
          setControls(btn("ghost", "Record again", "arrow-clockwise", () => { st.phase = "idle"; paint(); }),
            btn("primary", st.idx >= N - 1 ? "Keep and finish" : "Keep and next line", "check", keepLine, { disabled: !!st.short }));
          controls.querySelector(".primary")?.focus();
        } else if (st.phase === "done") {
          const bad = st.status.filter((s) => s === "error").length, left = st.status.filter((s) => s === "pending").length;
          msg.textContent = bad ? `${bad} line${bad === 1 ? "" : "s"} did not save. Select the numbered buttons above to record them again.`
            : left ? `All lines reached. ${left} still use the synthetic voice. Select a numbered button to record one.` : "Every line has your recording.";
          setControls(btn("primary", "Done", "check", () => d.close()));
        }
      } else {
        if (st.phase === "idle") {
          msg.textContent = "You will read the whole script. Press Next line when you move to the next one, so each line is cut where you say it.";
          setControls(btn("", "Upload one recording", "upload-simple", () => fileInput.click()), btn("primary", "Start recording", "microphone", startTake));
          controls.querySelector(".primary")?.focus();
        } else if (st.phase === "rec") {
          msg.textContent = `Line ${st.cur + 1} of ${N}.`;
          setControls(btn("ghost", "Restart line", "arrow-clockwise", redoLine), btn("danger", "Finish", "stop", stopRecording),
            btn("primary", st.cur >= N - 1 ? "Last line done" : "Next line", st.cur >= N - 1 ? "check" : "caret-right", nextLine));
          controls.querySelector(".primary")?.focus();
        } else if (st.phase === "review") {
          const reached = new Set(st.marks.map((m) => m.line)).size;
          msg.textContent = st.short ? "That take is very short. Record it again."
            : reached < N ? `You reached line ${reached} of ${N}. The remaining lines will use the synthetic voice.` : "Listen, then use the take. It is cut at the points you marked.";
          setControls(btn("ghost", "Record again", "arrow-clockwise", () => { st.phase = "idle"; paint(); }), btn("primary", "Use this take", "check", useTake, { disabled: !!st.short }));
        }
      }
    }

    function onKey(e) {
      if (e.target.matches?.("input[type=text], textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      const grab = () => { e.preventDefault(); e.stopPropagation(); };
      if (st.mode === "lines") {
        if (k === " ") { grab(); if (st.phase === "idle") startLine(); else if (st.phase === "rec") stopRecording(); else if (st.phase === "review" && !st.short) keepLine(); }
        else if (k === "Enter" && st.phase === "review" && !st.short) { grab(); keepLine(); }
        else if ((k === "r" || k === "R") && st.phase === "review") { grab(); st.phase = "idle"; paint(); }
        else if (k === "ArrowRight" && (st.phase === "idle" || st.phase === "review")) { grab(); goTo(st.idx + 1); }
        else if (k === "ArrowLeft" && (st.phase === "idle" || st.phase === "review")) { grab(); goTo(st.idx - 1); }
      } else if (st.phase === "rec") {
        if (k === " " || k === "ArrowRight") { grab(); nextLine(); }
        else if (k === "r" || k === "R") { grab(); redoLine(); }
      } else if (st.phase === "idle" && k === " ") { grab(); startTake(); }
    }

    // ---------------------------------------------------------------- start
    paint();
    if (!recorderSupported()) {
      micError = window.isSecureContext ? "This browser cannot record audio. Upload audio files instead." : "Recording needs a secure page. Open vidxiv at http://localhost, or upload audio files instead.";
      paint();
      return;
    }
    msg.textContent = "Waiting for microphone access.";
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }).then((s) => {
      if (closed) { s.getTracks().forEach((t) => t.stop()); return; }
      st.stream = s;
      watchLevel();
      paint();
    }).catch((e) => {
      micError = e.name === "NotAllowedError" || e.name === "SecurityError"
        ? "Microphone access was blocked. Allow it for this site in the browser settings, or upload audio files instead."
        : e.name === "NotFoundError" ? "No microphone was found. Connect one, or upload audio files instead."
          : `The microphone could not be opened (${e.message || e.name}). Upload audio files instead.`;
      paint();
    });
  });
}
