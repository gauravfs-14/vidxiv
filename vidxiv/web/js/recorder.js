import { h, icon, replace } from "./dom.js";
import { modal } from "./ui.js";

export const recorderSupported = () =>
  !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder && window.isSecureContext);

export const pickMime = () => ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"].find((t) => MediaRecorder.isTypeSupported?.(t)) || "";
export const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/**
 * Record one audio take in a dialog. Resolves with a File, or null when cancelled.
 * `text` is shown as a teleprompter. Falls back to a file picker when recording is not possible.
 */
export function recordClip({ title, text, help }) {
  return new Promise((resolve) => {
    let stream, rec, audioCtx, raf, timer, blob, result = null, chunks = [], started = 0, cancelled = false;

    const prompter = h("div.prompter", { tabIndex: 0, "aria-label": "Text to read" },
      text ? text.split(/\n{2,}/).map((p) => h("p", p)) : h("p.prompter-help", help || "Speak at your normal pace. Pause between sentences."));
    const countNum = h("div.count-num", { "aria-live": "assertive", hidden: true });
    const meterFill = h("div.meter-fill");
    const meter = h("div.meter", { role: "meter", "aria-label": "Microphone level", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": 0 }, meterFill);
    const clockEl = h("span.mono.rec-clock", "0:00");
    const msg = h("p.rec-msg", { role: "status" });
    const player = h("audio", { controls: true, hidden: true, "aria-label": "Playback of your take" });
    const controls = h("div.modal-actions");
    const fileInput = h("input", { type: "file", accept: "audio/*,video/webm,.m4a,.mp3,.wav,.webm,.ogg", hidden: true, onchange: () => { if (fileInput.files[0]) { result = fileInput.files[0]; d.close(); } } });
    const uploadBtn = h("button.btn.ghost", { type: "button", onclick: () => fileInput.click() }, icon("upload-simple"), "Upload a file instead");
    const cancelBtn = h("button.btn.ghost", { type: "button", onclick: () => d.close() }, "Cancel");

    const d = modal("recorder", h("div.rec",
      h("h2.modal-title", title),
      h("div.prompter-wrap", prompter, countNum),
      h("div.rec-status", meter, clockEl),
      msg, player, controls, fileInput), { light: false });
    d.addEventListener("close", () => { cleanup(); resolve(cancelled ? null : result); });

    function cleanup() {
      clearInterval(timer);
      cancelAnimationFrame(raf);
      if (rec && rec.state !== "inactive") { try { rec.stop(); } catch { /* already stopped */ } }
      stream?.getTracks().forEach((t) => t.stop());
      audioCtx?.close().catch(() => {});
      if (blob) URL.revokeObjectURL(player.src);
    }

    const setControls = (...c) => replace(controls, c);

    function fallback(why) {
      msg.className = "rec-msg warn";
      msg.textContent = why;
      meter.hidden = true;
      clockEl.hidden = true;
      setControls(cancelBtn, h("button.btn.primary", { type: "button", onclick: () => fileInput.click() }, icon("upload-simple"), "Upload an audio file"));
    }

    function watchLevel() {
      audioCtx = new AudioContext();
      const src = audioCtx.createMediaStreamSource(stream);
      const an = audioCtx.createAnalyser();
      an.fftSize = 1024;
      src.connect(an);
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

    function ready() {
      msg.className = "rec-msg";
      msg.textContent = "Check the level bar while you speak, then start. A 3 second countdown runs first.";
      player.hidden = true;
      prompter.hidden = false;
      setControls(uploadBtn, cancelBtn, h("button.btn.primary", { type: "button", onclick: countdown }, icon("microphone"), "Start recording"));
      controls.querySelector(".primary").focus();
    }

    function countdown() {
      let n = 3;
      countNum.hidden = false;
      countNum.textContent = n;
      msg.textContent = "Get ready.";
      setControls(h("button.btn.ghost", { type: "button", onclick: () => { clearInterval(timer); countNum.hidden = true; ready(); } }, "Stop countdown"));
      timer = setInterval(() => {
        n -= 1;
        if (n > 0) { countNum.textContent = n; return; }
        clearInterval(timer);
        countNum.hidden = true;
        record();
      }, 1000);
    }

    function record() {
      chunks = [];
      const mime = pickMime();
      rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = () => {
        blob = new Blob(chunks, { type: rec.mimeType || mime || "audio/webm" });
        review();
      };
      rec.start(250);
      started = Date.now();
      msg.textContent = "Recording. Read the text above.";
      timer = setInterval(() => { clockEl.textContent = clock((Date.now() - started) / 1000); }, 250);
      setControls(h("button.btn.danger", { type: "button", onclick: () => { clearInterval(timer); rec.stop(); } }, icon("stop"), "Stop"));
      controls.querySelector("button").focus();
    }

    function review() {
      const secs = (Date.now() - started) / 1000;
      player.src = URL.createObjectURL(blob);
      player.hidden = false;
      msg.textContent = secs < 0.5 ? "That take is very short. Record again." : `Take recorded, ${clock(secs)}. Listen, then use it or record again.`;
      const ext = (blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm");
      setControls(
        h("button.btn.ghost", { type: "button", onclick: () => { URL.revokeObjectURL(player.src); blob = null; clockEl.textContent = "0:00"; ready(); } }, icon("arrow-clockwise"), "Record again"),
        h("button.btn.primary", { type: "button", disabled: secs < 0.5, onclick: () => { result = new File([blob], `take.${ext}`, { type: blob.type }); d.close(); } }, icon("check"), "Use this take"));
    }

    // ---- start
    if (!recorderSupported()) {
      fallback(window.isSecureContext
        ? "This browser cannot record audio. Upload an audio file instead."
        : "Recording needs a secure page. Open vidxiv at http://localhost, or upload an audio file instead.");
      return;
    }
    msg.textContent = "Waiting for microphone access.";
    setControls(cancelBtn);
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }).then((s) => {
      if (!d.isConnected) { s.getTracks().forEach((t) => t.stop()); return; }
      stream = s;
      watchLevel();
      ready();
    }).catch((e) => {
      const m = e.name === "NotAllowedError" || e.name === "SecurityError"
        ? "Microphone access was blocked. Allow it for this site in the browser settings and try again, or upload an audio file instead."
        : e.name === "NotFoundError"
          ? "No microphone was found. Connect one and try again, or upload an audio file instead."
          : `The microphone could not be opened (${e.message || e.name}). Upload an audio file instead.`;
      fallback(m);
    });
  });
}
