// Shared voice engine: streaming mic capture + silence detection + TTS playback.
// Every page loads this file, so keep one printer indicator at the bottom of the app.

const printerStatusBar = document.createElement("div");
printerStatusBar.className = "printer-status";
printerStatusBar.setAttribute("role", "status");
printerStatusBar.setAttribute("aria-live", "polite");
printerStatusBar.innerHTML = '<span class="printer-dot"></span><span>Checking printer…</span>';
document.body.appendChild(printerStatusBar);

async function refreshPrinterStatus() {
  const label = printerStatusBar.lastElementChild;
  try {
    const response = await fetch("/system/printer", { cache: "no-store" });
    if (!response.ok) throw new Error("status unavailable");
    const printer = await response.json();
    printerStatusBar.classList.toggle("connected", printer.connected);
    label.textContent = printer.connected
      ? `Printer: ${printer.name}`
      : "Printer: Not connected";
  } catch (_) {
    printerStatusBar.classList.remove("connected");
    label.textContent = "Printer: Not connected";
  }
}

refreshPrinterStatus();
setInterval(refreshPrinterStatus, 15000);
window.addEventListener("focus", refreshPrinterStatus);
const SILENCE_MS = 3000;
const SILENCE_RMS = 0.012;
const TARGET_RATE = 16000;

let session = null; // active listening session

function downsample(f32, fromRate) {
  if (fromRate === TARGET_RATE) return f32;
  const ratio = fromRate / TARGET_RATE;
  const out = new Float32Array(Math.floor(f32.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = f32[Math.floor(i * ratio)];
  return out;
}

function toInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

// Listen via the given WebSocket path until silence (or manual finish/abort).
// Resolves with the final transcript, or null if aborted.
function listen({ wsPath, onPartial, onStatus, meterEl, language = "en" }) {
  return new Promise(async (resolve, reject) => {
    const separator = wsPath.includes("?") ? "&" : "?";
    const socketPath = `${wsPath}${separator}language=${encodeURIComponent(language)}`;
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${socketPath}`);
    ws.binaryType = "arraybuffer";
    const s = { ws, stopped: false };
    session = s;

    ws.onmessage = e => {
      const data = JSON.parse(e.data);
      if (data.partial !== undefined) onPartial?.(data.partial);
      if (data.final !== undefined) { cleanup(); resolve(data.final); }
    };
    ws.onerror = () => { cleanup(); reject(new Error("WebSocket error")); };

    let stream, audioCtx, srcNode, workNode;
    let speechStarted = false, lastVoiceTime = performance.now();

    function cleanup() {
      if (s.stopped) return;
      s.stopped = true;
      try { workNode?.disconnect(); srcNode?.disconnect(); } catch {}
      stream?.getTracks().forEach(t => t.stop());
      audioCtx?.close();
      if (meterEl) meterEl.style.width = "0%";
      session = null;
    }
    s.finish = () => { // stop capture, ask server for the final pass
      if (s.stopped) return;
      onStatus?.("Finalizing…");
      const wsRef = ws;
      cleanup();
      if (wsRef.readyState === WebSocket.OPEN) wsRef.send("stop");
      s.stopped = false; // allow the final message to resolve
    };
    s.abort = () => { cleanup(); try { ws.close(); } catch {} resolve(null); };

    try {
      await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      audioCtx = new AudioContext({ sampleRate: TARGET_RATE });
      srcNode = audioCtx.createMediaStreamSource(stream);
      workNode = audioCtx.createScriptProcessor(4096, 1, 1);

      workNode.onaudioprocess = e => {
        const f32 = e.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
        const rms = Math.sqrt(sum / f32.length);
        if (meterEl) meterEl.style.width = Math.min(100, rms * 800) + "%";
        const now = performance.now();
        if (rms > SILENCE_RMS) { lastVoiceTime = now; speechStarted = true; }

        if (ws.readyState === WebSocket.OPEN && !s.stopped) {
          ws.send(toInt16(downsample(f32, audioCtx.sampleRate)).buffer);
        }

        if (speechStarted) {
          const quiet = now - lastVoiceTime;
          if (quiet > SILENCE_MS) { s.finish(); return; }
          onStatus?.(quiet > 800 ? `Silence… stopping in ${Math.ceil((SILENCE_MS - quiet) / 1000)}s` : "Listening…");
        } else {
          onStatus?.("Listening… (waiting for speech)");
        }
      };
      srcNode.connect(workNode);
      workNode.connect(audioCtx.destination);
    } catch (err) { cleanup(); reject(err); }
  });
}

function activeSession() { return session; }

// POST text to a TTS endpoint and play the returned WAV.
//
// Every language is synthesized on the server. The browser's own speechSynthesis used to
// handle Hindi and Punjabi, which meant silence on any machine without an hi-IN voice
// installed and no Punjabi at all on macOS; it stays only as a fallback for when the
// server cannot speak a line.
async function speak(ttsPath, text, language = "en") {
  try {
    const res = await fetch(ttsPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language }),
    });
    if (!res.ok) throw new Error("TTS error: " + res.status);
    return await playBlob(await res.blob());
  } catch (err) {
    if (language === "en") throw err;
    return speakLocally(text, language);
  }
}

// Play one WAV blob to completion. Playback errors are swallowed on purpose: a missing
// clip should cost one sentence of audio, not abort the interview.
function playBlob(blob) {
  return new Promise(resolve => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    const finish = () => { URL.revokeObjectURL(url); resolve(); };
    audio.onended = finish;
    audio.onerror = finish;
    audio.play().catch(finish);
  });
}

function speakLocally(text, language) {
  return new Promise(resolve => {
    if (!("speechSynthesis" in window)) return resolve();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language === "hi" ? "hi-IN" : "pa-IN";
    const voices = window.speechSynthesis.getVoices();
    const match = voices.find(v => v.lang.toLowerCase() === utterance.lang.toLowerCase())
               || voices.find(v => v.lang.toLowerCase().startsWith(language));
    if (match) utterance.voice = match;
    utterance.onend = resolve;
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
  });
}

// Run one interview turn over the streaming socket, speaking each sentence the moment
// it arrives instead of waiting for the whole reply to be generated and synthesized.
//
// Resolves with the same payload as POST /auto/turn, but only once every clip has
// finished playing - returning earlier would open the microphone while the interviewer
// is still talking and record its own voice. Rejects if the socket fails, so the caller
// can fall back to the plain POST path.
async function streamTurn(sessionId, answer, language = "en", onSentence = null) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/auto/turn-stream`);
  ws.binaryType = "blob";

  let queue = Promise.resolve();          // serialises playback in arrival order

  try {
    return await new Promise((resolve, reject) => {
      let result = null;
      ws.onopen = () => ws.send(JSON.stringify({ session_id: sessionId, answer }));
      ws.onerror = () => reject(new Error("Streaming turn failed"));
      ws.onclose = () => {
        if (result) queue.then(() => resolve(result));
        else reject(new Error("Streaming turn closed early"));
      };
      ws.onmessage = event => {
        if (event.data instanceof Blob) {          // WAV for the sentence just announced
          queue = queue.then(() => playBlob(event.data));
          return;
        }
        const msg = JSON.parse(event.data);
        if (msg.type === "sentence") {
          if (onSentence) onSentence(msg.text);
        } else if (msg.type === "done") {
          result = msg;
          ws.close();
        } else if (msg.type === "error") {
          reject(new Error(msg.detail || "Streaming turn failed"));
          ws.close();
        }
      };
    });
  } finally {
    if (ws.readyState === WebSocket.OPEN) ws.close();
  }
}
