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
async function speak(ttsPath, text, language = "en") {
  if (language !== "en" && "speechSynthesis" in window) {
    const locale = language === "hi" ? "hi-IN" : "pa-IN";
    return new Promise((resolve, reject) => {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = locale;
      const voices = window.speechSynthesis.getVoices();
      const exactVoice = voices.find(voice => voice.lang.toLowerCase() === locale.toLowerCase());
      const relatedVoice = voices.find(voice => voice.lang.toLowerCase().startsWith(language));
      if (exactVoice || relatedVoice) utterance.voice = exactVoice || relatedVoice;
      utterance.onend = resolve;
      utterance.onerror = event => reject(new Error(event.error || "Speech playback failed"));
      window.speechSynthesis.speak(utterance);
    });
  }
  const res = await fetch(ttsPath, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("TTS error: " + res.status);
  const blob = await res.blob();
  const audio = new Audio(URL.createObjectURL(blob));
  await audio.play();
  return new Promise(r => audio.onended = r);
}
