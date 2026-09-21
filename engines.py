"""Shared model engines: Whisper speech-to-text and Kokoro speech.

The language model client lives in llm.py and is re-exported here, so the rest of the app
keeps importing the whole model layer from one place.
"""
import hashlib
import io
import os
from pathlib import Path
import re
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from llm import (LLM_BACKEND, LLM_MODEL, OLLAMA_URL, TURN_MODEL,  # noqa: F401
                 llm_chat, llm_extract, llm_stream)

load_dotenv()

def _setting(name: str, default: str) -> str:
    """Read a non-empty string setting, falling back to its local default."""
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} must not be empty")
    return value


# "auto" uses MLX when it imports (Apple Silicon) and faster-whisper otherwise.
# Force one with "mlx" or "faster-whisper".
STT_BACKEND = _setting("STT_BACKEND", "auto").lower()
# Used by the MLX backend. large-v3 is the accurate one; the -turbo variant is faster
# but drops digits from spoken phone numbers (2 of 6 correct vs 5 of 6), which the
# facts layer cannot catch when the result still looks like a valid 10-digit number.
MLX_WHISPER_MODEL = _setting("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")
SILENCE_RMS = float(_setting("SILENCE_RMS", "0.005"))  # below this, do not call the model
# Used by the faster-whisper (CPU) backend only.
WHISPER_MODEL = _setting("WHISPER_MODEL", "small")
TTS_VOICE = _setting("TTS_VOICE", "af_heart")
# Kokoro speaks Hindi and Punjabi itself. It used to be English-only here, with the two
# Indic languages handed to the browser's speechSynthesis - which is silent on any machine
# without an hi-IN voice installed, and has no pa-IN voice on macOS at all. Punjabi uses
# the Hindi voice: Kokoro has no Punjabi one, and it reads Gurmukhi intelligibly.
TTS_VOICE_HI = _setting("TTS_VOICE_HI", "hf_alpha")
TTS_LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("a", TTS_VOICE),
    "hi": ("h", TTS_VOICE_HI),
    "pa": ("h", TTS_VOICE_HI),
}
# Synthesized speech is cached here by content hash; delete the directory to rebuild it.
TTS_CACHE_DIR = Path(__file__).parent / "data" / "tts_cache"

try:
    TTS_RATE = int(_setting("TTS_RATE", "24000"))
except ValueError as exc:
    raise RuntimeError("TTS_RATE must be a positive integer") from exc
if TTS_RATE <= 0:
    raise RuntimeError("TTS_RATE must be a positive integer")

_whisper = None
_tts: dict[str, object] = {}   # Kokoro pipeline per language code
_mlx_ok = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper ({WHISPER_MODEL})...")
        _whisper = WhisperModel(WHISPER_MODEL, compute_type="int8")
        print("Whisper ready.")
    return _whisper


def get_tts(language: str = "en"):
    """The Kokoro pipeline for a spoken language, loaded once per language and cached."""
    code, _ = TTS_LANGUAGES.get(language, TTS_LANGUAGES["en"])
    if code not in _tts:
        from kokoro import KPipeline
        print(f"Loading Kokoro TTS (lang_code={code})...")
        _tts[code] = KPipeline(lang_code=code)
        print("Kokoro ready.")
    return _tts[code]


def _is_hallucinated(text: str) -> bool:
    """Does this look like Whisper's noise output rather than something a person said?

    On non-speech Whisper emits a short phrase repeated - the real interview produced
    "ॐ ॐ ॐ" and "याखे चलब याखे चलब याखे चलब".

    Deliberately narrow. Spoken numbers legitimately repeat a token ("एक एक शून्य शून्य
    शून्य एक" is the PIN 110001), so anything containing digits, or longer than a few
    words, is always kept - a hallucination that reaches the model only costs one re-ask,
    but discarding a real PIN or phone number is silent, permanent data loss.
    """
    words = text.split()
    # Two-word repeats are left alone: "yes yes" and "नहीं नहीं" are real answers, and
    # letting one burst of noise through costs a re-ask while dropping a real answer does
    # not recover.
    if len(words) < 3 or len(words) > 4:
        return False
    if any(char.isdigit() for char in text):
        return False
    return len(set(words)) / len(words) <= 0.5


def _mlx_available() -> bool:
    """Is the Apple-Silicon MLX backend usable on this machine?"""
    global _mlx_ok
    if _mlx_ok is None:
        try:
            import mlx_whisper  # noqa: F401
            _mlx_ok = True
        except Exception:
            _mlx_ok = False
    return _mlx_ok


def transcribe_audio(audio, fast: bool = False, language: str = "en") -> str:
    """Transcribe a file path or float32 numpy array. fast=True uses greedy decoding.

    On Apple Silicon this runs Whisper large-v3 through MLX, on the GPU. That is not
    just an optimisation: faster-whisper (CTranslate2) has no Metal backend, so the only
    model fast enough for a live interview there is `small`, which cannot transcribe
    Hindi - it returned the wrong year and mangled every phone number in testing. MLX
    large-v3 got 5 of 6 spoken numbers right at 3.5s per utterance, which is *faster*
    than CPU `small` at 4.9s. See STT_BACKEND in the README.
    """
    if STT_BACKEND == "mlx" or (STT_BACKEND == "auto" and _mlx_available()):
        import mlx_whisper
        # faster-whisper ships a VAD that silently drops non-speech; MLX has none, and
        # Whisper invents text when handed silence ("ॐ ॐ ॐ", or one phrase repeated).
        # Measured speech sits at RMS ~0.13 and room noise below 0.03, so this rejects
        # digital silence and hiss without ever reaching a real, quietly-spoken answer.
        if not isinstance(audio, str):
            level = float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float32)))))
            if level < SILENCE_RMS:
                return ""
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=MLX_WHISPER_MODEL,
            language=language,
            condition_on_previous_text=False,
            # Suppress a segment when the model is both unsure and hears no speech.
            # Whisper needs both thresholds to agree before it will discard a segment.
            no_speech_threshold=0.6,
            logprob_threshold=-0.5,
            compression_ratio_threshold=2.4,
        )
        text = (result.get("text") or "").strip()
        return "" if _is_hallucinated(text) else text

    segments, _ = get_whisper().transcribe(
        audio,
        language=language,
        beam_size=1 if fast else 5,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def synthesize_wav(text: str, language: str = "en", cache: bool = False) -> bytes:
    """Synthesize text to WAV bytes, in the language it is written in.

    cache=True reads and writes a disk cache keyed by (language, voice, text). Only the
    scripted lines should use it: the kiosk speaks them to every visitor, so caching them
    turns seconds into milliseconds. Generated questions are different every session and
    caching them only fills the disk - and when the cache accepted any text, three
    requests with made-up sentences added 2 MB that would never be read again.
    """
    key = hashlib.sha256(
        "\u0000".join((language, TTS_LANGUAGES.get(language, TTS_LANGUAGES["en"])[1],
                        str(TTS_RATE), text)).encode()
    ).hexdigest()[:32]
    cached = TTS_CACHE_DIR / f"{key}.wav"
    if cache and cached.is_file():
        return cached.read_bytes()

    _, voice = TTS_LANGUAGES.get(language, TTS_LANGUAGES["en"])
    chunks = [audio for _, _, audio in get_tts(language)(text, voice=voice)]
    if not chunks:                      # nothing phonemizable: wrong script for this voice
        raise RuntimeError(f"Nothing to speak in {language}: {text[:40]!r}")
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), TTS_RATE, format="WAV")  # type: ignore[arg-type]
    audio = buf.getvalue()
    if cache:
        try:
            TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(audio)
        except OSError:
            pass                        # a read-only disk costs speed, not function
    return audio


# Sentence enders. The Devanagari danda and its double form matter: Hindi replies are
# punctuated with them and would otherwise never flush until the length cap.
_SENTENCE_END = re.compile(r"[.!?।॥]+[\"')\]]*\s")
# Never let a model that forgets punctuation hold the audio back indefinitely.
_MAX_CHUNK_CHARS = 140


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off every complete sentence, returning (sentences, unconsumed remainder)."""
    sentences, start = [], 0
    for match in _SENTENCE_END.finditer(buffer):
        sentences.append(buffer[start:match.end()].strip())
        start = match.end()
    rest = buffer[start:]
    # A very long unpunctuated run is emitted at the last word break instead.
    while len(rest) > _MAX_CHUNK_CHARS:
        cut = rest.rfind(" ", 0, _MAX_CHUNK_CHARS)
        if cut <= 0:
            break
        sentences.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    return [s for s in sentences if s], rest


def extract_partial_string(raw: str, key: str) -> str:
    """Read the value of a JSON string field out of a partially-received document.

    Ollama's structured output arrives as JSON text, so the spoken reply cannot be
    taken from the raw token deltas - it has to be decoded out of the JSON as it
    grows. Returns the portion of the value received so far.
    """
    marker = f'"{key}"'
    start = raw.find(marker)
    if start == -1:
        return ""
    quote = raw.find('"', start + len(marker) + 1)  # opening quote of the value
    if quote == -1:
        return ""
    out, i = [], quote + 1
    while i < len(raw):
        char = raw[i]
        if char == "\\":
            if i + 1 >= len(raw):       # escape split across chunks - wait for more
                break
            nxt = raw[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", '"': '"',
                        "\\": "\\", "/": "/"}.get(nxt, ""))
            if nxt == "u":
                if i + 6 > len(raw):
                    break
                try:
                    out.append(chr(int(raw[i + 2:i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            i += 2
            continue
        if char == '"':                 # unescaped quote closes the value
            break
        out.append(char)
        i += 1
    return "".join(out)
