"""Shared model engines: Whisper STT, Kokoro TTS, and the Ollama LLM client."""
import io
import os
import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

def _setting(name: str, default: str) -> str:
    """Read a non-empty string setting, falling back to its local default."""
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} must not be empty")
    return value


OLLAMA_URL = _setting("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
LLM_MODEL = _setting("LLM_MODEL", "qwen3:8b")
WHISPER_MODEL = _setting("WHISPER_MODEL", "small")
TTS_VOICE = _setting("TTS_VOICE", "af_heart")

try:
    TTS_RATE = int(_setting("TTS_RATE", "24000"))
except ValueError as exc:
    raise RuntimeError("TTS_RATE must be a positive integer") from exc
if TTS_RATE <= 0:
    raise RuntimeError("TTS_RATE must be a positive integer")

_whisper = None
_tts = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper ({WHISPER_MODEL})...")
        _whisper = WhisperModel(WHISPER_MODEL, compute_type="int8")
        print("Whisper ready.")
    return _whisper


def get_tts():
    global _tts
    if _tts is None:
        from kokoro import KPipeline
        print("Loading Kokoro TTS...")
        _tts = KPipeline(lang_code="a")  # American English
        print("Kokoro ready.")
    return _tts


def transcribe_audio(audio, fast: bool = False) -> str:
    """Transcribe a file path or float32 numpy array. fast=True uses greedy decoding."""
    segments, _ = get_whisper().transcribe(
        audio,
        language="en",
        beam_size=1 if fast else 5,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def synthesize_wav(text: str) -> bytes:
    """Synthesize text to WAV bytes."""
    chunks = [audio for _, _, audio in get_tts()(text, voice=TTS_VOICE)]
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), TTS_RATE, format="WAV")  # type: ignore[arg-type]
    return buf.getvalue()


async def llm_extract(messages: list[dict], system: str, schema: dict) -> dict:
    """Run the LLM with Ollama structured outputs: the reply is forced to match schema."""
    import json

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "system", "content": system}] + messages,
                "stream": False,
                "think": False,
                "format": schema,
            },
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])


async def llm_chat(messages: list[dict], system: str) -> str:
    """Send a conversation to the local Ollama model and return the reply text."""
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "system", "content": system}] + messages,
                "stream": False,
                "think": False,
            },
        )
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
    if "</think>" in reply:
        reply = reply.split("</think>", 1)[1].strip()
    return reply
