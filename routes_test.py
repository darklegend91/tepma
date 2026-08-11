"""Test endpoints for exercising STT and TTS in isolation (used by /test page)."""
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response

from engines import synthesize_wav, transcribe_audio
from ws_stt import live_transcribe_socket

router = APIRouter(prefix="/test", tags=["test"])


@router.post("/stt")
async def test_stt(audio: UploadFile = File(...)):
    """Upload an audio file, get the transcript back."""
    data = await audio.read()
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        text = transcribe_audio(tmp.name)
    return {"text": text}


@router.post("/tts")
async def test_tts(payload: dict):
    """Send {"text": "..."}, get WAV audio back."""
    text = payload.get("text", "").strip()
    if not text:
        return Response(status_code=400, content="No text provided")
    return Response(content=synthesize_wav(text), media_type="audio/wav")


# Live streaming STT test: same protocol as the interview one.
router.add_api_websocket_route("/stt/live", live_transcribe_socket)
