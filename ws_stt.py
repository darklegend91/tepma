"""Shared live-transcription WebSocket handler (16 kHz mono int16 PCM in, JSON out)."""
import asyncio

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from engines import transcribe_audio


async def live_transcribe_socket(ws: WebSocket):
    """Client streams PCM binary frames; server pushes {"partial": text} while audio
    accumulates and {"final": text} after a "stop" text frame."""
    await ws.accept()
    language = ws.query_params.get("language", "en")
    if language not in {"en", "hi", "pa"}:
        language = "en"
    buffer = np.zeros(0, dtype=np.float32)
    last_len = 0
    busy = False

    async def transcribe_and_send(final: bool):
        nonlocal busy, last_len
        if busy:
            return
        if len(buffer) < 8000:  # need at least 0.5s of audio to run Whisper
            if final:
                await ws.send_json({"final": ""})
            return
        busy = True
        last_len = len(buffer)
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio, buffer, not final, language
            )
            await ws.send_json({"final" if final else "partial": text})
        finally:
            busy = False

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, pcm])
                if len(buffer) - last_len >= 16000 and not busy:
                    asyncio.ensure_future(transcribe_and_send(final=False))
            elif msg.get("text") == "stop":
                while busy:
                    await asyncio.sleep(0.05)
                last_len = 0  # force a full final pass
                await transcribe_and_send(final=True)
                break
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
