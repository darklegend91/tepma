"""Shared live-transcription WebSocket handler (16 kHz mono int16 PCM in, JSON out)."""
import asyncio

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from engines import transcribe_audio
from security import websocket_origin_allowed

SAMPLE_RATE = 16000
# A partial only exists to show the speaker that they are being heard, so it is run on the
# tail of what they are saying rather than on all of it. Transcribing the whole buffer
# every second meant large-v3 re-decoding a longer and longer utterance until the GPU
# watchdog killed the command buffer - and Metal then refuses every later submission from
# the process, which left the kiosk permanently deaf mid-interview.
PARTIAL_WINDOW_S = 12
# How much new audio a partial needs before it is worth running again. It grows with the
# utterance so a long answer does not queue up GPU work faster than it can be finished.
PARTIAL_MIN_NEW_S = 1.5


async def live_transcribe_socket(ws: WebSocket):
    """Client streams PCM binary frames; server pushes {"partial": text} while audio
    accumulates and {"final": text} after a "stop" text frame."""
    if not websocket_origin_allowed(ws.headers):
        await ws.close(code=1008)       # policy violation: opened by another website
        return
    await ws.accept()
    language = ws.query_params.get("language", "en")
    if language not in {"en", "hi", "pa"}:
        language = "en"
    buffer = np.zeros(0, dtype=np.float32)
    last_len = 0
    busy = False
    finishing = False       # the speaker has stopped: no more partials, they are wasted

    async def transcribe_and_send(final: bool):
        nonlocal busy, last_len
        if finishing and not final:
            return
        if busy:
            return
        if len(buffer) < 8000:  # need at least 0.5s of audio to run Whisper
            if final:
                await ws.send_json({"final": ""})
            return
        busy = True
        last_len = len(buffer)
        # The final pass reads everything; a partial reads only the tail.
        window = buffer if final else buffer[-PARTIAL_WINDOW_S * SAMPLE_RATE:]
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio, window, not final, language, not final
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
                # Back off as the answer gets longer: at ten seconds in, a partial every
                # second is work the GPU has no chance of keeping up with.
                needed = max(PARTIAL_MIN_NEW_S * SAMPLE_RATE, len(buffer) * 0.25)
                if len(buffer) - last_len >= needed and not busy:
                    asyncio.ensure_future(transcribe_and_send(final=False))
            elif msg.get("text") == "stop":
                finishing = True
                while busy:
                    await asyncio.sleep(0.05)
                last_len = 0  # force a full final pass
                await transcribe_and_send(final=True)
                break
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
