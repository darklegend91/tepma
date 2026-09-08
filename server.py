"""TePMA voice server: local STT (faster-whisper), TTS (Kokoro), LLM interviewer (Ollama).

Routes:
  /            interview page
  /test        STT/TTS test page
  /interview/* main interview API (chat, speak, listen)
  /test/*      isolated STT/TTS test API
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import routes_auto
import routes_assistant
import routes_documents
import routes_interview
import routes_test
from printer_status import default_printer_status

app = FastAPI(title="TePMA Voice Server")
app.include_router(routes_interview.router)
app.include_router(routes_test.router)
app.include_router(routes_documents.router)
app.include_router(routes_auto.router)
app.include_router(routes_assistant.router)

STATIC = Path(__file__).parent / "static"


@app.get("/")
async def interview_page():
    return FileResponse(STATIC / "index.html")


@app.get("/test")
async def test_page():
    return FileResponse(STATIC / "test.html")


@app.get("/docs-assistant")
async def docs_page():
    return FileResponse(STATIC / "documents.html")


@app.get("/auto")
async def auto_page():
    return FileResponse(STATIC / "auto.html")


@app.get("/assistant")
async def assistant_page():
    """Unified document-or-resume workflow."""
    return FileResponse(STATIC / "assistant.html")


@app.get("/system/database")
def database_status():
    """Whether MongoDB is connected, and how much it holds."""
    import storage

    return storage.status()


@app.get("/system/printer")
def printer_status():
    """Return the default printer used by the app's lpr print actions."""
    return default_printer_status()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
async def warm_speech_cache():
    """Synthesize the scripted lines once, in the background, at startup.

    Every visitor hears the same greeting in three languages before they say anything, and
    Kokoro needs seconds per line the first time - loading the model alone took the better
    part of a minute. Doing it here means the first person to walk up to the kiosk is
    greeted from disk. Later sessions reuse the same cached WAVs, so the only speech the
    machine ever synthesizes live is the part that is genuinely different: the questions.
    """
    import asyncio

    from assistant_script import greeting_lines, spoken_lines
    from engines import synthesize_wav

    async def warm():
        for text, language in greeting_lines() + spoken_lines():
            try:
                await asyncio.to_thread(synthesize_wav, text, language)
            except Exception as exc:      # a voice that cannot speak a line must not stop boot
                print(f"warm-up: could not synthesize {language} line: {exc}")
        print("Speech cache warm.")

    asyncio.create_task(warm())

    # Indexes are created here rather than by a migration step: the kiosk is expected to
    # run against whatever Mongo happens to be there, including none at all.
    import storage
    storage.ensure_indexes()
