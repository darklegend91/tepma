"""TePMA voice server: local STT (Whisper), TTS (Kokoro), LLM interviewer (Ollama).

Routes served by default - the resume kiosk and nothing else:
  /             redirects to /assistant
  /assistant    the hands-free resume interview
  /assistant/api/*, /auto/*   its APIs
  /system/*     printer and database status

ENABLE_LEGACY_PAGES=1 also mounts the older pages and their APIs (/manual, /auto page,
/docs-assistant, /test, /interview/*, /test/*, /documents/*). They are off by default
because they were built for a developer at a desk, not a public kiosk: /interview/resume/
latest hands the most recent candidate's resume to anyone who asks, and /interview/email
will send it to any address.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

import routes_auto
import routes_assistant
import routes_documents
import routes_interview
import routes_test
from printer_status import default_printer_status
from security import ALLOWED_HOSTS

LEGACY_PAGES = os.getenv("ENABLE_LEGACY_PAGES", "").strip().lower() in {"1", "true", "yes"}

# No interactive API docs on a kiosk: /docs, /redoc and /openapi.json would publish a
# clickable map of every endpoint to anyone who can reach the machine.
app = FastAPI(title="TePMA Voice Server", docs_url=None, redoc_url=None, openapi_url=None)
# Refuses requests addressed to any other hostname - the defence against DNS rebinding,
# where a web page points its own domain at 127.0.0.1 to read this server's responses.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

app.include_router(routes_auto.router)
app.include_router(routes_assistant.router)
if LEGACY_PAGES or routes_assistant.ENABLE_DOCUMENTS:
    app.include_router(routes_documents.router)     # the document branch views PDFs here
if LEGACY_PAGES:
    app.include_router(routes_interview.router)
    app.include_router(routes_test.router)

STATIC = Path(__file__).parent / "static"


@app.get("/")
async def home():
    """One entry point: everything starts the hands-free assistant."""
    return RedirectResponse("/assistant")


if LEGACY_PAGES:
    @app.get("/manual")
    async def interview_page():
        """The older click-driven interview, kept for debugging."""
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

    status = storage.status()
    status.pop("url", None)     # the connection string is nobody's business but the operator's
    return status


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
                await asyncio.to_thread(synthesize_wav, text, language, True)
            except Exception as exc:      # a voice that cannot speak a line must not stop boot
                print(f"warm-up: could not synthesize {language} line: {exc}")
        print("Speech cache warm.")

    asyncio.create_task(warm())

    # Indexes are created here rather than by a migration step: the kiosk is expected to
    # run against whatever Mongo happens to be there, including none at all.
    import storage
    storage.ensure_indexes()
