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


@app.get("/system/printer")
def printer_status():
    """Return the default printer used by the app's lpr print actions."""
    return default_printer_status()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
