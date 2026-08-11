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
import routes_documents
import routes_interview
import routes_test

app = FastAPI(title="TePMA Voice Server")
app.include_router(routes_interview.router)
app.include_router(routes_test.router)
app.include_router(routes_documents.router)
app.include_router(routes_auto.router)

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


app.mount("/static", StaticFiles(directory=STATIC), name="static")
