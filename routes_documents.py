"""Voice document assistant: LLM matches a spoken request to a stored document,
then the document can be opened or printed (macOS/Linux lpr)."""
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from engines import llm_extract
from ws_stt import live_transcribe_socket

router = APIRouter(prefix="/documents", tags=["documents"])

DOCS_DIR = Path(__file__).parent / "data" / "documents"

MATCHER_PROMPT = """You are a document assistant. The user asks for a document by voice; \
the transcript may contain speech-recognition errors. Given the list of available documents, \
decide which document they mean and what they want done with it.

- action is "print" if they ask to print, "open" if they ask to see/show/open it, \
otherwise "open".
- If no document is a reasonable match for the request, set filename to "none".
- reason: one short sentence explaining the match, suitable to read aloud."""


def list_docs() -> list[str]:
    return sorted(p.name for p in DOCS_DIR.glob("*.pdf"))


def title_of(filename: str) -> str:
    return filename.removesuffix(".pdf").replace("_", " ").title()


@router.get("/list")
async def documents_list():
    return {"documents": [{"filename": f, "title": title_of(f)} for f in list_docs()]}


@router.post("/find")
async def documents_find(payload: dict):
    """Accept {"query": "<spoken request>"}, return the matched document and action."""
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(400, "No query provided")
    docs = list_docs()
    if not docs:
        raise HTTPException(404, "No documents in the library")

    schema = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "enum": docs + ["none"]},
            "action": {"type": "string", "enum": ["open", "print"]},
            "reason": {"type": "string"},
        },
        "required": ["filename", "action", "reason"],
    }
    listing = "\n".join(f"- {f}: {title_of(f)}" for f in docs)
    result = await llm_extract(
        [{"role": "user", "content": f"Available documents:\n{listing}\n\nUser said: \"{query}\""}],
        MATCHER_PROMPT,
        schema,
    )
    if result["filename"] == "none":
        return {"match": None, "reason": result["reason"]}
    return {
        "match": {"filename": result["filename"], "title": title_of(result["filename"])},
        "action": result["action"],
        "reason": result["reason"],
    }


@router.get("/view/{filename}")
async def documents_view(filename: str):
    path = DOCS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Document not found")
    return FileResponse(path, media_type="application/pdf")


@router.post("/print")
async def documents_print(payload: dict):
    """Send a document to the default system printer via lpr."""
    filename = Path(payload.get("filename", "")).name
    path = DOCS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Document not found")
    try:
        subprocess.run(["lpr", str(path)], check=True, capture_output=True, timeout=30)
    except FileNotFoundError:
        raise HTTPException(500, "lpr not available on this system")
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Print failed: {e.stderr.decode().strip() or 'no printer configured?'}")
    return {"status": "sent", "filename": filename}


# Live transcription for spoken document requests.
router.add_api_websocket_route("/listen", live_transcribe_socket)
