"""Unified document-or-resume assistant APIs used by the /assistant page.

The resume branch intentionally reuses /auto. This router owns only the document branch:
stored-document selection, guided application details, deterministic rendering, and safe
best-effort printing.
"""
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from document_render import render_application_docx, render_application_pdf
from engines import llm_extract, synthesize_wav
from facts import date_context
from printer_status import default_printer_status
from routes_documents import DOCS_DIR, list_docs, title_of
from ws_stt import live_transcribe_socket

router = APIRouter(prefix="/assistant/api", tags=["unified-assistant"])

GENERATED_DIR = Path(__file__).parent / "data" / "generated_documents"
LANGUAGES = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}
MAX_FIELDS = 10

_sessions: dict[str, dict] = {}


ROUTER_PROMPT = """You route requests for a safe document kiosk.

Choose "existing" only when one available PDF is a reasonable match for what the user
described. Choose "generate_application" only for a user-authored application, request
letter, cover letter, complaint, or similar correspondence that can safely be drafted
from details supplied by the user. Choose "unsupported" for identity documents,
certificates, marksheets, licences, prescriptions, court/government-issued records, or
anything the kiosk must not fabricate.

Never invent a filename. reason must be one short, user-friendly sentence."""

PLAN_PROMPT = """Plan a short voice interview that collects only the information required
to draft the requested application or letter. Extract any values already present in the
description. Return 3 to 8 concise fields in the order they should be asked. Use stable
snake_case keys. Questions must ask one thing at a time and use the requested conversation
language. Keep names, addresses, dates, identifiers, and organization names exactly as the
user gives them. The final printable document will be produced separately."""

WRITER_PROMPT = """Write a polished formal application or letter using ONLY the supplied
request and collected field values. Do not invent names, dates, addresses, reasons,
identifiers, qualifications, or claims. If a detail was explicitly skipped, omit it.
Produce the printable document in professional English. Use short paragraphs and an
appropriate Indian formal-letter style. Return only the required structured fields."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "document_title": {"type": "string"},
        "opening": {"type": "string"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "label": {"type": "string"},
                    "question": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "label", "question", "value"],
            },
        },
    },
    "required": ["document_title", "opening", "fields"],
}

DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "date": {"type": "string"},
        "recipient_lines": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "string"},
        "salutation": {"type": "string"},
        "paragraphs": {"type": "array", "items": {"type": "string"}},
        "closing": {"type": "string"},
        "sender_lines": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title", "date", "recipient_lines", "subject", "salutation",
        "paragraphs", "closing", "sender_lines",
    ],
}


def _new_session(language: str, query: str) -> dict:
    session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:5]}"
    directory = GENERATED_DIR / session_id
    directory.mkdir(parents=True, exist_ok=True)
    session = {
        "session_id": session_id,
        "dir": directory,
        "language": language,
        "query": query,
        "status": "routing",
        "fields": [],
    }
    _sessions[session_id] = session
    _save(session)
    return session


def _public_state(session: dict) -> dict:
    return {
        key: value
        for key, value in session.items()
        if key not in {"dir"}
    }


def _save(session: dict):
    (session["dir"] / "workflow.json").write_text(
        json.dumps(_public_state(session), ensure_ascii=False, indent=2)
    )


def _session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None and re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{5}", session_id):
        directory = GENERATED_DIR / session_id
        workflow = directory / "workflow.json"
        if workflow.is_file():
            try:
                session = json.loads(workflow.read_text())
                session["dir"] = directory
                _sessions[session_id] = session
            except (OSError, ValueError, TypeError):
                session = None
    if session is None:
        raise HTTPException(404, "Document session not found; please start again")
    return session


def _language_instruction(language: str) -> str:
    return {
        "en": "Ask every question in natural English.",
        "hi": "Ask every question in natural Hindi using Devanagari script.",
        "pa": "Ask every question in Indian Punjabi using Gurmukhi script.",
    }[language]


def _fallback_fields(language: str) -> list[dict]:
    questions = {
        "en": [
            ("applicant_name", "Applicant name", "What is your full name?"),
            ("recipient", "Recipient", "Who should this application be addressed to?"),
            ("organization", "Organization", "What is the organization or institution name?"),
            ("purpose", "Purpose", "Please describe the exact purpose of the application."),
            ("relevant_dates", "Relevant dates", "Are there any dates or time period to include?"),
            ("contact_details", "Contact details", "What contact details should appear on it?"),
        ],
        "hi": [
            ("applicant_name", "आवेदक का नाम", "आपका पूरा नाम क्या है?"),
            ("recipient", "प्राप्तकर्ता", "यह आवेदन किसे संबोधित करना है?"),
            ("organization", "संस्था", "संस्था या संगठन का नाम क्या है?"),
            ("purpose", "उद्देश्य", "आवेदन का सही उद्देश्य बताइए।"),
            ("relevant_dates", "तारीखें", "इसमें कौन-सी तारीख या समय अवधि शामिल करनी है?"),
            ("contact_details", "संपर्क विवरण", "इसमें कौन-सा संपर्क विवरण लिखना है?"),
        ],
        "pa": [
            ("applicant_name", "ਬਿਨੈਕਾਰ ਦਾ ਨਾਮ", "ਤੁਹਾਡਾ ਪੂਰਾ ਨਾਮ ਕੀ ਹੈ?"),
            ("recipient", "ਪ੍ਰਾਪਤਕਰਤਾ", "ਇਹ ਅਰਜ਼ੀ ਕਿਸ ਨੂੰ ਸੰਬੋਧਿਤ ਕਰਨੀ ਹੈ?"),
            ("organization", "ਸੰਸਥਾ", "ਸੰਸਥਾ ਜਾਂ ਸੰਗਠਨ ਦਾ ਨਾਮ ਕੀ ਹੈ?"),
            ("purpose", "ਮਕਸਦ", "ਅਰਜ਼ੀ ਦਾ ਸਹੀ ਮਕਸਦ ਦੱਸੋ।"),
            ("relevant_dates", "ਤਾਰੀਖਾਂ", "ਕਿਹੜੀ ਤਾਰੀਖ ਜਾਂ ਸਮਾਂ ਮਿਆਦ ਸ਼ਾਮਲ ਕਰਨੀ ਹੈ?"),
            ("contact_details", "ਸੰਪਰਕ ਵੇਰਵਾ", "ਕਿਹੜਾ ਸੰਪਰਕ ਵੇਰਵਾ ਲਿਖਣਾ ਹੈ?"),
        ],
    }
    return [
        {"key": key, "label": label, "question": question, "value": ""}
        for key, label, question in questions[language]
    ]


def _normalize_fields(raw_fields, language: str) -> list[dict]:
    fields = []
    used = set()
    for index, raw in enumerate(raw_fields if isinstance(raw_fields, list) else []):
        if not isinstance(raw, dict):
            continue
        key = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("key", "")).lower()).strip("_")
        key = key or f"detail_{index + 1}"
        if key in used:
            continue
        label = str(raw.get("label", "")).strip() or key.replace("_", " ").title()
        question = str(raw.get("question", "")).strip()
        if not question:
            continue
        fields.append({
            "key": key,
            "label": label[:100],
            "question": question[:300],
            "value": str(raw.get("value", "")).strip()[:2000],
        })
        used.add(key)
        if len(fields) == MAX_FIELDS:
            break
    return fields or _fallback_fields(language)


def _next_missing(session: dict) -> tuple[int | None, dict | None]:
    for index, field in enumerate(session["fields"]):
        if not field.get("value"):
            return index, field
    return None, None


def _progress(session: dict) -> dict:
    total = len(session["fields"])
    collected = sum(bool(field.get("value")) for field in session["fields"])
    return {"collected": collected, "total": total}


def _print_pdf(path: Path) -> dict:
    printer = default_printer_status()
    if not printer["connected"]:
        return {
            "printed": False,
            "print_status": "not_connected",
            "printer_name": None,
            "print_error": None,
        }
    try:
        subprocess.run(["lpr", str(path)], check=True, capture_output=True, timeout=30)
        return {
            "printed": True,
            "print_status": "printed",
            "printer_name": printer["name"],
            "print_error": None,
        }
    except FileNotFoundError:
        error = "The lpr printing command is not available on this machine."
    except subprocess.TimeoutExpired:
        error = "The printer did not respond in time."
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode().strip() or "The printer could not accept the PDF."
    return {
        "printed": False,
        "print_status": "failed",
        "printer_name": printer["name"],
        "print_error": error,
    }


def _result_for_path(path: Path, pdf_url: str) -> dict:
    return {
        "saved": True,
        "pdf_url": pdf_url,
        **_print_pdf(path),
    }


async def _finish_generated(session: dict) -> dict:
    supplied = {
        field["label"]: field["value"]
        for field in session["fields"]
        if field.get("value") and field["value"] != "(skipped)"
    }
    prompt = (
        f"Original request: {session['query']}\n"
        f"Planned document: {session.get('document_title', 'Application')}\n"
        f"User-supplied details:\n{json.dumps(supplied, ensure_ascii=False, indent=2)}"
    )
    try:
        document = await llm_extract(
            [{"role": "user", "content": prompt}],
            WRITER_PROMPT + date_context(),
            DOCUMENT_SCHEMA,
        )
        pdf = render_application_pdf(document)
        docx = render_application_docx(document)
    except Exception as exc:
        raise HTTPException(500, f"Could not generate the application: {exc}") from exc

    (session["dir"] / "document.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2)
    )
    pdf_path = session["dir"] / "document.pdf"
    pdf_path.write_bytes(pdf)
    (session["dir"] / "document.docx").write_bytes(docx)
    session["status"] = "completed"
    session["document"] = document
    _save(session)
    result = {
        "session_id": session["session_id"],
        "generated": True,
        "title": document.get("title", session.get("document_title", "Application")),
        "docx_url": f"/assistant/api/generated/{session['session_id']}.docx",
        **_result_for_path(
            pdf_path,
            f"/assistant/api/generated/{session['session_id']}.pdf",
        ),
    }
    return result


@router.post("/document/start")
async def document_start(payload: dict):
    """Find a stored PDF or start a guided application-generation session."""
    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(400, "Please describe the document you need")
    language = str(payload.get("language", "en"))
    if language not in LANGUAGES:
        language = "en"
    session = _new_session(language, query)
    docs = list_docs()
    route_schema = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["existing", "generate_application", "unsupported"],
            },
            "filename": {"type": "string", "enum": docs + ["none"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "filename", "reason"],
    }
    listing = "\n".join(f"- {name}: {title_of(name)}" for name in docs) or "(empty)"
    try:
        route = await llm_extract(
            [{"role": "user", "content": f"Available PDFs:\n{listing}\n\nRequest: {query}"}],
            ROUTER_PROMPT,
            route_schema,
        )
    except Exception as exc:
        raise HTTPException(500, f"Could not understand the document request: {exc}") from exc

    filename = route.get("filename")
    if route.get("decision") == "existing" and filename in docs:
        session["status"] = "matched"
        session["filename"] = filename
        _save(session)
        path = DOCS_DIR / filename
        return {
            "status": "matched",
            "session_id": session["session_id"],
            "reply": route.get("reason", "I found the requested document."),
            "document": {
                "filename": filename,
                "title": title_of(filename),
            },
            **_result_for_path(path, f"/documents/view/{filename}"),
        }

    if route.get("decision") == "unsupported":
        session["status"] = "unsupported"
        _save(session)
        return {
            "status": "unsupported",
            "session_id": session["session_id"],
            "reply": route.get("reason") or "This type of official document cannot be generated here.",
        }

    try:
        plan = await llm_extract(
            [{"role": "user", "content": f"Requested document: {query}"}],
            PLAN_PROMPT + "\n\n" + _language_instruction(language),
            PLAN_SCHEMA,
        )
    except Exception as exc:
        raise HTTPException(500, f"Could not plan the application questions: {exc}") from exc
    session["status"] = "collecting"
    session["document_title"] = str(plan.get("document_title", "Application")).strip() or "Application"
    session["fields"] = _normalize_fields(plan.get("fields"), language)
    _save(session)
    index, field = _next_missing(session)
    if field is None:
        return {"status": "completed", "result": await _finish_generated(session)}
    session["current_index"] = index
    _save(session)
    return {
        "status": "collecting",
        "session_id": session["session_id"],
        "reply": field["question"],
        "document_title": session["document_title"],
        **_progress(session),
    }


@router.post("/document/turn")
async def document_turn(payload: dict):
    """Save one answer and return the next missing application detail."""
    session = _session(str(payload.get("session_id", "")))
    if session.get("status") != "collecting":
        raise HTTPException(409, "This document is not waiting for more details")
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise HTTPException(400, "Please provide an answer or say skip")
    index, field = _next_missing(session)
    if field is None or index is None:
        result = await _finish_generated(session)
        return {"status": "completed", "result": result}
    normalized = answer.lower().strip(" .")
    field["value"] = "(skipped)" if normalized in {
        "skip", "pass", "none", "not applicable", "n/a", "नहीं", "ਛੱਡੋ",
    } else answer[:2000]
    _save(session)

    next_index, next_field = _next_missing(session)
    if next_field is None:
        result = await _finish_generated(session)
        return {
            "status": "completed",
            "reply": "Thank you. Your document has been generated and saved.",
            **_progress(session),
            "result": result,
        }
    session["current_index"] = next_index
    _save(session)
    return {
        "status": "collecting",
        "session_id": session["session_id"],
        "reply": next_field["question"],
        **_progress(session),
    }


@router.post("/document/print")
async def document_print(payload: dict):
    """Retry printing a matched or generated document."""
    session = _session(str(payload.get("session_id", "")))
    if session.get("status") == "matched":
        path = DOCS_DIR / Path(session.get("filename", "")).name
    else:
        path = session["dir"] / "document.pdf"
    if not path.is_file():
        raise HTTPException(404, "The PDF has not been created")
    return _print_pdf(path)


@router.get("/generated/{filename}")
async def generated_document(filename: str):
    name = Path(filename).name
    session_id, separator, extension = name.rpartition(".")
    if (
        not separator
        or extension not in {"pdf", "docx"}
        or re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{5}", session_id) is None
    ):
        raise HTTPException(404, "Unknown document format")
    path = GENERATED_DIR / session_id / f"document.{extension}"
    if not path.is_file():
        raise HTTPException(404, "Generated document not found")
    media_type = (
        "application/pdf" if extension == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"TePMA-document-{session_id}.{extension}",
        content_disposition_type="inline" if extension == "pdf" else "attachment",
    )


@router.post("/speak")
async def assistant_speak(payload: dict):
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "No text provided")
    return Response(content=synthesize_wav(text), media_type="audio/wav")


router.add_api_websocket_route("/listen", live_transcribe_socket)
