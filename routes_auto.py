"""Fully automated kiosk interview: one click, then hands-free through to printing.

Flow:  POST /auto/start  -> greeting
       POST /auto/turn   -> next question, or done=true when the LLM has everything
       POST /auto/finish -> extract profile, build PDF, send to printer

Unlike /interview/*, conversation state lives on the SERVER (keyed by session id) and is
written to disk every turn, so a refresh or crash never loses an interview.
"""
import json
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from engines import llm_extract, synthesize_wav
from facts import apply_facts, date_context
from printer_status import default_printer_status
from profile_schema import EXTRACTOR_PROMPT, PROFILE_SCHEMA
from resume_docx import render_resume_docx
from resume_pdf import render_resume
from ws_stt import live_transcribe_socket

router = APIRouter(prefix="/auto", tags=["auto"])

DATA_DIR = Path(__file__).parent / "data" / "sessions"
MAX_TURNS = 22          # hard stop so a rambling interview always terminates
LANGUAGES = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}

# Everything a resume needs. The server tracks which of these the candidate has answered
# so the model cannot get stuck re-asking a question that was already answered - a real
# failure mode when the LLM is left to infer coverage from the transcript alone.
TOPICS = {
    "name": "full name (ask them to spell it if unclear)",
    "target_role": "the role or job they are targeting",
    "education": "degree, college/university and year",
    "experience": "work experience or internships, with numbers and impact",
    "projects": "personal or academic projects",
    "skills": "technical and professional skills",
    "achievements": "awards, positions of responsibility or achievements",
    "contact": "email address and 10-digit phone number",
    "location": "city and 6-digit PIN code",
}

# session_id -> {"messages": [...], "turns": int, "dir": Path, "covered": set}
_sessions: dict[str, dict] = {}

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {
            "type": "array",
            "items": {"type": "string", "enum": list(TOPICS)},
        },
        "reply": {"type": "string"},
    },
    "required": ["covered", "reply"],
}

AUTO_PROMPT = """You are a friendly, professional resume interviewer on a voice call.

You must return two things:
1. "covered": the list of topics the candidate has NOW answered well enough for a resume, \
judging from the whole conversation so far. Only include a topic once you genuinely have \
the information; include every topic that has been answered, not just the newest one.
2. "reply": what you say next, out loud.

Rules for "reply":
- Keep it SHORT (1-2 sentences) and natural to say aloud. Plain text only: no markdown, \
no bullet points, no emoji.
- Ask exactly ONE question, about the FIRST topic in the "still needed" list you are given.
- NEVER re-ask something already answered. If the candidate already told you, move on.
- Probe for specifics and numbers (team size, users, percentages, years) when vague.
- Speech recognition garbles names, emails and colleges: ask them to spell the important ones.
- The candidate may speak English, Hindi or Punjabi and may mix them. ALWAYS reply in the \
language they are mainly using.
- Candidates are in India: expect Indian colleges, cities and PIN codes.
- When the "still needed" list is empty, do not ask anything: give a short thank-you \
saying their resume is being prepared."""


def _remaining(covered: set) -> list[str]:
    return [t for t in TOPICS if t not in covered]


def _coverage_note(covered: set) -> str:
    """Explicit state handed to the model each turn - this is what stops repeat questions."""
    remaining = _remaining(covered)
    if not remaining:
        return "\n\nAlready covered: everything. Still needed: NOTHING - close the interview now."
    return (f"\n\nAlready covered: {', '.join(sorted(covered)) or 'nothing yet'}."
            f"\nStill needed, in order: {'; '.join(f'{t} ({TOPICS[t]})' for t in remaining)}."
            f"\nAsk about the FIRST item in that list.")


def _language_note(language: str) -> str:
    name = LANGUAGES[language]
    script_note = {
        "en": "Use natural English.",
        "hi": "Write Hindi in Devanagari script.",
        "pa": "Use Indian Punjabi and write it only in Gurmukhi script, not Shahmukhi.",
    }[language]
    return (
        f"\n\nThe candidate selected {name} for this interview. Conduct the entire spoken "
        f"interview in {name}, including the greeting and every question. Keep names, email "
        f"addresses, phone numbers, and technical terms in their natural form. {script_note}"
    )


def _is_substantive(answer: str) -> bool:
    """Did the candidate actually answer, or deflect? Used to guarantee forward progress."""
    a = answer.strip().lower()
    if len(a) < 3:
        return False
    return not any(a.startswith(p) for p in (
        "no", "nope", "nothing", "none", "skip", "i don't", "i dont", "not really",
        "pass", "next", "nahi", "kuch nahi",
    ))


def _session(session_id: str) -> dict:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found - please start a new interview")
    return s


def _save(s: dict):
    (s["dir"] / "transcript.json").write_text(
        json.dumps(s["messages"], indent=2, ensure_ascii=False)
    )


@router.post("/start")
async def auto_start(payload: dict):
    """Begin an interview: create a session and return the spoken greeting."""
    language = payload.get("language", "en")
    if language not in LANGUAGES:
        language = "en"
    session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content":
                 "(The candidate has joined the voice call. Greet them and begin the interview.)"}]
    result = await llm_extract(
        messages,
        AUTO_PROMPT + date_context() + _language_note(language) + _coverage_note(set()),
        TURN_SCHEMA,
    )
    reply = result["reply"].strip()
    messages.append({"role": "assistant", "content": reply})
    s = {"messages": messages, "turns": 0, "dir": session_dir,
         "covered": set(), "asked": next(iter(TOPICS)), "language": language}
    _sessions[session_id] = s
    _save(s)
    return {"session_id": session_id, "reply": reply, "done": False,
            "language": language}


@router.post("/turn")
async def auto_turn(payload: dict):
    """Submit the candidate's answer, get the next question (or done=true)."""
    s = _session(payload.get("session_id", ""))
    answer = (payload.get("answer") or "").strip()
    if not answer:
        raise HTTPException(400, "Empty answer")

    s["messages"].append({"role": "user", "content": answer})
    s["turns"] += 1

    # The topic we asked about last turn counts as covered once they answer it, whatever
    # the model reports. Without this the interview can stall on a topic forever.
    asked = s.get("asked")
    if asked and _is_substantive(answer):
        s["covered"].add(asked)
    elif asked:
        s["covered"].add(asked)  # they declined it - do not ask again either

    result = await llm_extract(
        s["messages"],
        AUTO_PROMPT + date_context() + _language_note(s.get("language", "en"))
        + _coverage_note(s["covered"]),
        TURN_SCHEMA,
    )
    # coverage only ever grows: a later turn must not "un-answer" an earlier topic
    s["covered"].update(t for t in result.get("covered", []) if t in TOPICS)
    reply = result["reply"].strip()
    remaining = _remaining(s["covered"])
    s["asked"] = remaining[0] if remaining else None

    done = not remaining or s["turns"] >= MAX_TURNS
    if done:
        reply = reply or "Thank you, that is everything I need. Your resume is being prepared now."

    s["messages"].append({"role": "assistant", "content": reply})
    _save(s)
    return {
        "reply": reply,
        "done": done,
        "turns": s["turns"],
        "covered": sorted(s["covered"]),
        "remaining": remaining,
    }


async def _build_profile(s: dict) -> dict:
    real = [m for m in s["messages"] if not m["content"].startswith("(")]
    if len(real) < 2:
        raise HTTPException(400, "Interview too short to build a resume")

    transcript = "\n".join(
        f"{'Interviewer' if m['role'] == 'assistant' else 'Candidate'}: {m['content']}"
        for m in real
    )
    try:
        profile = await llm_extract(
            [{"role": "user", "content": f"Interview transcript:\n\n{transcript}"}],
            EXTRACTOR_PROMPT + date_context(),
            PROFILE_SCHEMA,
        )
    except Exception as e:
        raise HTTPException(500, f"Could not build the profile: {e}")
    profile = apply_facts(profile)
    session_dir = s["dir"]
    (session_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    return profile


def _build_documents(s: dict, profile: dict) -> dict:
    session_dir = s["dir"]
    try:
        (session_dir / "resume.pdf").write_bytes(render_resume(profile))
        (session_dir / "resume.docx").write_bytes(render_resume_docx(profile))
    except Exception as e:
        raise HTTPException(500, f"Could not generate the resume document: {e}")
    return {
        "session_id": session_dir.name,
        "pdf_url": f"/auto/resume/{session_dir.name}.pdf",
        "docx_url": f"/auto/resume/{session_dir.name}.docx",
        "saved": True,
    }


def _print_saved_pdf(s: dict) -> dict:
    """Print only when CUPS reports a ready default printer."""
    printer = default_printer_status()
    if not printer["connected"]:
        return {
            "printed": False,
            "print_status": "not_connected",
            "printer_name": None,
            "print_error": None,
        }

    pdf = s["dir"] / "resume.pdf"
    if not pdf.exists():
        raise HTTPException(404, "Resume PDF has not been generated")
    try:
        subprocess.run(["lpr", str(pdf)], check=True, capture_output=True, timeout=30)
        return {
            "printed": True,
            "print_status": "printed",
            "printer_name": printer["name"],
            "print_error": None,
        }
    except FileNotFoundError:
        error = "No printing system found (lpr is not available on this machine)."
    except subprocess.TimeoutExpired:
        error = "The printer did not respond in time."
    except subprocess.CalledProcessError as e:
        error = e.stderr.decode().strip() or "The printer could not accept the PDF."
    return {
        "printed": False,
        "print_status": "failed",
        "printer_name": printer["name"],
        "print_error": error,
    }


@router.post("/profile")
async def auto_profile(payload: dict):
    """Extract and save a structured profile from the completed interview."""
    s = _session(payload.get("session_id", ""))
    profile = await _build_profile(s)
    return {"profile": profile, "corrections": profile.get("_corrections", [])}


@router.post("/build-resume")
async def auto_build_resume(payload: dict):
    """Generate and save PDF and Word files from the extracted profile."""
    s = _session(payload.get("session_id", ""))
    profile_file = s["dir"] / "profile.json"
    if not profile_file.exists():
        raise HTTPException(404, "Profile has not been generated")
    return _build_documents(s, json.loads(profile_file.read_text()))


@router.post("/print")
async def auto_print(payload: dict):
    """Print the saved PDF only when the default printer is ready."""
    return _print_saved_pdf(_session(payload.get("session_id", "")))


@router.post("/finish")
async def auto_finish(payload: dict):
    """Compatibility endpoint: build, save, and conditionally print the resume."""
    s = _session(payload.get("session_id", ""))
    profile = await _build_profile(s)
    documents = _build_documents(s, profile)
    printing = _print_saved_pdf(s)

    return {
        **documents,
        **printing,
        "profile": profile,
        "corrections": profile.get("_corrections", []),
    }


@router.get("/resume/{filename}")
async def auto_resume(filename: str):
    """Serve a generated resume: <session_id>.pdf or <session_id>.docx."""
    name = Path(filename).name
    session_id, _, ext = name.rpartition(".")
    if ext not in ("pdf", "docx"):
        raise HTTPException(404, "Unknown format")
    path = DATA_DIR / session_id / f"resume.{ext}"
    if not path.exists():
        raise HTTPException(404, "Resume not found")
    media = ("application/pdf" if ext == "pdf"
             else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    return FileResponse(path, media_type=media, filename=f"resume-{session_id}.{ext}",
                        content_disposition_type="inline" if ext == "pdf" else "attachment")


@router.post("/speak")
async def auto_speak(payload: dict):
    """Synthesize an interviewer line."""
    from fastapi.responses import Response

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "No text provided")
    return Response(content=synthesize_wav(text), media_type="audio/wav")


router.add_api_websocket_route("/listen", live_transcribe_socket)
