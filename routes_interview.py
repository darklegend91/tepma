"""Main interview endpoints (used by the interview page)."""
import json
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from engines import llm_chat, llm_extract, synthesize_wav
from facts import apply_facts, date_context
from profile_schema import EXTRACTOR_PROMPT, PROFILE_SCHEMA
from resume_docx import render_resume_docx
from resume_pdf import render_resume
from ws_stt import live_transcribe_socket

DATA_DIR = Path(__file__).parent / "data" / "sessions"

router = APIRouter(prefix="/interview", tags=["interview"])

INTERVIEWER_PROMPT = """You are a friendly, professional resume interviewer on a voice call. \
Your goal is to collect everything needed for the candidate's resume: name, contact details, \
target role, education, work experience, skills, projects, and achievements.

Rules:
- This is a VOICE conversation. Keep replies short (1-3 sentences) and natural to say aloud.
- Plain text only: no markdown, no bullet points, no emoji.
- Ask ONE question at a time.
- Probe for specifics and numbers (team size, users, impact, dates) when the answer is vague.
- The transcript comes from speech recognition, so names and emails may be misspelled; \
confirm important spellings by asking, not by guessing.
- Start by greeting the candidate, then ask for their name and the role they are targeting.
- When you have covered all resume sections, thank them and say the interview is complete.
- The candidate may speak English, Hindi, or Punjabi, and may mix them in one sentence. \
ALWAYS reply in the language the candidate is mainly using. If they switch, you switch.
- Candidates are usually in India: expect Indian universities, cities, PIN codes, and \
phone numbers. Ask for a 6-digit PIN code when collecting their address."""


@router.post("/chat")
async def interview_chat(payload: dict):
    """Accept {"messages": [{role, content}, ...]} and return the interviewer's reply."""
    reply = await llm_chat(payload.get("messages", []), INTERVIEWER_PROMPT + date_context())
    return {"reply": reply}


@router.post("/speak")
async def interview_speak(payload: dict):
    """Synthesize the interviewer's reply: {"text": "..."} -> WAV."""
    text = payload.get("text", "").strip()
    if not text:
        return Response(status_code=400, content="No text provided")
    return Response(content=synthesize_wav(text), media_type="audio/wav")


@router.post("/finish")
async def interview_finish(payload: dict):
    """Turn the interview transcript into a structured profile JSON and a resume PDF.

    Accepts {"messages": [...]}. Saves transcript, profile, and PDF under data/sessions/
    and returns {"session_id", "profile", "pdf_url"}.
    """
    messages = [m for m in payload.get("messages", []) if not m["content"].startswith("(")]
    if len(messages) < 2:
        raise HTTPException(400, "Not enough conversation to build a resume")

    transcript = "\n".join(
        f"{'Interviewer' if m['role'] == 'assistant' else 'Candidate'}: {m['content']}"
        for m in messages
    )
    profile = await llm_extract(
        [{"role": "user", "content": f"Interview transcript:\n\n{transcript}"}],
        EXTRACTOR_PROMPT + date_context(),
        PROFILE_SCHEMA,
    )
    profile = apply_facts(profile)  # deterministic corrections (see finetune/GUIDE.md)

    session_id = time.strftime("%Y%m%d-%H%M%S")
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.json").write_text(json.dumps(messages, indent=2))
    (session_dir / "profile.json").write_text(json.dumps(profile, indent=2))
    (session_dir / "resume.pdf").write_bytes(render_resume(profile))

    return {
        "session_id": session_id,
        "profile": profile,
        "pdf_url": f"/interview/resume/{session_id}",
    }


@router.get("/resume/latest")
async def interview_resume_latest():
    """Serve the most recent session's resume PDF (regenerating it if missing)."""
    if not DATA_DIR.exists():
        raise HTTPException(404, "No sessions yet")
    for session_dir in sorted(DATA_DIR.iterdir(), reverse=True):
        pdf = session_dir / "resume.pdf"
        profile_file = session_dir / "profile.json"
        if not pdf.exists() and profile_file.exists():
            pdf.write_bytes(render_resume(json.loads(profile_file.read_text())))
        if pdf.exists():
            return FileResponse(pdf, media_type="application/pdf",
                                filename=f"resume-{session_dir.name}.pdf",
                                content_disposition_type="inline")
    raise HTTPException(404, "No resume found in any session")


def _latest_session() -> Path | None:
    """Most recent session dir that has a profile.json, or None."""
    if not DATA_DIR.exists():
        return None
    for session_dir in sorted(DATA_DIR.iterdir(), reverse=True):
        if (session_dir / "profile.json").exists():
            return session_dir
    return None


@router.get("/resume/latest/docx")
async def interview_resume_latest_docx():
    """Editable Word version of the most recent resume."""
    session_dir = _latest_session()
    if session_dir is None:
        raise HTTPException(404, "No sessions yet")
    profile = json.loads((session_dir / "profile.json").read_text())
    return Response(
        content=render_resume_docx(profile),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="resume-{session_dir.name}.docx"'},
    )


@router.post("/print")
async def interview_print():
    """Print the latest resume directly on the default system printer (no dialog)."""
    import subprocess

    session_dir = _latest_session()
    if session_dir is None:
        raise HTTPException(404, "No resume to print yet")
    pdf_file = session_dir / "resume.pdf"
    if not pdf_file.exists():
        profile = json.loads((session_dir / "profile.json").read_text())
        pdf_file.write_bytes(render_resume(profile))
    try:
        subprocess.run(["lpr", str(pdf_file)], check=True, capture_output=True, timeout=30)
    except FileNotFoundError:
        raise HTTPException(500, "Printer error: lpr command not available on this system")
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode().strip() or "no default printer configured"
        raise HTTPException(500, f"Printer error: {detail}")
    return {"status": "printed", "session": session_dir.name}


@router.post("/email")
async def interview_email(payload: dict):
    """Email the latest resume (PDF + editable Word) to {"to": "<address>"}.

    Uses SMTP credentials from env: SMTP_USER, SMTP_PASS, and optionally
    SMTP_HOST (default smtp.gmail.com) and SMTP_PORT (default 587).
    """
    to_addr = payload.get("to", "").strip()
    if "@" not in to_addr:
        raise HTTPException(400, "Invalid email address")
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not user or not password:
        raise HTTPException(
            500,
            "Email not configured. Set SMTP_USER and SMTP_PASS environment variables "
            "before starting the server (for Gmail, use an App Password).",
        )
    session_dir = _latest_session()
    if session_dir is None:
        raise HTTPException(404, "No resume to send yet")
    profile = json.loads((session_dir / "profile.json").read_text())
    pdf_file = session_dir / "resume.pdf"
    if not pdf_file.exists():
        pdf_file.write_bytes(render_resume(profile))

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = f"Resume - {profile.get('name', 'TePMA')}"
    msg.set_content(
        f"Hi,\n\nPlease find attached the resume of {profile.get('name', '')} "
        "generated with TePMA.\nThe PDF is print-ready; the Word file is editable.\n"
    )
    msg.add_attachment(pdf_file.read_bytes(), maintype="application", subtype="pdf",
                       filename="resume.pdf")
    msg.add_attachment(
        render_resume_docx(profile),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="resume.docx",
    )
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(500, "SMTP login failed - check SMTP_USER / SMTP_PASS")
    except OSError as e:
        raise HTTPException(500, f"Could not reach SMTP server: {e}")
    return {"status": "sent", "to": to_addr}


@router.get("/resume/{session_id}")
async def interview_resume(session_id: str):
    """Download the generated resume PDF for a session."""
    pdf = DATA_DIR / Path(session_id).name / "resume.pdf"
    if not pdf.exists():
        raise HTTPException(404, "No resume for this session")
    return FileResponse(pdf, media_type="application/pdf", filename=f"resume-{session_id}.pdf",
                        content_disposition_type="inline")


# Live transcription of the candidate's answers.
router.add_api_websocket_route("/listen", live_transcribe_socket)
