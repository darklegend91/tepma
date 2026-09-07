"""Fully automated kiosk interview: one click, then hands-free through to printing.

Flow:  POST /auto/start        -> greeting
       WS   /auto/turn-stream  -> next question, spoken sentence by sentence as it is
                                  generated; POST /auto/turn is the fallback
       POST /auto/finish       -> extract profile, build PDF, send to printer

Unlike /interview/*, conversation state lives on the SERVER (keyed by session id) and is
written to disk every turn, so a refresh or crash never loses an interview.
"""
import asyncio
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engines import (TURN_MODEL, extract_partial_string, llm_extract, llm_stream,
                     split_sentences, synthesize_wav)
import mailer
from facts import apply_facts, date_context
from printer_status import default_printer_status
from profile_schema import EXTRACTOR_PROMPT, PROFILE_SCHEMA, romanize_profile
from resume_docx import render_resume_docx
from resume_pdf import render_resume
from ws_stt import live_transcribe_socket

router = APIRouter(prefix="/auto", tags=["auto"])

DATA_DIR = Path(__file__).parent / "data" / "sessions"
MAX_TURNS = 22          # hard stop so a rambling interview always terminates
LANGUAGES = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}

# The interview is a fixed script, asked in this order, because a resume has fields that
# are simply not optional. Letting the model choose what to ask next - the previous design -
# meant a candidate who rambled about projects could reach the end without a phone number.
# The model still phrases each question, in the candidate's language; it never picks the topic.
SECTIONS = [
    ("identity", "their full name. Ask them to spell it out letter by letter if it is at "
                 "all unclear."),
    # One fact per section. A section that bundled the number and the email together made
    # the model judge itself incomplete for whichever half it had not heard yet, and it
    # spent the follow-ups re-asking instead of moving on.
    ("phone", "their 10-digit mobile number. Ask them to say it digit by digit."),
    ("email", "their email address. Ask them to spell it out letter by letter - people "
              "dictate addresses rather than spelling them, and it is never heard correctly."),
    ("target_role", "the job profile or role they want to apply for."),
    ("education", "their education: highest qualification, the institution, and the year."),
    ("experience", "their past work experience and their current job status - whether they "
                   "are working, studying, or looking for work right now. Ask for numbers "
                   "and impact where there is any."),
    ("projects", "any projects, work samples or other proof of experience in their field."),
]
SECTION_IDS = [key for key, _ in SECTIONS]

# How many follow-ups one section may take before the interview moves on regardless. A
# candidate who cannot produce an email address must not trap the kiosk on question two.
# One is enough: anything genuinely missing is caught by the closing gap pass, which
# judges the extracted profile rather than the interviewer's opinion of the conversation.
MAX_FOLLOWUPS = 1
# Cap on the closing gap-filling questions, so the interview always ends.
MAX_GAP_QUESTIONS = 4

# session_id -> {"messages", "turns", "dir", "index", "followups", "phase", "gaps", ...}
_sessions: dict[str, dict] = {}

# "reply" is deliberately first: constrained decoding emits properties in schema order,
# so putting the spoken text first lets /auto/turn-stream start speaking a sentence while
# the model is still deciding whether the section is finished.
TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
    },
    "required": ["reply"],
}

AUTO_PROMPT = """You are a friendly, professional resume interviewer on a voice call.

You are told which ONE thing to ask about next. Return "reply": what you say out loud -
exactly one question about that one thing, and nothing else. You do not decide what comes
next or when the interview ends; you are told that every turn.

Rules for "reply":
- Keep it SHORT (1-2 sentences) and natural to say aloud. Plain text only: no markdown, \
no bullet points, no emoji.
- Ask about the CURRENT item only. Never skip ahead to something you were not asked to \
collect, and never re-ask something the candidate already answered.
- Probe for specifics and numbers (team size, users, percentages, years) when vague.
- Speech recognition garbles names, emails and colleges: ask them to spell those out.
- Candidates are in India: expect Indian colleges, cities and PIN codes."""


def _needs(s: dict) -> tuple[str | None, str | None]:
    """(what this turn must get, what comes after it) - (None, _) means the interview is over."""
    if s["phase"] == "sections":
        current = SECTIONS[s["index"]][1] if s["index"] < len(SECTIONS) else None
        following = SECTIONS[s["index"] + 1][1] if s["index"] + 1 < len(SECTIONS) else None
        return current, following
    gaps = s["gaps"]
    return (gaps[0] if gaps else None), (gaps[1] if len(gaps) > 1 else None)


def _current_need(s: dict) -> str | None:
    return _needs(s)[0]


def _turn_note(s: dict) -> str:
    """The per-turn stage direction. Kept out of the transcript - nobody said it.

    It tells the model exactly one thing to ask. Whether the interview has moved on is the
    server's decision alone: a 4B instruct model asked to judge "did they answer that?"
    said no to plainly complete answers and re-asked them, which is the most irritating
    thing a kiosk can do. Anything genuinely missing is caught later by the gap pass,
    which reads the extracted profile instead of guessing from the conversation.

    Delivered as the last message rather than in the system prompt because it changes every
    turn, and Ollama can only reuse its cached prefix up to the first token that differs.
    """
    current = _current_need(s)
    if current is None:
        return ("(Interview status: everything has been collected. Do not ask anything. "
                "Give a short thank-you saying their profile is being prepared.)")
    if s["followups"]:
        return (f"(Interview status - not spoken by the candidate. The candidate did not "
                f"answer the last question. Ask once more, in different words, about: "
                f"{current}\nAsk nothing else.)")
    if s["phase"] == "gaps":
        return (f"(Interview status - not spoken by the candidate. The interview is over "
                f"apart from one detail that was missing or not captured clearly: {current}"
                f"\nAsk for exactly that, once, and nothing else.)")
    collected = ", ".join(SECTION_IDS[:s["index"]]) or "nothing yet"
    return (f"(Interview status - not spoken by the candidate."
            f"\nAlready collected: {collected}. Do not ask about any of those again."
            f"\nAsk them now about: {current}"
            f"\nAsk that and nothing else.)")


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
    s = {"messages": messages, "turns": 0, "dir": session_dir, "language": language,
         "index": 0, "followups": 0, "phase": "sections", "gaps": [], "gaps_asked": 0}
    result = await llm_extract(
        messages + [{"role": "user", "content": _turn_note(s)}],
        AUTO_PROMPT + date_context() + _language_note(language),
        TURN_SCHEMA,
        model=TURN_MODEL,
    )
    reply = result["reply"].strip()
    messages.append({"role": "assistant", "content": reply})
    _sessions[session_id] = s
    _save(s)
    return {"session_id": session_id, "reply": reply, "done": False,
            "language": language, "section": SECTION_IDS[0],
            "collected": 0, "total": len(SECTIONS)}


@router.post("/turn")
async def auto_turn(payload: dict):
    """Submit the candidate's answer, get the next question (or done=true)."""
    s = _session(payload.get("session_id", ""))
    answer = (payload.get("answer") or "").strip()
    if not answer:
        raise HTTPException(400, "Empty answer")

    _begin_turn(s, answer)
    result = await llm_extract(_turn_messages(s), _turn_system(s), TURN_SCHEMA,
                               model=TURN_MODEL)
    return _finish_turn(s, result)


def _begin_turn(s: dict, answer: str):
    """Record the answer and move the interview on, BEFORE the next question is generated.

    Order matters: the question is written from the session's state, so if the state only
    advanced afterwards every question would ask for the thing that was just answered.

    Progression is mechanical - an answer with anything in it moves on, a refusal or a
    blank buys MAX_FOLLOWUPS re-asks and then moves on regardless. That is what makes the
    interview finite and its order fixed; what is genuinely missing is caught at the end
    by the gap pass, which reads the extracted profile rather than guessing.
    """
    s["messages"].append({"role": "user", "content": answer})
    s["turns"] += 1

    if not (_is_substantive(answer) or s["followups"] >= MAX_FOLLOWUPS):
        s["followups"] += 1
        return
    s["followups"] = 0
    if s["phase"] == "sections" and s["index"] < len(SECTIONS):
        s["index"] += 1
    elif s["phase"] == "gaps" and s["gaps"]:
        s["gaps"].pop(0)


def _turn_system(s: dict) -> str:
    """The stable half of the prompt. Constant for the whole session, so it stays cached."""
    return AUTO_PROMPT + date_context() + _language_note(s.get("language", "en"))


def _turn_messages(s: dict) -> list[dict]:
    """Conversation plus the current coverage note, without storing the note.

    The note must not go into s["messages"]: that is the saved transcript, and the
    extraction pass reads it back to build the resume.
    """
    return s["messages"] + [{"role": "user", "content": _turn_note(s)}]


def _finish_turn(s: dict, result: dict) -> dict:
    """Record the question that was just asked and report where the interview stands."""
    reply = (result.get("reply") or "").strip()
    done = (s["phase"] == "gaps" and _current_need(s) is None) or s["turns"] >= MAX_TURNS
    if done:
        reply = reply or "Thank you, that is everything I need. Your profile is being prepared now."

    s["messages"].append({"role": "assistant", "content": reply})
    _save(s)
    collected = s["index"] if s["phase"] == "sections" else len(SECTIONS)
    return {
        "reply": reply,
        "done": done,
        "turns": s["turns"],
        "phase": s["phase"],
        "section": (SECTION_IDS[s["index"]] if s["phase"] == "sections"
                    and s["index"] < len(SECTIONS) else "gaps"),
        "collected": collected,
        "total": len(SECTIONS),
        "needs_gap_check": s["phase"] == "sections" and s["index"] >= len(SECTIONS),
    }


# What "recorded with confidence" means, field by field. These are deliberately mechanical:
# a spoken email that never reached an "@" or a phone that lost digits is exactly the kind
# of thing the candidate should be given one more chance to correct before printing.
def _said_by_candidate(s: dict) -> str:
    """Everything the candidate actually said this session, lower-cased."""
    return " ".join(m["content"] for m in s["messages"]
                    if m["role"] == "user" and not m["content"].startswith("(")).lower()


def _profile_gaps(profile: dict) -> list[str]:
    gaps = []
    name = (profile.get("name") or "").strip()
    if len(name.split()) < 2:
        gaps.append("their full name, first and last, spelled out letter by letter.")
    email = (profile.get("email") or "").strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        gaps.append("their email address, spelled out letter by letter.")
    digits = "".join(c for c in (profile.get("phone") or "") if c.isdigit())
    if len(digits) not in (10, 12):     # 10 digits, or 12 with the 91 country code
        gaps.append("their 10-digit mobile number, digit by digit.")
    if not (profile.get("target_role") or "").strip():
        gaps.append("the job profile or role they are applying for.")
    if not profile.get("education"):
        gaps.append("their highest qualification, the institution and the year.")
    if not profile.get("experience") and not profile.get("projects"):
        gaps.append("their work experience, current job status, or any project they can show.")
    if not profile.get("skills"):
        gaps.append("the main skills they would want on their resume.")
    # A PIN is the test, not a non-empty string: asked for a location it never heard, the
    # extractor will happily infer a plausible city from the college name. A 6-digit PIN
    # is something only the candidate can supply, and facts.py validates it against the
    # postal directory afterwards.
    location = (profile.get("location") or "").strip()
    if not re.search(r"\b\d{6}\b", location):
        gaps.append("the city they live in, and its 6-digit PIN code.")
    return gaps[:MAX_GAP_QUESTIONS]


async def enter_gap_phase(s: dict) -> dict:
    """After the fixed sections: extract a draft profile and queue what is still missing.

    This is the "anything we missed" pass. It runs on a real extraction rather than on the
    interviewer's memory of the conversation, so what gets re-asked is what would actually
    have been blank or malformed on the printed resume.
    """
    s["phase"] = "gaps"
    s["followups"] = 0
    try:
        draft = await _build_profile(s)
    except HTTPException:
        gaps = []
    else:
        gaps = _profile_gaps(draft)

    # The PIN code is asked here unless the candidate has already said one, and the test is
    # their own words rather than the extracted profile: no section collects a location, and
    # asked for one it never heard, the extractor infers a plausible city from the college
    # name - it produced "Rajpura, 140401" on one pass and "Rajpura" on the next from the
    # same transcript. A resume must not carry an address nobody gave.
    pin_question = "the city they live in, and its 6-digit PIN code."
    gaps = [gap for gap in gaps if gap != pin_question]
    if not re.search(r"\b\d{6}\b", _said_by_candidate(s)):
        gaps.insert(0, pin_question)
    s["gaps"] = gaps[:MAX_GAP_QUESTIONS]
    _save(s)
    return {"gaps": len(s["gaps"]), "done": not s["gaps"]}


@router.post("/gap-check")
async def auto_gap_check(payload: dict):
    """Close the interview: check the draft profile and ask about whatever is missing.

    Called once, when a turn comes back with needs_gap_check. Returns the next question
    the same shape a turn does, or done=true when nothing needs clarifying.
    """
    s = _session(payload.get("session_id", ""))
    state = await enter_gap_phase(s)
    if state["done"] or s["turns"] >= MAX_TURNS:
        return {"reply": "", "done": True, "phase": "gaps", "section": "gaps",
                "collected": len(SECTIONS), "total": len(SECTIONS), "gaps": 0}
    result = await llm_extract(_turn_messages(s), _turn_system(s), TURN_SCHEMA,
                               model=TURN_MODEL)
    reply = (result.get("reply") or "").strip()
    s["messages"].append({"role": "assistant", "content": reply})
    _save(s)
    return {"reply": reply, "done": False, "phase": "gaps", "section": "gaps",
            "collected": len(SECTIONS), "total": len(SECTIONS), "gaps": len(s["gaps"])}


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
    # PIN validation performs a live postal-directory request; keep it off the
    # FastAPI event loop so other sessions remain responsive.
    # Before the facts layer, never after: the institution matcher and PIN lookup
    # are Latin-only and silently miss anything still written in Devanagari.
    profile = await romanize_profile(profile)
    profile = await asyncio.to_thread(apply_facts, profile)
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
    return {
        "profile": profile,
        "corrections": profile.get("_corrections", []),
        # A PIN that is valid but disagrees with the spoken city is never auto-merged,
        # so the operator has to be told - the kiosk otherwise prints it unnoticed.
        "warnings": profile.get("_validation_warnings", []),
    }


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
    """Deliver the finished resume: email it to the candidate, and print it.

    Both are best-effort and reported separately, because the kiosk tells the candidate
    what actually happened - promising "it has been emailed to you" when no mail server
    is configured is worse than saying the print is the only copy.
    """
    s = _session(payload.get("session_id", ""))
    result = _print_saved_pdf(s)

    emailed, email_error, address = False, None, ""
    profile_file = s["dir"] / "profile.json"
    if profile_file.exists():
        profile = json.loads(profile_file.read_text())
        address = (profile.get("email") or "").strip()
        try:
            mailer.send_resume(address, profile.get("name", ""),
                               s["dir"] / "resume.pdf", s["dir"] / "resume.docx")
            emailed = True
        except RuntimeError as exc:
            email_error = str(exc)
    return {**result, "emailed": emailed, "email_error": email_error,
            "email_to": address if emailed else ""}


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
        "warnings": profile.get("_validation_warnings", []),
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
    audio = await asyncio.to_thread(synthesize_wav, text)   # blocking, CPU-bound
    return Response(content=audio, media_type="audio/wav")


async def turn_stream_socket(ws: WebSocket):
    """Stream one interview turn, speaking each sentence as soon as it is generated.

    The non-streaming POST /auto/turn produces nothing until the whole JSON reply is
    complete, and only then is the audio synthesized - several seconds of silence on a
    voice call. Here the reply is decoded out of the partial JSON as it arrives, split
    on sentence boundaries, and each sentence is synthesized and sent immediately, so
    the candidate hears the first words while the model is still writing the rest.

    Protocol, per turn:
        <- {"session_id": ..., "answer": ...}
        -> {"type": "sentence", "text": ...}   followed by a binary WAV frame for
           English; text only for Hindi/Punjabi, which the browser speaks itself
        -> {"type": "done", ...}               same payload as POST /auto/turn
        -> {"type": "error", "detail": ...}    client should fall back to POST
    """
    await ws.accept()
    try:
        request = await ws.receive_json()
        s = _session(request.get("session_id", ""))
        answer = (request.get("answer") or "").strip()
        if not answer:
            await ws.send_json({"type": "error", "detail": "Empty answer"})
            return

        # Kokoro is English-only (KPipeline lang_code="a"); Hindi and Punjabi are
        # spoken by the browser, so for those we stream the text and skip synthesis.
        speak_here = s.get("language", "en") == "en"
        _begin_turn(s, answer)

        raw, spoken, pending = "", "", ""
        async for delta in llm_stream(_turn_messages(s), _turn_system(s), TURN_SCHEMA,
                                      model=TURN_MODEL):
            raw += delta
            reply_so_far = extract_partial_string(raw, "reply")
            if len(reply_so_far) <= len(spoken) + len(pending):
                continue
            pending = reply_so_far[len(spoken):]
            sentences, pending = split_sentences(pending)
            for sentence in sentences:
                spoken += sentence if spoken.endswith(" ") or not spoken else " " + sentence
                await ws.send_json({"type": "sentence", "text": sentence})
                if speak_here:
                    # Kokoro is blocking and CPU-bound - never run it on the event loop.
                    audio = await asyncio.to_thread(synthesize_wav, sentence)
                    await ws.send_bytes(audio)

        try:
            result = json.loads(raw)
        except ValueError:
            # Constrained decoding makes this near-impossible, but a dropped connection
            # mid-stream would leave truncated JSON. Salvage whatever was spoken.
            result = {"reply": extract_partial_string(raw, "reply"), "covered": []}

        # Anything after the last sentence break (no trailing punctuation) is still unsaid.
        tail = (result.get("reply") or "")[len(spoken):].strip()
        if tail:
            await ws.send_json({"type": "sentence", "text": tail})
            if speak_here:
                await ws.send_bytes(await asyncio.to_thread(synthesize_wav, tail))

        await ws.send_json({"type": "done", **_finish_turn(s, result)})
    except WebSocketDisconnect:
        pass
    except HTTPException as exc:
        await ws.send_json({"type": "error", "detail": exc.detail})
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})


router.add_api_websocket_route("/listen", live_transcribe_socket)
router.add_api_websocket_route("/turn-stream", turn_stream_socket)
