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
import os
import re
import secrets
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engines import (TURN_MODEL, extract_partial_string, llm_extract, llm_stream,
                     split_sentences, synthesize_wav)
import mailer
import storage
from security import MAX_SPEAK_CHARS, websocket_origin_allowed
from assistant_script import is_scripted, section_question
from facts import EMAIL_RE, apply_facts, date_context, normalise_email, normalise_phone
from printer_status import default_printer_status, queue_is_empty
from profile_schema import (EXTRACTOR_PROMPT, PROFILE_SCHEMA, english_transcript,
                            romanize_profile)
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
    # The town matters as much as the name. "ITI" alone is not an institution - there is
    # one in nearly every district - and a garbled name ("चिपकारे इन्विरसिटी") cannot be
    # matched against the reference list without somewhere to anchor it.
    ("education", "their education: highest qualification, the institution, the town or "
                  "city that institution is in, and the year."),
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
- Candidates are in India: expect Indian colleges, cities and PIN codes.
- Never promise anything that happens after this conversation. You are a machine in a \
room, not a recruiter: you cannot call them back, shortlist them, forward their profile, \
or be in touch. This is a kiosk that prints a resume before they walk away, and "we will \
contact you soon" is a lie that sends people home waiting for a call that will never \
come."""


def _needs(s: dict) -> tuple[str | None, str | None]:
    """(what this turn must get, what comes after it) - (None, _) means the interview is over."""
    if s["phase"] == "sections":
        current = SECTIONS[s["index"]][1] if s["index"] < len(SECTIONS) else None
        following = SECTIONS[s["index"] + 1][1] if s["index"] + 1 < len(SECTIONS) else None
        return current, following
    gaps = s["gaps"]
    return (GAP_DESCRIPTIONS.get(gaps[0]) if gaps else None,
            GAP_DESCRIPTIONS.get(gaps[1]) if len(gaps) > 1 else None)


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
                "Give a short thank-you saying their resume is being prepared right now. "
                "Do not say anyone will contact them, get back to them, or be in touch.)")
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
    """Persist the interview: to disk always, and to MongoDB when it is reachable.

    The file is the copy the kiosk itself depends on - it is what a refresh or a crash
    recovers from - so it is written first and unconditionally. Mongo is the queryable
    record on top of it, and never a reason for an interview to fail.
    """
    touch_kiosk(s["dir"].name)      # this interview is alive; keep its claim on the mic
    (s["dir"] / "transcript.json").write_text(
        json.dumps(s["messages"], indent=2, ensure_ascii=False)
    )
    storage.save_session(
        s["dir"].name,
        language=s.get("language", "en"),
        phase=s.get("phase"),
        section=(SECTION_IDS[s["index"]] if s.get("phase") == "sections"
                 and s["index"] < len(SECTIONS) else s.get("phase")),
        turns=s.get("turns", 0),
        gaps=s.get("gaps", []),
        transcript=s["messages"],
    )


# How long an interview may go untouched before the kiosk decides nobody is standing
# there any more. Every turn refreshes it, so this only expires on a candidate who walked
# away or a browser tab that was closed mid-question.
KIOSK_IDLE_S = float(os.getenv("KIOSK_IDLE_S", "90"))

# The interview that currently owns the microphone: {"id": str, "at": float}, or None.
_active: dict | None = None


class KioskBusy(Exception):
    """Somebody else is already being interviewed on this machine."""


def claim_kiosk(session_id: str) -> None:
    """Take the microphone for this interview, or refuse.

    There is one microphone and one printer, and nothing used to stop two interviews
    running against them at once. Two browser tabs on the same machine did exactly that:
    both greeted the candidate, both listened to the same room, and each recorded the
    answers the *other* one had asked for. The result was a printed resume for a man named
    Ajay who does not exist, assembled from two interleaved conversations - the candidate's
    real name never reached either transcript.
    """
    global _active
    if _active and _active["id"] != session_id \
            and time.time() - _active["at"] < KIOSK_IDLE_S:
        raise KioskBusy(_active["id"])
    _active = {"id": session_id, "at": time.time()}


def touch_kiosk(session_id: str) -> None:
    """Note that this interview is still going, so it keeps its claim."""
    if _active and _active["id"] == session_id:
        _active["at"] = time.time()


def release_kiosk(session_id: str) -> None:
    """Hand the microphone back, at the end of an interview or when one is abandoned."""
    global _active
    if _active and _active["id"] == session_id:
        _active = None


def create_session(language: str) -> tuple[str, dict]:
    """A new interview, registered and on disk. Shared by POST /auto/start and WS /interview.

    Raises KioskBusy when another interview already has the microphone.

    The id is the only thing between /auto/resume/<id>.pdf and a stranger's phone number
    and address, so the random part is 64 bits. It used to be 16 - one guess in 65,536 per
    second of the timestamp, which is an afternoon's brute force.
    """
    session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(8)}"
    claim_kiosk(session_id)     # before anything is written: a refused start leaves no trace
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    s = {
        "messages": [{"role": "user", "content":
                      "(The candidate has joined the voice call. Greet them and begin the interview.)"}],
        "turns": 0, "dir": session_dir, "language": language,
        "index": 0, "followups": 0, "phase": "sections", "gaps": [], "gaps_asked": 0,
    }
    _sessions[session_id] = s
    _save(s)
    return session_id, s


@router.get("/status")
async def auto_status():
    """Is somebody already being interviewed here?

    The page asks before it opens its mouth. A second tab that only found out at
    /auto/start would already have said the whole greeting out loud, over the top of the
    interview actually in progress.
    """
    return {"busy": bool(_active and time.time() - _active["at"] < KIOSK_IDLE_S)}


# Public names for the pieces WS /interview drives. The underscored originals stay so the
# HTTP endpoints below read the same as they always did.
@router.post("/start")
async def auto_start(payload: dict):
    """Begin an interview: create a session and return the spoken greeting."""
    language = payload.get("language", "en")
    if language not in LANGUAGES:
        language = "en"
    # The id is the only thing between /auto/resume/<id>.pdf and a stranger's phone number
    # and address, so the random part is 64 bits. It used to be 16 - one guess in 65,536
    # per second of the timestamp, which is an afternoon's brute force.
    try:
        session_id, s = create_session(language)
    except KioskBusy:
        # 409, not 500: the caller is a second browser tab, and the right thing for it to
        # do is stop talking rather than compete for the room.
        raise HTTPException(409, "An interview is already in progress on this kiosk.")
    reply = await _ask_question(s)
    s["messages"].append({"role": "assistant", "content": reply})
    _save(s)
    return {"session_id": session_id, "reply": reply, "done": False,
            "language": language, "section": SECTION_IDS[0], "gap": None,
            "collected": 0, "total": len(SECTIONS)}


# A spoken question is one or two sentences. Capping generation keeps a turn bounded and
# stops a small model looping in an unfamiliar script - see llm_extract's max_tokens.
TURN_MAX_TOKENS = 220


# Languages the interviewer model is trusted to phrase questions in. Punjabi is not one of
# them: qwen3:4b-instruct produced partly meaningless Gurmukhi ("ਪਤੱਤੀ" for letter,
# "10-ਅੱਖਰ ਮੋਬਾਈਲ ਨੰਬਰ" for a 10-digit number) and once invented an email address inside the
# question. The scripted wording is stiffer but correct, and it is already in the speech
# cache. Set PHRASED_LANGUAGES to override.
PHRASED_LANGUAGES = {code.strip() for code in
                     os.getenv("PHRASED_LANGUAGES", "en,hi").split(",") if code.strip()}


async def _ask_question(s: dict) -> str:
    """The next question, phrased by the model - or the scripted one if it cannot.

    An interview that dies in the middle loses everything the candidate has already said,
    so any model failure here degrades to the fixed wording for the section instead.
    """
    language = s.get("language", "en")
    # Nothing left to ask is not a question, so the model never phrases it. Asked for a
    # thank-you it reliably added one of its own: "we will contact you soon" - a promise
    # from a machine that prints a resume and forgets you, which sends people home waiting
    # for a call. The scripted line says only what is true, and it comes from the cache.
    if _current_need(s) is None:
        return _scripted_question(s, language)
    if language not in PHRASED_LANGUAGES:
        return _scripted_question(s, language)
    try:
        result = await llm_extract(_turn_messages(s), _turn_system(s), TURN_SCHEMA,
                                   model=TURN_MODEL, max_tokens=TURN_MAX_TOKENS)
        reply = (result.get("reply") or "").strip()
        if reply:
            return reply
    except Exception as exc:
        print(f"interview: falling back to the scripted question ({exc})")
    return _scripted_question(s, language)


def _scripted_question(s: dict, language: str) -> str:
    """The fixed wording for whatever the interview is collecting right now."""
    if _current_need(s) is None:
        return section_question("closing", language)
    if s["phase"] == "gaps":
        return section_question(f"gap_{s['gaps'][0]}", language) if s["gaps"] \
            else section_question("closing", language)
    section = SECTION_IDS[s["index"]] if s["index"] < len(SECTIONS) else "gaps"
    return section_question(section, language)


@router.post("/turn")
async def auto_turn(payload: dict):
    """Submit the candidate's answer, get the next question (or done=true)."""
    s = _session(payload.get("session_id", ""))
    answer = (payload.get("answer") or "").strip()
    if not answer:
        raise HTTPException(400, "Empty answer")

    _begin_turn(s, answer)
    return _finish_turn(s, {"reply": await _ask_question(s)})


def _asking_for(s: dict) -> str | None:
    """Which field this turn is collecting, section or gap alike."""
    if s["phase"] == "gaps":
        return s["gaps"][0] if s["gaps"] else None
    return SECTION_IDS[s["index"]] if s["index"] < len(SECTIONS) else None


# Digits as people say them out loud, in the three languages the kiosk speaks. A number
# read out in words is a perfectly good answer that normalise_phone cannot yet turn into
# digits - the extractor does that later - so it must not be sent back as a non-answer.
_DIGIT_WORDS = (
    "zero one two three four five six seven eight nine oh double triple "
    "शून्य सुन्न जीरो एक दो तीन चार पांच पाँच छह छे सात आठ नौ "
    "ਸਿਫ਼ਰ ਸਿਫਰ ਜ਼ੀਰੋ ਇੱਕ ਇਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌਂ ਨੌ"
).split()
# Split on whitespace, not on a word-character class: Python's \w excludes Devanagari and
# Gurmukhi combining vowel marks, so "पांच" came back as "प" and "च" and was never counted.
_EDGE_PUNCTUATION = ''' \t\n.,!?;:"'()[]{}-–—।॥'''


def _spoken_digits(answer: str) -> int:
    """How many digits this answer contains, counting the ones said as words."""
    return sum(1 for word in answer.lower().split()
               if word.strip(_EDGE_PUNCTUATION) in _DIGIT_WORDS)


def _answered_it(field: str | None, answer: str) -> bool:
    """Did that answer actually produce the thing that was asked for?

    Two fields have a right shape, and an answer that does not have it is not an answer,
    however long it was. This is worth checking while the candidate is still standing
    there: one interview accepted a hallucinated lecture about the University of
    California as an email address and a nine-digit number as a mobile, and only found
    out at the very end, when the gap pass had to ask for both again.

    The same normalisers the facts layer uses decide it, so the test is exactly "would
    this have survived onto the resume". A wrong answer buys MAX_FOLLOWUPS re-asks and
    then the interview moves on regardless - somebody who has no email address must not
    be trapped at question three.
    """
    if field == "email":
        return bool(EMAIL_RE.match(normalise_email(answer)))
    if field == "phone":
        return normalise_phone(answer) != answer or _spoken_digits(answer) >= 10
    return _is_substantive(answer)


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

    if not (_answered_it(_asking_for(s), answer)
            or s["followups"] >= MAX_FOLLOWUPS):
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
        # Which gap is being asked about, so the page can show the answer against the
        # right field. "gaps" alone does not say whether this is the email or the PIN.
        "gap": s["gaps"][0] if s["phase"] == "gaps" and s["gaps"] else None,
        "collected": collected,
        "total": len(SECTIONS),
        "needs_gap_check": s["phase"] == "sections" and s["index"] >= len(SECTIONS),
    }


def _said_by_candidate(s: dict) -> str:
    """Everything the candidate actually said this session, lower-cased."""
    return " ".join(m["content"] for m in s["messages"]
                    if m["role"] == "user" and not m["content"].startswith("(")).lower()


# What "recorded with confidence" means, field by field. These are deliberately mechanical:
# a spoken email that never reached an "@" or a phone that lost digits is exactly the kind
# of thing the candidate should be given one more chance to correct before printing.
#
# Each gap carries a key and an English description. The key selects the scripted question
# (localized, and already in the speech cache); the description is what the model is told
# to ask about when it is phrasing the question itself.
GAP_DESCRIPTIONS = {
    "name": "their full name, first and last, spelled out letter by letter.",
    "email": "their email address, spelled out letter by letter.",
    "phone": "their 10-digit mobile number, digit by digit.",
    "target_role": "the job profile or role they are applying for.",
    "education": "their highest qualification, the institution, the town or city that "
                 "institution is in, and the year.",
    "experience": "their work experience, current job status, or any project they can show.",
    "skills": "the main skills they would want on their resume.",
    "location": "the city they live in, and its 6-digit PIN code.",
}


def _profile_gaps(profile: dict) -> list[str]:
    gaps = []
    name = (profile.get("name") or "").strip()
    if len(name.split()) < 2:
        gaps.append("name")
    email = (profile.get("email") or "").strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        gaps.append("email")
    digits = "".join(c for c in (profile.get("phone") or "") if c.isdigit())
    if len(digits) not in (10, 12):     # 10 digits, or 12 with the 91 country code
        gaps.append("phone")
    if not (profile.get("target_role") or "").strip():
        gaps.append("target_role")
    # An entry with no institution is not an education: "Carpentry, 2026" tells an
    # employer nothing, and it is what comes back when the answer landed on the wrong
    # question. The named institution is the test, not the presence of a list.
    if not any((entry.get("institution") or "").strip()
               for entry in profile.get("education") or []):
        gaps.append("education")
    if not profile.get("experience") and not profile.get("projects"):
        gaps.append("experience")
    if not profile.get("skills"):
        gaps.append("skills")
    # A PIN is the test, not a non-empty string: asked for a location it never heard, the
    # extractor will happily infer a plausible city from the college name. And six digits
    # are not a PIN: one resume went out reading "Rajpura, 166001" - Rajpura's PIN is
    # 140401, and 166001 is not a PIN at all. The postal directory says so, in a warning
    # the facts layer leaves on the profile, so ask again rather than print it.
    impossible_pin = any(w.get("code") in {"pincode_not_found", "pincode_place_not_matched"}
                         for w in profile.get("_validation_warnings") or [])
    if impossible_pin or not re.search(r"\b\d{6}\b", (profile.get("location") or "")):
        gaps.append("location")
    return gaps[:MAX_GAP_QUESTIONS]


async def enter_gap_phase(s: dict) -> dict:
    """After the fixed sections: extract a draft profile and queue what is still missing.

    This is the "anything we missed" pass. It runs on a real extraction rather than on the
    interviewer's memory of the conversation, so what gets re-asked is what would actually
    have been blank or malformed on the printed resume.
    """
    s["phase"] = "gaps"
    s["followups"] = 0
    draft: dict = {}
    try:
        draft = await _build_profile(s)
    except HTTPException:
        gaps = []           # extraction failed; the fixed questions still ran
    else:
        gaps = _profile_gaps(draft)

    # The PIN code is asked here unless the candidate has already said one, and the test is
    # their own words rather than the extracted profile: no section collects a location, and
    # asked for one it never heard, the extractor infers a plausible city from the college
    # name - it produced "Rajpura, 140401" on one pass and "Rajpura" on the next from the
    # same transcript. A resume must not carry an address nobody gave.
    said_a_pin = re.search(r"\b\d{6}\b", _said_by_candidate(s))
    # Read from the draft, not from `gaps`: the gap list is capped at MAX_GAP_QUESTIONS
    # and location sits at the end of it, so a truncated list would drop the very thing
    # the postal directory just objected to.
    bad_pin = any(w.get("code") in {"pincode_not_found", "pincode_place_not_matched"}
                  for w in draft.get("_validation_warnings") or [])
    gaps = [gap for gap in gaps if gap != "location"]
    # Ask for the PIN when none was ever spoken, and ask again when the one that was
    # spoken is not a real PIN code - in both cases the resume would otherwise carry an
    # address nobody can post a letter to.
    if not said_a_pin or bad_pin:
        gaps.insert(0, "location")
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
    reply = await _ask_question(s)
    s["messages"].append({"role": "assistant", "content": reply})
    _save(s)
    return {"reply": reply, "done": False, "phase": "gaps", "section": "gaps",
            "collected": len(SECTIONS), "total": len(SECTIONS), "gaps": len(s["gaps"])}


def _keep_spoken_pin(s: dict, profile: dict) -> dict:
    """Put back a PIN code the candidate said but the extractor left out.

    The gap pass asks for the PIN at the very end, so it is the last thing in the
    transcript - and the extractor drops it often enough to matter: a Hindi candidate
    answered "पिन कोड 140401" and the resume came out as just "Rajpura". The translation
    kept the PIN on every run; extraction lost it. Deterministic, and it only ever adds
    digits the candidate actually spoke - facts.py then validates the PIN and fills in the
    district and state.
    """
    said = re.findall(r"\b(\d{6})\b", _said_by_candidate(s))
    location = (profile.get("location") or "").strip()
    if said and not re.search(r"\b\d{6}\b", location):
        pin = said[-1]      # the latest mention: a correction supersedes the first answer
        profile["location"] = f"{location}, {pin}" if location else pin
        profile.setdefault("_corrections", []).append(
            {"field": "location", "from": location, "to": profile["location"]})
    return profile


async def _build_profile(s: dict) -> dict:
    real = [m for m in s["messages"] if not m["content"].startswith("(")]
    if len(real) < 2:
        raise HTTPException(400, "Interview too short to build a resume")

    transcript = "\n".join(
        f"{'Interviewer' if m['role'] == 'assistant' else 'Candidate'}: {m['content']}"
        for m in real
    )
    # Translated first when it is not already English: extracting straight from Devanagari
    # or Gurmukhi silently changes numbers. See english_transcript().
    as_spoken = transcript            # both copies ground the dates: the original carries
    transcript = await english_transcript(transcript)   # Indic digits, the translation the
                                                        # English words for years
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
    profile = _keep_spoken_pin(s, profile)
    profile = await asyncio.to_thread(apply_facts, profile,
                                      f"{as_spoken}\n{transcript}")
    session_dir = s["dir"]
    (session_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    storage.save_session(session_dir.name, profile=profile,
                         corrections=profile.get("_corrections", []))
    return profile


def _build_documents(s: dict, profile: dict) -> dict:
    session_dir = s["dir"]
    try:
        (session_dir / "resume.pdf").write_bytes(render_resume(profile))
        (session_dir / "resume.docx").write_bytes(render_resume_docx(profile))
    except Exception as e:
        raise HTTPException(500, f"Could not generate the resume document: {e}")
    files = {
        "session_id": session_dir.name,
        "pdf_url": f"/auto/resume/{session_dir.name}.pdf",
        "docx_url": f"/auto/resume/{session_dir.name}.docx",
        "saved": True,
    }
    storage.save_session(session_dir.name, files=files)
    return files


# How long to wait for the page to actually come out before saying it has.
PRINT_CONFIRM_S = float(os.getenv("PRINT_CONFIRM_S", "8"))


def _confirm_printed(name: str) -> dict:
    """Did the page actually print, or is it only sitting in the queue?

    lpr accepting a job has never meant paper. One resume was queued at 14:37 and was
    still there when CUPS disabled the whole printer at 14:50 for a filter failure - and
    the candidate had been told to collect it from the printer thirteen minutes earlier.
    So wait for the queue to drain, and check the printer did not fall over while it did.
    """
    deadline = time.monotonic() + PRINT_CONFIRM_S
    while time.monotonic() < deadline:
        state = default_printer_status()
        if not state["connected"]:
            return {"printed": False, "print_status": state["state"]}
        empty = queue_is_empty(name)
        if empty is None:
            break                       # cannot tell; do not claim either way
        if empty:
            return {"printed": True, "print_status": "printed"}
        time.sleep(0.5)
    # Accepted, not yet out. Truthful either way: the closing line the candidate hears
    # has a variant that does not promise a printout.
    return {"printed": False, "print_status": "queued"}


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
        return {**_confirm_printed(printer["name"]), "printer_name": printer["name"],
                "print_error": None}
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


def _deliver(s: dict) -> dict:
    """Email the finished resume to the candidate and print it, then record what happened.

    Both are best-effort and reported separately, because the kiosk tells the candidate
    what actually happened - promising "it has been emailed to you" when no mail server
    is configured is worse than saying the print is the only copy.
    """
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
    delivery = {**result, "emailed": emailed, "email_error": email_error,
                "email_to": address if emailed else ""}
    storage.save_session(s["dir"].name, delivery=delivery, completed_at=time.time())
    release_kiosk(s["dir"].name)    # done: the next person can walk up
    return delivery


@router.post("/print")
async def auto_print(payload: dict):
    """The last step of an interview: deliver the resume and report how it went."""
    return _deliver(_session(payload.get("session_id", "")))


@router.post("/finish")
async def auto_finish(payload: dict):
    """Compatibility endpoint: build, save, and conditionally print the resume."""
    s = _session(payload.get("session_id", ""))
    profile = await _build_profile(s)
    documents = _build_documents(s, profile)
    # The same delivery as /auto/print: this endpoint exists so a caller can do the whole
    # tail in one request, not so it can quietly behave differently.
    delivery = _deliver(s)

    return {
        **documents,
        **delivery,
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
    if len(text) > MAX_SPEAK_CHARS:
        raise HTTPException(413, "Line too long to speak")
    language = payload.get("language", "en")
    if language not in LANGUAGES:
        language = "en"
    audio = await asyncio.to_thread(synthesize_wav, text, language,      # blocking, CPU-bound
                                    is_scripted(text, language))
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
    if not websocket_origin_allowed(ws.headers):
        await ws.close(code=1008)       # opened by another website - see security.py
        return
    await ws.accept()
    try:
        request = await ws.receive_json()
        s = _session(request.get("session_id", ""))
        answer = (request.get("answer") or "").strip()
        if not answer:
            await ws.send_json({"type": "error", "detail": "Empty answer"})
            return

        language = s.get("language", "en")
        _begin_turn(s, answer)

        # The closing line is scripted, never generated - see _ask_question.
        if _current_need(s) is None:
            reply = _scripted_question(s, language)
            await ws.send_json({"type": "sentence", "text": reply})
            # cache=True: every candidate hears this one, so it comes off the disk.
            await ws.send_bytes(
                await asyncio.to_thread(synthesize_wav, reply, language, True))
            await ws.send_json({"type": "done", **_finish_turn(s, {"reply": reply})})
            return

        raw, spoken, pending = "", "", ""
        async for delta in llm_stream(_turn_messages(s), _turn_system(s), TURN_SCHEMA,
                                      model=TURN_MODEL, max_tokens=TURN_MAX_TOKENS):
            raw += delta
            reply_so_far = extract_partial_string(raw, "reply")
            if len(reply_so_far) <= len(spoken) + len(pending):
                continue
            pending = reply_so_far[len(spoken):]
            sentences, pending = split_sentences(pending)
            for sentence in sentences:
                spoken += sentence if spoken.endswith(" ") or not spoken else " " + sentence
                await ws.send_json({"type": "sentence", "text": sentence})
                # Kokoro is blocking and CPU-bound - never run it on the event loop.
                audio = await asyncio.to_thread(synthesize_wav, sentence, language)
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
            await ws.send_bytes(await asyncio.to_thread(synthesize_wav, tail, language))

        await ws.send_json({"type": "done", **_finish_turn(s, result)})
    except WebSocketDisconnect:
        pass
    except HTTPException as exc:
        await ws.send_json({"type": "error", "detail": exc.detail})
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})


router.add_api_websocket_route("/listen", live_transcribe_socket)
router.add_api_websocket_route("/turn-stream", turn_stream_socket)


# --- used by routes_session.py (WS /interview) ---------------------------------------
ask_question = _ask_question
begin_turn = _begin_turn
finish_turn = _finish_turn
build_profile = _build_profile
build_documents = _build_documents
deliver = _deliver
save_session = _save
