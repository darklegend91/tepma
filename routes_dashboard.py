"""An operator's view of every interview this kiosk has taken.

The kiosk itself shows a candidate nothing but their own session. This is the other side
of it: who has been through, what came out, and - the reason it exists - which profiles
came out wrong, so a person can see the failures instead of waiting for somebody to
complain about a printed resume.

It reads MongoDB when Mongo is up and falls back to data/sessions on disk when it is not,
because the disk copy is the one the kiosk actually depends on and a dashboard that goes
blank whenever the database is down is worse than useless for diagnosing why.

It is OFF by default. Every record here is somebody's name, phone number, email and home
address, and this is a machine that stands in a public place: ENABLE_DASHBOARD=1 turns it
on, and it is still bound to the hosts in security.ALLOWED_HOSTS.
"""
import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import storage
from facts import PIN_RE, lookup_pincode, pin_ranges

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

ENABLED = os.getenv("ENABLE_DASHBOARD", "").strip().lower() in {"1", "true", "yes"}
DATA_DIR = Path(__file__).parent / "data" / "sessions"
STATIC = Path(__file__).parent / "static"

# Warnings the facts layer leaves on a profile, in words an operator can act on.
WARNING_TEXT = {
    "pincode_not_found": "PIN code does not exist",
    "pincode_place_not_matched": "PIN code does not match the town",
}


def _from_disk(limit: int) -> list[dict]:
    """Every saved interview, newest first, read straight from data/sessions."""
    rows = []
    for folder in sorted(DATA_DIR.glob("*"), reverse=True)[:limit]:
        profile_file = folder / "profile.json"
        if not profile_file.is_file():
            continue
        try:
            profile = json.loads(profile_file.read_text())
        except (OSError, ValueError):
            continue
        rows.append({"_id": folder.name, "profile": profile,
                     "files": {"pdf_url": f"/auto/resume/{folder.name}.pdf"}})
    return rows


def _address_problems(location: str, already: list) -> list[str]:
    """What is wrong with this address, checked offline.

    The facts layer records a warning when the postal directory rejects a PIN, but only
    for interviews taken since it learned to; the records already in the database were
    written before that. The two-digit prefix is enough to catch the worst of them
    without the network - it is what tells 999999 from a PIN code - and it is all this
    can honestly claim, since 166001 has a real Chandigarh prefix and is still not a
    PIN that was ever allocated.
    """
    if any("PIN" in problem for problem in already):
        return []
    if not location.strip():
        return ["no address"]
    match = PIN_RE.search(location)
    if not match:
        return ["no PIN code"]
    prefixes = pin_ranges()
    if prefixes and match.group(1)[:2] not in prefixes:
        return ["PIN code is not a real one"]
    return []


def _summarise(row: dict) -> dict:
    """One interview, reduced to what a person scanning a list needs to see."""
    profile = row.get("profile") or {}
    education = (profile.get("education") or [{}])[0]
    problems = [WARNING_TEXT.get(w.get("code"), w.get("code"))
                for w in profile.get("_validation_warnings") or []]
    # A resume with no name or no way to reach the candidate is a failed interview,
    # whatever the interview itself thought.
    if len((profile.get("name") or "").split()) < 2:
        problems.append("no full name")
    if not (profile.get("phone") or "").strip():
        problems.append("no phone number")
    problems.extend(_address_problems(profile.get("location") or "", problems))
    return {
        "id": row.get("_id"),
        "created_at": row.get("created_at"),
        "language": row.get("language"),
        "name": profile.get("name") or "",
        "role": profile.get("target_role") or "",
        "phone": profile.get("phone") or "",
        "email": profile.get("email") or "",
        "location": profile.get("location") or "",
        "degree": education.get("degree") or "",
        "institution": ", ".join(p for p in (education.get("institution"),
                                             education.get("location")) if p),
        "year": education.get("year") or "",
        "corrections": profile.get("_corrections") or [],
        "problems": problems,
        "printed": bool((row.get("delivery") or {}).get("printed")),
        "pdf_url": (row.get("files") or {}).get("pdf_url") or "",
    }


@router.get("/api/profiles")
async def profiles(limit: int = 200):
    """Every interview that produced a profile, newest first."""
    rows = [row for row in storage.recent_sessions(limit) if row.get("profile")]
    source = "mongodb"
    if not rows:
        rows, source = _from_disk(limit), "disk"
    summaries = [_summarise(row) for row in rows]
    return {
        "source": source,
        "database": storage.status(),
        "total": len(summaries),
        "with_problems": sum(1 for s in summaries if s["problems"]),
        "printed": sum(1 for s in summaries if s["printed"]),
        "languages": {language: sum(1 for s in summaries if s["language"] == language)
                      for language in ("en", "hi", "pa")},
        "profiles": summaries,
    }


@router.get("/api/addresses")
async def addresses(limit: int = 60):
    """Check every stored address against the postal directory, for real.

    The prefix test in _address_problems is offline and coarse: it cannot tell that
    166001 is not an allocated PIN, only that 999999 is not one at all. This asks India
    Post. It is a deliberate, separate request rather than part of loading the page,
    because it is one network round trip per address.
    """
    rows = [row for row in storage.recent_sessions(limit) if row.get("profile")]
    if not rows:
        rows = _from_disk(limit)

    def check(location: str) -> dict:
        match = PIN_RE.search(location or "")
        if not match:
            return {"verdict": "no PIN", "detail": ""}
        try:
            found = lookup_pincode(location)
        except Exception as exc:                # the directory is not a dependency
            return {"verdict": "could not check", "detail": type(exc).__name__}
        if found is None:
            return {"verdict": "not a real PIN",
                    "detail": f"India Post has no {match.group(1)}"}
        if not found.get("verified"):
            return {"verdict": "unverified",
                    "detail": f"offline guess: {found.get('state') or 'unknown'}"}
        if found.get("place_match") is False:
            return {"verdict": "town does not match",
                    "detail": f"{match.group(1)} is in "
                              f"{', '.join(found.get('districts') or []) or found.get('state')}"}
        return {"verdict": "matches",
                "detail": ", ".join(p for p in (found.get("district"), found.get("state")) if p)}

    checked = []
    for row in rows:
        location = (row.get("profile") or {}).get("location") or ""
        checked.append({"id": row.get("_id"),
                        "name": (row.get("profile") or {}).get("name") or "",
                        "location": location,
                        **await asyncio.to_thread(check, location)})
    return {"checked": checked}


@router.get("/api/profile/{session_id}")
async def profile(session_id: str):
    """One interview in full, including the transcript."""
    row = storage.get_session(session_id)
    if row is None:
        folder = DATA_DIR / session_id
        if not folder.is_dir() or not (folder / "profile.json").is_file():
            raise HTTPException(404, "No such interview")
        row = {"_id": session_id,
               "profile": json.loads((folder / "profile.json").read_text())}
        transcript = folder / "transcript.json"
        if transcript.is_file():
            row["transcript"] = json.loads(transcript.read_text())
    return row


@router.get("")
async def dashboard_page():
    return FileResponse(STATIC / "dashboard.html")
