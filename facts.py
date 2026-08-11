"""Deterministic facts layer — the part that must NOT be fine-tuned.

Handles the knowledge an LLM should never be trusted to memorise:
  * today's date          -> injected into prompts at request time
  * Indian institutions   -> fuzzy-corrected from speech-recognition errors
  * PIN codes             -> validated, state auto-filled

See finetune/GUIDE.md, Lesson 1, for why this is a lookup table and not training data.
"""
import json
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

REF_DIR = Path(__file__).parent / "data" / "reference"

# A candidate name must be at least this similar to a canonical one to be corrected.
# Lower = more corrections but more wrong ones. 0.62 catches "Thapadi"->"Thapar" while
# leaving genuinely unknown colleges untouched.
MATCH_THRESHOLD = 0.62

_institutions: list[str] | None = None
_pin_ranges: dict[str, str] | None = None

LANG_NAMES = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}


def institutions() -> list[str]:
    global _institutions
    if _institutions is None:
        data = json.loads((REF_DIR / "universities.json").read_text())
        _institutions = data["institutions"]
    return _institutions


def pin_ranges() -> dict[str, str]:
    global _pin_ranges
    if _pin_ranges is None:
        data = json.loads((REF_DIR / "pincode_ranges.json").read_text())
        _pin_ranges = {k: v for k, v in data.items() if not k.startswith("_")}
    return _pin_ranges


# ---------------------------------------------------------------- date awareness

def date_context() -> str:
    """Text appended to system prompts so the model can resolve relative dates.

    Never train a date into weights - it is wrong the day after training.
    """
    today = date.today()
    academic_year = f"{today.year}-{str(today.year + 1)[2:]}" if today.month >= 6 else \
                    f"{today.year - 1}-{str(today.year)[2:]}"
    return (
        f"\n\nToday's date is {today:%d %B %Y}. The current academic year in India is "
        f"{academic_year}. Use this to resolve relative dates the candidate mentions "
        f"(for example 'two years back', 'last semester', 'final year') into actual years."
    )


# ------------------------------------------------------- institution correction

def _norm(s: str) -> str:
    s = s.lower()
    # expand common spoken/abbreviated forms so fuzzy matching lines up
    for short, full in (
        (r"\biit\b", "indian institute of technology"),
        (r"\bnit\b", "national institute of technology"),
        (r"\biiit\b", "indian institute of information technology"),
        (r"\biim\b", "indian institute of management"),
        (r"\biisc\b", "indian institute of science"),
        (r"\bbits\b", "birla institute of technology and science"),
        (r"\bvit\b", "vellore institute of technology"),
        (r"\bdtu\b", "delhi technological university"),
        (r"\bpu\b", "panjab university"),
        (r"\bgndu\b", "guru nanak dev university"),
        (r"\blpu\b", "lovely professional university"),
        (r"\bcu\b", "chandigarh university"),
    ):
        s = re.sub(short, full, s)
    s = re.sub(r"\b(university|institute|college|of|the|and|technology|engineering)\b", " ", s)
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


def correct_institution(name: str) -> tuple[str, float]:
    """Map a possibly-garbled institution name to a canonical one.

    Returns (name, score). If nothing is similar enough the input is returned
    unchanged with its best score - we never invent an institution.
    """
    if not name or not name.strip():
        return name, 0.0
    target = _norm(name)
    if not target:
        return name, 0.0
    best, best_score = name, 0.0
    for canonical in institutions():
        score = SequenceMatcher(None, target, _norm(canonical)).ratio()
        # a full token match ("thapar" inside both) is strong evidence
        if set(target.split()) & set(_norm(canonical).split()):
            score = max(score, 0.75)
        if score > best_score:
            best, best_score = canonical, score
    return (best, best_score) if best_score >= MATCH_THRESHOLD else (name, best_score)


# ------------------------------------------------------------- pincode handling

PIN_RE = re.compile(r"\b(\d{6})\b")


def lookup_pincode(text: str) -> dict | None:
    """Find a 6-digit PIN in text and resolve its region. None if absent/invalid."""
    m = PIN_RE.search(text or "")
    if not m:
        return None
    pin = m.group(1)
    if pin[0] == "0":  # Indian PINs never start with 0
        return None
    region = pin_ranges().get(pin[:2])
    if not region:
        return None
    return {"pincode": pin, "state": region}


# ------------------------------------------------------------ phone normalisation

def normalise_phone(raw: str) -> str:
    """Format an Indian mobile number as '+91 XXXXXXXXXX'.

    Deterministic formatting belongs here, not in the prompt: an LLM asked to format
    numbers will occasionally drop or invent a digit, whereas this cannot.
    Anything that is not a recognisable 10-digit Indian number is returned unchanged.
    """
    if not raw:
        return raw
    digits = re.sub(r"\D", "", raw)
    for prefix in ("91", "091", "0"):  # country code or trunk prefix
        if len(digits) > 10 and digits.startswith(prefix):
            digits = digits[len(prefix):]
            break
    if len(digits) == 10 and digits[0] in "6789":  # Indian mobiles start 6-9
        return f"+91 {digits}"
    return raw


# ------------------------------------------------------------------ entry point

def apply_facts(profile: dict) -> dict:
    """Run every deterministic correction over an extracted profile.

    Adds a "_corrections" list describing what changed, so the UI can show the
    candidate what was auto-fixed instead of silently rewriting their answers.
    """
    corrections = []

    for edu in profile.get("education", []):
        original = edu.get("institution", "")
        fixed, score = correct_institution(original)
        if fixed != original:
            edu["institution"] = fixed
            corrections.append({
                "field": "education.institution",
                "from": original, "to": fixed, "confidence": round(score, 2),
            })

    phone = profile.get("phone", "")
    fixed_phone = normalise_phone(phone)
    if fixed_phone != phone:
        profile["phone"] = fixed_phone
        corrections.append({"field": "phone", "from": phone, "to": fixed_phone,
                            "confidence": 1.0})

    location = profile.get("location", "")
    pin = lookup_pincode(location)
    if pin:
        if pin["state"].split(" /")[0].lower() not in location.lower():
            profile["location"] = f"{location.strip()}, {pin['state']}".strip(", ")
            corrections.append({
                "field": "location", "from": location, "to": profile["location"],
                "confidence": 1.0,
            })

    if corrections:
        profile["_corrections"] = corrections
    return profile
