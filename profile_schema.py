"""Resume profile JSON schema (Ollama structured-output format) and extraction prompt."""
import json
import re

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "location": {"type": "string"},
        "target_role": {"type": "string"},
        "summary": {"type": "string"},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "institution": {"type": "string"},
                    # Where the institution is. An employer reading "ITI" learns nothing;
                    # "ITI, Hamirpur" is a place they can check.
                    "location": {"type": "string"},
                    "year": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": ["degree", "institution"],
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "company", "bullets"],
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "technologies": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "achievements": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "name", "email", "phone", "location", "target_role", "summary",
        "education", "experience", "projects", "skills", "achievements",
    ],
}

EXTRACTOR_PROMPT = """You are a resume writer. You are given the transcript of a voice \
interview with a job candidate. Extract their information into the given JSON structure.

THE MOST IMPORTANT RULE - the interview may be in English, Hindi or Punjabi, often mixed, \
but EVERY value you write must be in English, in the Latin alphabet. Never copy Devanagari \
or Gurmukhi text into any field, not even for names, cities or institutions:
- Translate what was said: "पेंटर" -> "Painter", "डिप्लोमा" -> "Diploma".
- Transliterate names and places rather than translating them: "आदित्य पठानिया" -> \
"Aditya Pathania", "राजपुरा" -> "Rajpura", "आई टी आई हमीरपुर" -> "ITI Hamirpur".
- Spoken email addresses are dictated, not spelled: "अदित्य एट दि रेट जीमेल डॉट कॉम" -> \
"aditya@gmail.com".

Rules:
- Use ONLY information the candidate actually stated. Never invent employers, dates, \
degrees, or numbers. Leave a field as an empty string or empty list if it was not covered.
- "target_role" is a JOB TITLE, the work a person does - "Carpenter", "Electrician", \
"Data Entry Operator". Never the trade or subject they named it by: a man who says he \
wants carpentry work is applying to be a "Carpenter", not to be "Carpentry".
- Never write a note about what you did not learn. "Startup (name not provided)", \
"Unknown", "N/A" - leave the field empty instead. The resume is the candidate's, and it \
is not the place for your remarks.
- "degree" is the qualification EARNED, written the way it appears on a certificate - \
"ITI - Carpenter Trade", "Diploma in Civil Engineering", "10th", "B.A.". Never an \
enrolment ("ITI Admission") and never the act of studying ("Did a course").
- Write experience bullets in strong resume style: start with an action verb, include \
numbers and impact the candidate mentioned.
- Write a 2-3 sentence professional summary based on the whole conversation.
- Fix obvious speech-recognition errors (e.g. "gee mail" -> "gmail") but do not guess \
spellings of names the candidate spelled out; use exactly what they spelled.
- skills should be short entries like "Python" or "React", not sentences.
- Indian context: keep institution names as the candidate said them (they are corrected \
automatically later),put any 6-digit PIN code into the location field , and make sure the phone numbers are 10 digits and in resume its written as +91 then the phone number."""


# Scripts a resume must never be typeset in. The extractor is told to write English only,
# but a local model asked for structured output tends to mirror the language it was given,
# and one Devanagari field is enough to break both the PDF and the institution matcher -
# so the rule is enforced here rather than merely requested in the prompt.
_INDIC_SCRIPTS = re.compile(
    "[ऀ-ॿ"      # Devanagari
    "਀-੿"       # Gurmukhi
    "ঀ-৿"       # Bengali
    "஀-௿"       # Tamil
    "ఀ-౿"       # Telugu
    "ഀ-ൿ]"      # Malayalam
)

ROMANIZE_PROMPT = """You convert resume fields into English for typesetting.

Each input is a resume field and its value. Return the English version of every value,
using the field name to decide how:
- name, institution, company, location: TRANSLITERATE, never translate. "आदित्य पठानिया"
  -> "Aditya Pathania", "राजपुरा" -> "Rajpura". A name that reads like nonsense is still
  transliterated as sounds ("कि चीज" -> "Ki Chij"), because it is what the microphone
  heard of a real person's name - translating it into English words invents a new one.
- email: a dictated address becomes a real one. "एट दि रेट" / "एड दिरेट" / "ऐट" all mean
  "@", and "डॉट" means ".", so "अदित्य एट दि रेट जीमेल डॉट कॉम" -> "aditya@gmail.com".
- every other field: translate the meaning into natural resume English. "पेंटर" ->
  "Painter", "डिज़ाइन" -> "Design", "डिप्लोमा" -> "Diploma".
- Keep digits, punctuation and any English already present exactly as they are.
- Add nothing that was not said."""

ROMANIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "source": {"type": "string"},
                    "english": {"type": "string"},
                },
                "required": ["field", "source", "english"],
            },
        },
    },
    "required": ["translations"],
}


def _indic_strings(value, found: set, field: str = "text"):
    """Collect (field, string) for every value that is not typesettable as Latin text.

    The field name travels with the string: "name" and "institution" have to be
    transliterated while everything else is translated, and the model cannot tell which
    is which from a bare list of words.
    """
    if isinstance(value, str):
        if _INDIC_SCRIPTS.search(value):
            found.add((field, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if not key.startswith("_"):        # _corrections and friends are bookkeeping
                _indic_strings(item, found, key)
    elif isinstance(value, list):
        for item in value:
            _indic_strings(item, found, field)
    return found


def _apply_translations(value, mapping: dict):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, dict):
        return {key: (item if key.startswith("_") else _apply_translations(item, mapping))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_apply_translations(item, mapping) for item in value]
    return value


async def romanize_profile(profile: dict) -> dict:
    """Rewrite any non-Latin profile field in English, recording each change.

    Runs before the facts layer on purpose: the university matcher and PIN lookup are
    Latin-only, so a Devanagari institution name can never be corrected until it is
    romanized here. A failure returns the profile untouched - a resume with Hindi in it
    is worse than one without, but losing the interview entirely is worse than both.
    """
    from engines import llm_extract

    sources = sorted(_indic_strings(profile, set()))
    if not sources:
        return profile
    request = [{"field": field, "value": value} for field, value in sources]
    try:
        result = await llm_extract(
            [{"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1)}],
            ROMANIZE_PROMPT,
            ROMANIZE_SCHEMA,
        )
    except Exception:
        return profile

    originals = {value for _, value in sources}
    mapping = {
        item["source"]: item["english"].strip()
        for item in result.get("translations", [])
        if isinstance(item, dict) and item.get("source") in originals
        and str(item.get("english", "")).strip()
        and not _INDIC_SCRIPTS.search(str(item.get("english", "")))
    }
    if not mapping:
        return profile
    profile = _apply_translations(profile, mapping)
    profile.setdefault("_corrections", []).extend(
        {"field": "language", "from": source, "to": english}
        for source, english in mapping.items()
    )
    return profile


# Extraction reads a Hindi or Punjabi transcript badly enough to change what it says. Asked
# to build a profile straight from Gurmukhi, qwen3:8b turned "ਲੇਟੈਂਸੀ ਚਾਲੀ ਪ੍ਰਤੀਸ਼ਤ ਘਟੀ" (40%)
# into 15% - twice out of two runs - while translating the transcript first and extracting
# from the English got it right both times. A wrong number on a printed resume is silent,
# plausible-looking damage, so the translation pass is worth the extra call.
TRANSLATE_PROMPT = """Translate this interview transcript into English, line by line.

- Keep every number, digit, percentage, year and identifier EXACTLY as spoken. A number
  word becomes its digits unchanged: "ਚਾਲੀ" and "चालीस" are 40, "ਬਾਰਾਂ ਹਜ਼ਾਰ" is 12,000.
- Transliterate names of people, places and institutions instead of translating them:
  "ਆਦਿਤਿਆ ਪਠਾਨੀਆ" -> "Aditya Pathania", "ਰਾਜਪੁਰਾ" -> "Rajpura".
- Keep the "Interviewer:" and "Candidate:" labels and the line structure.
- Add nothing, drop nothing, and never guess at something that was not said."""

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"english": {"type": "string"}},
    "required": ["english"],
}


async def english_transcript(transcript: str) -> str:
    """The transcript in English, or the original if translation fails.

    Only called when the transcript is not already Latin - an English interview pays
    nothing for this. A failure returns the original text: extracting from Gurmukhi is
    worse than extracting from English, but it is far better than losing the interview.
    """
    from engines import llm_extract

    if not _INDIC_SCRIPTS.search(transcript):
        return transcript
    try:
        result = await llm_extract([{"role": "user", "content": transcript}],
                                   TRANSLATE_PROMPT, TRANSLATE_SCHEMA)
    except Exception:
        return transcript
    english = (result.get("english") or "").strip()
    return english or transcript
