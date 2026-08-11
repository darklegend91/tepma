"""Resume profile JSON schema (Ollama structured-output format) and extraction prompt."""

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

Rules:
- Use ONLY information the candidate actually stated. Never invent employers, dates, \
degrees, or numbers. Leave a field as an empty string or empty list if it was not covered.
- Write experience bullets in strong resume style: start with an action verb, include \
numbers and impact the candidate mentioned.
- Write a 2-3 sentence professional summary based on the whole conversation.
- Fix obvious speech-recognition errors (e.g. "gee mail" -> "gmail") but do not guess \
spellings of names the candidate spelled out; use exactly what they spelled.
- skills should be short entries like "Python" or "React", not sentences.
- The interview may be in English, Hindi, or Punjabi (often mixed). ALWAYS write the \
resume itself in professional English, translating what the candidate said.
- Indian context: keep institution names as the candidate said them (they are corrected \
automatically later),put any 6-digit PIN code into the location field , and make sure the phone numbers are 10 digits and in resume its written as +91 then the phone number."""
