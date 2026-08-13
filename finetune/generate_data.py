"""Generate synthetic training data, using the local Qwen3-8B as TEACHER (distillation).

Each sample: invent an Indian candidate persona -> simulate a full voice interview in
English / Hindi / Punjabi (with realistic speech-recognition noise and code-switching)
-> extract the gold profile JSON with the exact production prompt + schema.

See finetune/GUIDE.md for why the data looks like this.

Run:   .venv/bin/python finetune/generate_data.py --count 400
Safe to re-run: existing samples are skipped, so you can stop and resume anytime.

On Colab, point TEPMA_RAW_DIR at a Google Drive folder so samples survive a disconnect,
and raise --concurrency (with OLLAMA_NUM_PARALLEL set to match) to use the GPU properly:
       TEPMA_RAW_DIR=/content/drive/MyDrive/tepma/raw \
       python finetune/generate_data.py --count 400 --concurrency 8
"""
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engines
from engines import llm_extract
from facts import date_context
from profile_schema import EXTRACTOR_PROMPT, PROFILE_SCHEMA

# Overridable so Colab can write straight to mounted Drive and survive a disconnect.
RAW_DIR = Path(os.getenv("TEPMA_RAW_DIR") or Path(__file__).parent / "raw")

MAX_REQUEUES = 5  # how many times a whole batch may be retried after Ollama drops out

# Language mix. English dominates (most resumes are English) but Hindi/Punjabi and
# code-switching must be well represented or the model will answer them in English.
LANGUAGES = (["en"] * 5) + (["hi"] * 3) + (["pa"] * 2) + (["mixed"] * 3)

LANGUAGE_RULES = {
    "en": "The whole interview is in English (Indian English).",
    "hi": "The whole interview is in Hindi, written in Devanagari script. Technical words "
          "like 'software engineer', 'Python', 'B.Tech' stay in English, as real speakers do.",
    "pa": "The whole interview is in Punjabi, written in Gurmukhi script. Technical words "
          "stay in English, as real speakers do.",
    "mixed": "The interview is code-switched Hinglish: the interviewer and candidate mix "
             "Hindi (Devanagari) and English within the same sentences, like real Indian "
             "speech, e.g. 'मेरा naam Aditya hai, I am doing B.Tech'.",
}

ROLES = [
    "software engineering intern", "data analyst", "frontend developer", "DevOps engineer",
    "mechanical design engineer", "digital marketing executive", "graphic designer",
    "chartered accountant", "HR executive", "sales manager", "civil site engineer",
    "product manager", "machine learning engineer", "customer support lead",
    "content writer", "QA tester", "bank probationary officer", "pharmacist",
    "school teacher", "operations manager", "UI UX designer", "network engineer",
    "business analyst", "electrical engineer", "hotel management trainee",
]
LEVELS = [
    "a college student in their pre-final year with no job experience, only projects",
    "a fresh graduate with one internship",
    "a professional with 2-4 years of experience",
    "a senior professional with 8-15 years of experience changing companies",
    "a career switcher moving from a completely different field",
    "someone returning to work after a two year break",
]
QUIRKS = [
    "gives very short vague answers, so the interviewer must probe for details",
    "rambles and mixes several topics into one answer",
    "readily mentions numbers, percentages and team sizes",
    "has a name the speech recogniser garbles, and spells it out letter by letter",
    "corrects themselves mid-answer about a date",
    "is nervous and asks the interviewer to repeat a question once",
]
CITIES = [
    "Chandigarh 160014", "Patiala 147004", "Ludhiana 141001", "Amritsar 143001",
    "New Delhi 110001", "Gurugram 122001", "Noida 201301", "Jaipur 302001",
    "Mumbai 400001", "Pune 411001", "Bengaluru 560001", "Hyderabad 500001",
    "Chennai 600001", "Kolkata 700001", "Lucknow 226001", "Indore 452001",
    "Ahmedabad 380001", "Kochi 682001", "Bhopal 462001", "Dehradun 248001",
]
INSTITUTIONS = [
    "Thapar Institute of Engineering and Technology", "Chitkara University",
    "Chandigarh University", "Punjabi University Patiala", "Panjab University Chandigarh",
    "Guru Nanak Dev University Amritsar", "Lovely Professional University",
    "Indian Institute of Technology Ropar", "National Institute of Technology Jalandhar",
    "Delhi Technological University", "Vellore Institute of Technology",
    "Manipal Institute of Technology", "SRM Institute of Science and Technology",
    "Anna University", "Savitribai Phule Pune University", "Amity University",
    "Jamia Millia Islamia", "University of Delhi", "Osmania University",
    "Visvesvaraya Technological University",
]

TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["interviewer", "candidate"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["transcript"],
}

SIMULATOR_PROMPT = """You write realistic transcripts of voice interviews for a resume builder.

The interviewer is a friendly Indian professional who greets the candidate, asks ONE short
question at a time (name and target role, education, work experience, projects, skills,
achievements, then email, phone and address with PIN code), probes vague answers for
specific numbers and dates, and ends by saying the interview is complete.

The candidate's lines must read like real speech-to-text output: filler words, self
corrections, numbers sometimes spelled as words, and occasional mis-transcription of names,
colleges and emails (the interviewer then asks them to spell it out).

Write 14 to 20 alternating turns starting with the interviewer. Every fact must be concrete
and internally consistent. Use Indian names, cities, companies and colleges."""


async def make_sample(path: Path):
    lang = random.choice(LANGUAGES)
    persona = (
        f"The candidate is {random.choice(LEVELS)}, targeting a {random.choice(ROLES)} role, "
        f"studied at {random.choice(INSTITUTIONS)}, lives in {random.choice(CITIES)}, and "
        f"{random.choice(QUIRKS)}.\n\nLANGUAGE: {LANGUAGE_RULES[lang]}"
    )
    sim = await llm_extract(
        [{"role": "user", "content": f"{persona}\n\nWrite the interview transcript."}],
        SIMULATOR_PROMPT + date_context(),
        TRANSCRIPT_SCHEMA,
    )
    turns = sim["transcript"]
    if len(turns) < 10:
        raise ValueError(f"transcript too short ({len(turns)} turns)")

    text = "\n".join(
        f"{'Interviewer' if t['speaker'] == 'interviewer' else 'Candidate'}: {t['text']}"
        for t in turns
    )
    profile = await llm_extract(
        [{"role": "user", "content": f"Interview transcript:\n\n{text}"}],
        EXTRACTOR_PROMPT + date_context(),
        PROFILE_SCHEMA,
    )
    # quality gate: a sample the teacher botched would teach the student to botch it too
    if not profile.get("name") or not profile.get("target_role"):
        raise ValueError("extraction missing name/role")
    if not profile.get("skills"):
        raise ValueError("extraction found no skills")

    path.write_text(json.dumps(
        {"lang": lang, "persona": persona, "transcript": turns, "profile": profile},
        indent=2, ensure_ascii=False,
    ))
    return lang


async def wait_for_ollama(max_wait: int = 300) -> bool:
    """Ollama can be killed by memory pressure during a long run. Wait for it to come back
    rather than burning through the remaining samples with connection errors."""
    import httpx

    for _ in range(max_wait // 5):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                if (await client.get(f"{engines.OLLAMA_URL}/api/version")).status_code == 200:
                    return True
        except Exception:
            pass
        print("  ...waiting for Ollama to come back (run `ollama serve`)", flush=True)
        await asyncio.sleep(5)
    return False


def _is_conn_error(msg: str) -> bool:
    return "connect" in msg.lower() or "disconnect" in msg.lower()


async def one_sample(i: int) -> tuple[str | None, str | None]:
    """Generate sample i. Returns (lang, None) on success or (None, error message)."""
    out = RAW_DIR / f"sample_{i:04d}.json"
    for attempt in (1, 2):  # one retry: transient model hiccups are common
        try:
            lang = await make_sample(out)
            print(f"[{i}] ok ({lang})", flush=True)
            return lang, None
        except Exception as e:
            msg = str(e) or type(e).__name__
            if attempt == 2:
                print(f"[{i}] FAILED: {msg}", flush=True)
                return None, msg
    return None, "unreachable"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="samples in flight at once. Match OLLAMA_NUM_PARALLEL; "
                             "8 is reasonable on a Colab T4, 1 on a laptop.")
    args = parser.parse_args()
    if args.concurrency < 1:
        sys.exit("--concurrency must be at least 1")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing to {RAW_DIR}  (concurrency {args.concurrency})", flush=True)

    todo = [i for i in range(args.count) if not (RAW_DIR / f"sample_{i:04d}.json").exists()]
    done, failed, langs = 0, 0, {}

    # Batched rather than one big gather: a batch boundary is where we can notice that
    # Ollama has died and wait for it, instead of burning every remaining sample.
    queue = list(todo)
    requeues = 0
    while queue:
        batch, queue = queue[:args.concurrency], queue[args.concurrency:]
        results = await asyncio.gather(*(one_sample(i) for i in batch))

        errors = [err for _, err in results if err]

        # Whole batch died on connection errors: Ollama was almost certainly OOM-killed.
        # Put the batch back and wait for it rather than counting the samples as lost.
        # Capped, because /api/version can answer while generation still fails — without
        # the cap that combination requeues the same batch forever.
        if (len(errors) == len(batch) and all(_is_conn_error(e) for e in errors)
                and requeues < MAX_REQUEUES):
            if not await wait_for_ollama():
                print("Ollama did not return; stopping. Re-run to resume.")
                break
            requeues += 1
            queue = batch + queue
            continue

        for lang, err in results:
            if err:
                failed += 1
            else:
                done += 1
                langs[lang] = langs.get(lang, 0) + 1

    total = len(list(RAW_DIR.glob("sample_*.json")))
    print(f"\ngenerated {done} new ({failed} failed). language mix this run: {langs}")
    print(f"total samples on disk: {total}")


if __name__ == "__main__":
    asyncio.run(main())
