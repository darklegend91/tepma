"""Run a whole interview through WS /interview without a microphone.

This is the kiosk's end-to-end test: it connects to the one interview socket, answers
every question as a candidate would - in Hindi, Punjabi or English - and reports what
came out, so a change can be checked in a minute instead of by standing at the kiosk
and talking for one.

    .venv/bin/python scripts/interview_client.py hi

The answers are typed rather than spoken, so this exercises everything except the
microphone and Whisper: the state machine, the questions, the extraction, the resume and
the print. Speak into the browser at /assistant to test the two it skips.
"""
import asyncio
import json
import os
import re
import sys
import time

import websockets

URL = os.getenv("TEPMA_WS", "ws://127.0.0.1:8000/interview")
# Typed answers arrive the instant a question ends, which no person does: a candidate
# speaks for several seconds and the server then transcribes it. THINK_S puts that time
# back, so the measured question latency is the one a visitor would actually experience.
THINK_S = float(os.getenv("THINK_S", "0"))

# One candidate, three languages. Every answer carries something the resume must not lose:
# a spelled name, a ten-digit number, a year, and "forty percent" - the figure that used to
# come back as 15% when the model read Devanagari numerals.
ANSWERS = {
    "en": [
        "My name is Aditya Pathania, A D I T Y A, Pathania",
        "Nine eight seven six five four three two one zero",
        "aditya at gmail dot com",
        "I live in Rajpura, PIN 140401",
        "I am applying for a machine learning engineer role",
        "Computer engineering at Thapar Institute in Patiala, graduated in twenty twenty five",
        "I interned at a startup for six months, right now I am looking for a job",
        "I built a search pipeline that cut latency by forty percent for twelve thousand users",
    ],
    "hi": [
        "मेरा नाम आदित्य पठानिया है",
        "मेरा नंबर 9876543210 है",
        "आदित्य ऐट जीमेल डॉट कॉम",
        "मैं राजपुरा में रहता हूँ, पिन कोड 140401",
        "मुझे मशीन लर्निंग इंजीनियर की नौकरी चाहिए",
        "मैंने पटियाला के थापर इंस्टीट्यूट से कंप्यूटर इंजीनियरिंग की है, 2025 में पास हुआ",
        "मैंने एक स्टार्टअप में छह महीने इंटर्नशिप की, अभी नौकरी ढूंढ रहा हूँ",
        "मैंने एक सर्च पाइपलाइन बनाई जिससे लेटेंसी चालीस प्रतिशत कम हुई, बारह हज़ार यूज़र थे",
    ],
    "pa": [
        "ਮੇਰਾ ਨਾਮ ਆਦਿਤਿਆ ਪਠਾਨੀਆ ਹੈ",
        "ਮੇਰਾ ਨੰਬਰ 9876543210 ਹੈ",
        "ਆਦਿਤਿਆ ਐਟ ਜੀਮੇਲ ਡਾਟ ਕਾਮ",
        "ਮੈਂ ਰਾਜਪੁਰਾ ਵਿੱਚ ਰਹਿੰਦਾ ਹਾਂ, ਪਿੰਨ ਕੋਡ 140401",
        "ਮੈਨੂੰ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਇੰਜੀਨੀਅਰ ਦੀ ਨੌਕਰੀ ਚਾਹੀਦੀ ਹੈ",
        "ਮੈਂ ਪਟਿਆਲਾ ਦੇ ਥਾਪਰ ਇੰਸਟੀਚਿਊਟ ਤੋਂ ਕੰਪਿਊਟਰ ਇੰਜੀਨੀਅਰਿੰਗ ਕੀਤੀ ਹੈ, 2025 ਵਿੱਚ ਪਾਸ ਹੋਇਆ",
        "ਮੈਂ ਇੱਕ ਸਟਾਰਟਅੱਪ ਵਿੱਚ ਛੇ ਮਹੀਨੇ ਇੰਟਰਨਸ਼ਿਪ ਕੀਤੀ, ਹੁਣ ਨੌਕਰੀ ਲੱਭ ਰਿਹਾ ਹਾਂ",
        "ਮੈਂ ਇੱਕ ਸਰਚ ਪਾਈਪਲਾਈਨ ਬਣਾਈ ਜਿਸ ਨਾਲ ਲੇਟੈਂਸੀ ਚਾਲੀ ਪ੍ਰਤੀਸ਼ਤ ਘਟੀ, ਬਾਰਾਂ ਹਜ਼ਾਰ ਯੂਜ਼ਰ ਸਨ",
    ],
}

INDIC = re.compile("[ऀ-ॿ਀-੿]")


async def main():
    language = sys.argv[1] if len(sys.argv) > 1 else "en"
    if language not in ANSWERS:
        raise SystemExit(f"language must be one of {', '.join(ANSWERS)}")
    answers = list(ANSWERS[language])
    started = time.perf_counter()
    question_at = None
    waits, result, spoken = [], None, 0
    last_event = time.perf_counter()

    async with websockets.connect(URL, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "start", "language": language}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=600)
            gap = time.perf_counter() - last_event
            last_event = time.perf_counter()
            if isinstance(raw, (bytes, bytearray)):
                print(f"   +{gap:5.1f}s  AUDIO {len(raw)/2/24000:.1f}s")
                await ws.send(json.dumps({"type": "playback_done"}))
                continue
            message = json.loads(raw)
            kind = message.get("type")
            if kind != "partial":
                print(f"   +{gap:5.1f}s  {kind:8} "
                      f"{message.get('status') or message.get('phase') or ''}")
            if kind == "say":
                spoken += 1
                print(f"  SAY  {message['text'][:92]}")
            elif kind == "state" and message.get("status") == "listening":
                if question_at is not None:
                    waits.append(("question", time.perf_counter() - question_at))
                reply = answers.pop(0) if answers else ANSWERS[language][-1]
                if THINK_S:
                    await asyncio.sleep(THINK_S)
                print(f"  <-   {reply[:70]}")
                await ws.send(json.dumps({"type": "text", "text": reply}))
                question_at = time.perf_counter()
            elif kind == "state" and message.get("status") in {
                    "profile", "documents", "printing", "checking"}:
                waits.append((message["status"], time.perf_counter() - (question_at or started)))
                question_at = time.perf_counter()
            elif kind == "result":
                result = message
            elif kind == "error":
                print("  ERROR:", message.get("detail"))
                break
            elif kind == "done":
                break

    print(f"\n  wall: {time.perf_counter() - started:.1f}s   spoken lines: {spoken}")
    if result:
        profile = result.get("profile", {})
        blob = json.dumps(profile, ensure_ascii=False)
        print("  name:", profile.get("name"), "| email:", profile.get("email"),
              "| phone:", profile.get("phone"))
        print("  location:", profile.get("location"))
        print("  percentages:", re.findall(r"(\d+)\s*%", blob), "(candidate said 40)")
        print("  English only:", not INDIC.search(blob))
        print("  education:", json.dumps(profile.get("education"), ensure_ascii=False))
        print("  printed:", result.get("printed"), "| pdf:", result.get("pdf_url"))
    by_kind: dict[str, list[float]] = {}
    for name, seconds in waits:
        by_kind.setdefault(name, []).append(seconds)
    print("  timings:", ", ".join(
        f"{k} n={len(v)} median={sorted(v)[len(v)//2]:.1f}s" for k, v in by_kind.items()))


asyncio.run(main())
