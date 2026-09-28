"""One WebSocket that owns a whole resume interview: WS /interview.

Everything a session needs happens inside this connection - greeting, language choice,
the questions, listening, the profile, the PDF, the printer - so the browser holds no
conversation logic at all. It captures the microphone, plays what it is sent, and draws
what it is told. That is the point: the old flow spent six HTTP calls and two sockets per
interview with the state machine living in the page, which left no single place to make
the thing faster. Now there is one.

Protocol
--------
Client -> server
    binary frame            16 kHz mono int16 PCM from the microphone
    {"type": "start", "language": "en"|"hi"|"pa"|null}
                            begin. A language skips the spoken language question.
    {"type": "stop_listening"}
                            the candidate pressed Stop listening - end the answer now
    {"type": "playback_done"}
                            finished playing the last line (the server otherwise waits
                            out the clip's own duration)
    {"type": "text", "text": "..."}
                            an answer as text instead of speech. Used by the tests, which
                            would otherwise need a microphone to exercise the flow.
    {"type": "cancel"}      stop the session

Server -> client
    {"type": "say", "text": ..., "language": ...}   followed by one binary WAV frame
    {"type": "partial", "text": ...}                live transcript while they speak
    {"type": "heard", "text": ...}                  the answer that was accepted
    {"type": "state", "phase": ..., "collected": n, "total": m, "status": ...}
    {"type": "result", ...}                         urls, printed, emailed
    {"type": "done"} | {"type": "error", "detail": ...}

Endpointing (deciding an answer is finished) is done here rather than in the browser,
because it is one of the things we want to tune: it is a plain energy-and-silence rule,
and every threshold is a constant at the top of this file.
"""
import asyncio
import json
import os
import re
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import routes_auto
import storage
from assistant_script import SCRIPT, is_scripted, section_question
from engines import synthesize_wav, transcribe_audio
from security import websocket_origin_allowed

router = APIRouter(tags=["interview"])

# --- listening ---------------------------------------------------------------------
# Below this level a frame counts as silence. The browser used the same figure for its
# meter; measured speech sits far above it and room noise well below.
SPEECH_RMS = float(os.getenv("SPEECH_RMS", "0.012"))
# How long a pause ends an answer. The browser waited 3 seconds; the wait is now server
# side so it can be tuned in one place, and 1.2s is closer to a normal conversational gap.
END_SILENCE_S = float(os.getenv("END_SILENCE_S", "1.2"))
# Give up waiting for someone who never speaks, and re-ask.
NO_SPEECH_TIMEOUT_S = float(os.getenv("NO_SPEECH_TIMEOUT_S", "12"))
# A single answer cannot run forever.
MAX_UTTERANCE_S = float(os.getenv("MAX_UTTERANCE_S", "60"))
# How often to show what has been heard so far while the candidate is still talking.
PARTIAL_EVERY_S = 1.0
SAMPLE_RATE = 16000
# A partial is only shown, never kept, so it reads the tail of the answer rather than all
# of it. Running large-v3 over the whole buffer every second meant re-decoding a longer and
# longer utterance until the GPU watchdog killed the command buffer - after which Metal
# refuses every later submission from the process and the kiosk is deaf for good.
PARTIAL_WINDOW_S = 12

# How many times a question is re-asked before the interview moves on regardless.
MAX_UNCLEAR = 3
# Longest a single line may take to synthesize before the interview carries on without it.
SPEAK_TIMEOUT_S = float(os.getenv("SPEAK_TIMEOUT_S", "45"))

LANGUAGE_WORDS = {
    "en": ("english", "angrezi", "इंग्लिश", "अंग्रेज़ी", "अंग्रेजी", "ਅੰਗਰੇਜ਼ੀ"),
    "hi": ("hindi", "हिंदी", "हिन्दी", "ਹਿੰਦੀ"),
    "pa": ("punjabi", "panjabi", "पंजाबी", "ਪੰਜਾਬੀ"),
}


class Cancelled(Exception):
    """The candidate pressed Stop, or the socket went away."""


def detect_language(said: str) -> str | None:
    """Which language a spoken answer names, or None.

    Keyword matching first: it never mishears "English", costs nothing, and works while
    the models are still loading. Failing that, the script Whisper transcribed in is the
    answer - Devanagari means Hindi, Gurmukhi means Punjabi.
    """
    text = f" {said.lower()} "
    hits = [(text.find(word), code)
            for code, words in LANGUAGE_WORDS.items()
            for word in words if word in text]
    if hits:
        return min(hits)[1]
    if re.search("[ऀ-ॿ]", said):
        return "hi"
    if re.search("[਀-੿]", said):
        return "pa"
    return None


class Microphone:
    """The incoming audio, and whether the candidate has stopped talking.

    Deliberately dumb: energy over a threshold is speech, and a long enough gap after
    speech ends the answer. It is muted while the assistant is talking so the kiosk's own
    voice cannot be transcribed as an answer.
    """

    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.speech_started = False
        self.last_voice = 0.0
        self.muted = True

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.speech_started = False
        self.last_voice = time.monotonic()

    def feed(self, raw: bytes):
        if self.muted:
            return
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if not len(pcm):
            return
        self.buffer = np.concatenate([self.buffer, pcm])
        if float(np.sqrt(np.mean(np.square(pcm)))) > SPEECH_RMS:
            self.speech_started = True
            self.last_voice = time.monotonic()

    @property
    def seconds(self) -> float:
        return len(self.buffer) / SAMPLE_RATE


class Session:
    """One connection: the socket, the audio, and the interview state behind it."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.mic = Microphone()
        self.controls: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.state: dict | None = None      # the routes_auto session dict
        self.session_id: str | None = None
        self.language = "en"

    # --- plumbing ---------------------------------------------------------------
    async def reader(self):
        """Feed audio and control messages in until the socket closes."""
        try:
            while True:
                message = await self.ws.receive()
                if message.get("bytes") is not None:
                    self.mic.feed(message["bytes"])
                elif message.get("text") is not None:
                    try:
                        await self.controls.put(json.loads(message["text"]))
                    except ValueError:
                        pass                # a malformed frame is not worth dying over
                elif message.get("type") == "websocket.disconnect":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.closed = True
            # The reader is the first to know the candidate has gone. The interview task
            # may still be inside a synthesis call for another half a minute, so hand the
            # microphone back here rather than making the next person wait it out.
            if self.session_id:
                routes_auto.release_kiosk(self.session_id)
            await self.controls.put({"type": "cancel"})

    async def send(self, **payload):
        if self.closed:
            raise Cancelled()
        await self.ws.send_json(payload)

    def take_control(self) -> dict | None:
        try:
            return self.controls.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # --- speaking ---------------------------------------------------------------
    async def say(self, text: str, language: str | None = None):
        """Send a line and its audio, then wait for it to finish playing.

        The wait matters: without it the microphone opens while the speaker is still
        talking and the kiosk interviews itself. The client reports playback_done, and
        the clip's own duration is the fallback for a client that does not.
        """
        if not text:
            return
        language = language or self.language
        await self.send(type="say", text=text, language=language)

        self.mic.muted = True
        try:
            # Bounded: a misconfigured synthesizer hangs rather than raising - an espeak-ng
            # pointed at a data directory that does not exist froze the first uncached line
            # for minutes. Better a silent question, with its text on screen, than a kiosk
            # that stops mid-interview.
            audio = await asyncio.wait_for(
                asyncio.to_thread(synthesize_wav, text, language, is_scripted(text, language)),
                timeout=SPEAK_TIMEOUT_S)
        except (Exception, asyncio.TimeoutError) as exc:
            print(f"interview: could not speak a line ({type(exc).__name__}: {exc})")
            return
        if self.closed:
            raise Cancelled()
        await self.ws.send_bytes(audio)
        await self._await_playback(audio)

    async def _await_playback(self, wav: bytes):
        # 44-byte header, 16-bit mono samples.
        duration = max(0.0, (len(wav) - 44) / 2 / 24000)
        deadline = time.monotonic() + duration + 3.0
        while time.monotonic() < deadline:
            control = self.take_control()
            if control:
                if control.get("type") == "cancel":
                    raise Cancelled()
                if control.get("type") == "playback_done":
                    return
                await self.controls.put(control)       # not ours - leave it for hear()
            await asyncio.sleep(0.05)

    async def say_script(self, key: str, language: str | None = None):
        language = language or self.language
        phrases = SCRIPT.get(key, {})
        await self.say(phrases.get(language) or phrases.get("en", ""), language)

    # --- listening --------------------------------------------------------------
    async def hear(self) -> str:
        """One answer. Returns "" when the room stays quiet."""
        self.mic.reset()
        self.mic.muted = False
        await self.send(type="state", phase="listening", status="listening")

        started = time.monotonic()
        last_partial = started
        partial_task: asyncio.Task | None = None
        forced = False

        while True:
            control = self.take_control()
            if control:
                kind = control.get("type")
                if kind == "cancel":
                    raise Cancelled()
                if kind == "text":                  # the tests speak in text
                    self.mic.muted = True
                    said = str(control.get("text", "")).strip()
                    if said:
                        await self.send(type="heard", text=said)
                    return said
                if kind == "stop_listening":
                    forced = True

            now = time.monotonic()
            quiet_for = now - self.mic.last_voice
            if forced and self.mic.seconds > 0.3:
                break
            if self.mic.speech_started and quiet_for >= END_SILENCE_S:
                break
            if self.mic.seconds >= MAX_UTTERANCE_S:
                break
            if not self.mic.speech_started and now - started > NO_SPEECH_TIMEOUT_S:
                self.mic.muted = True
                return ""

            if (self.mic.speech_started and now - last_partial >= PARTIAL_EVERY_S
                    and (partial_task is None or partial_task.done())):
                last_partial = now
                partial_task = asyncio.create_task(self._send_partial(self.mic.buffer.copy()))
            await asyncio.sleep(0.05)

        self.mic.muted = True
        if partial_task and not partial_task.done():
            partial_task.cancel()
        await self.send(type="state", phase="thinking", status="transcribing")
        said = await asyncio.to_thread(transcribe_audio, self.mic.buffer, False, self.language)
        said = (said or "").strip()
        if said:
            await self.send(type="heard", text=said)
        return said

    async def _send_partial(self, buffer: np.ndarray):
        """Greedy pass over what has been said so far - discarded, only ever displayed."""
        if len(buffer) < SAMPLE_RATE // 2:
            return
        buffer = buffer[-PARTIAL_WINDOW_S * SAMPLE_RATE:]
        try:
            text = await asyncio.to_thread(transcribe_audio, buffer, True, self.language)
        except Exception:
            return
        if text and not self.closed:
            try:
                await self.send(type="partial", text=text)
            except Cancelled:
                pass


async def _choose_language(session: Session) -> str:
    """Greet in all three languages and take the answer. English if nobody answers."""
    for language in ("en", "hi", "pa"):
        await session.say(SCRIPT["askLanguage"][language], language)

    for attempt in range(MAX_UNCLEAR):
        said = await session.hear()
        if said:
            language = detect_language(said)
            if language:
                return language
            for code in ("en", "hi", "pa"):
                await session.say(SCRIPT["languageUnclear"][code], code)
        else:
            for code in ("en", "hi"):
                await session.say(SCRIPT["notHeard"][code], code)
    return "en"


async def _run_interview(session: Session):
    """The sections, then the gap pass. The state machine itself lives in routes_auto."""
    state = session.state
    question = await routes_auto.ask_question(state)
    state["messages"].append({"role": "assistant", "content": question})
    routes_auto.save_session(state)

    unclear = 0
    while True:
        await session.send(type="state", phase=state["phase"],
                           collected=min(state["index"], len(routes_auto.SECTIONS)),
                           total=len(routes_auto.SECTIONS), status="asking")
        await session.say(question)

        answer = await session.hear()
        if not answer:
            unclear += 1
            if unclear >= MAX_UNCLEAR:
                raise Cancelled()           # an empty room: reset for the next person
            await session.say_script("notHeard")
            continue
        unclear = 0

        await session.send(type="state", phase=state["phase"], status="thinking",
                           collected=min(state["index"], len(routes_auto.SECTIONS)),
                           total=len(routes_auto.SECTIONS))
        routes_auto.begin_turn(state, answer)
        question = await routes_auto.ask_question(state)
        turn = routes_auto.finish_turn(state, {"reply": question})

        if turn["needs_gap_check"]:
            await session.say(turn["reply"])            # "thank you, preparing it now"
            await session.send(type="state", phase="gaps", status="checking",
                               collected=len(routes_auto.SECTIONS),
                               total=len(routes_auto.SECTIONS))
            gap = await routes_auto.enter_gap_phase(state)
            if gap["done"]:
                return
            question = await routes_auto.ask_question(state)
            state["messages"].append({"role": "assistant", "content": question})
            routes_auto.save_session(state)
            continue
        if turn["done"]:
            await session.say(turn["reply"])
            return


async def _deliver(session: Session):
    """Profile, files, printer, mail - each step announced as it happens."""
    state = session.state
    await session.send(type="state", phase="building", status="profile")
    profile = await routes_auto.build_profile(state)

    await session.send(type="state", phase="building", status="documents")
    files = routes_auto.build_documents(state, profile)

    await session.send(type="state", phase="delivering", status="printing")
    delivery = routes_auto.deliver(state)

    await session.send(type="result", profile=profile,
                       corrections=profile.get("_corrections", []), **files, **delivery)
    closing = ("finishedEmailed" if delivery["emailed"] and delivery["printed"] else
               "finishedEmailedNoPrint" if delivery["emailed"] else
               "finishedPrintOnly" if delivery["printed"] else "finishedSavedOnly")
    await session.say_script(closing)


async def interview_socket(ws: WebSocket):
    if not websocket_origin_allowed(ws.headers):
        await ws.close(code=1008)
        return
    await ws.accept()

    session = Session(ws)
    reader = asyncio.create_task(session.reader())
    try:
        start = await asyncio.wait_for(session.controls.get(), timeout=120)
        if start.get("type") != "start":
            await session.send(type="error", detail="expected a start message")
            return

        language = start.get("language")
        session.language = language if language in routes_auto.LANGUAGES \
            else await _choose_language(session)
        # Claim the microphone before announcing anything: a socket that is about to be
        # told the kiosk is busy should not first report that an interview is starting.
        try:
            session.session_id, session.state = routes_auto.create_session(session.language)
        except routes_auto.KioskBusy:
            # Another interview already owns the microphone. Saying so and closing is the
            # only safe answer: two interviews listening to one room record each other's
            # answers - see claim_kiosk().
            await session.send(type="error", detail="busy",
                               message="An interview is already in progress on this kiosk.")
            return
        await session.send(type="state", phase="sections", status="starting",
                           language=session.language, collected=0,
                           total=len(routes_auto.SECTIONS))
        await session.send(type="session", session_id=session.session_id,
                           language=session.language)

        await _run_interview(session)
        await _deliver(session)
        await session.send(type="done")
    except Cancelled:
        pass
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:
        print(f"interview: session failed ({type(exc).__name__}: {exc})")
        try:
            await session.send(type="error", detail=str(exc))
        except Exception:
            pass
    finally:
        session.closed = True
        reader.cancel()
        if session.session_id:
            storage.save_session(session.session_id, ended_at=time.time())
            # However this ended - finished, cancelled, or the candidate walking away -
            # the microphone goes back so the next person is not told the kiosk is busy.
            routes_auto.release_kiosk(session.session_id)
        try:
            await ws.close()
        except Exception:
            pass


router.add_api_websocket_route("/interview", interview_socket)
