# TePMA — Voice Document and Resume Assistant

A fully local, voice-operated platform. The new unified assistant first asks whether the
user wants a document or a résumé. It can select a stored PDF, collect details to generate
a missing application, or run a complete résumé interview and create PDF and Word files.

Everything runs on your own machine — no cloud APIs:

| Piece | Model / tech |
|---|---|
| Speech-to-text | faster-whisper (`small`) |
| Text-to-speech | Kokoro-82M |
| LLM (interviewer + extraction) | Qwen3-8B via Ollama |
| Backend | FastAPI (Python 3.11) |
| PDF generation | fpdf2 |

## First-time setup

```bash
# 1. Install Ollama (if not installed) and pull the model
brew install ollama
ollama pull qwen3:8b

# 2. Create the venv and install dependencies
python3.11 -m venv .venv
.venv/bin/pip install fastapi "uvicorn[standard]" python-multipart \
    faster-whisper kokoro soundfile httpx fpdf2 python-docx

# 3. (Optional) regenerate the sample documents
.venv/bin/python make_sample_docs.py
```

Whisper (~500 MB) and Kokoro (~330 MB) download automatically on first use, then work offline.

## Start the project

```bash
# 1. Start the LLM server (leave running)
ollama serve
```

```bash
# 2. In another terminal, start the voice server
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000/assistant> in your browser (use a real browser and allow
the microphone). The first voice reply is slow while models load; after that it is faster.

Pages:

- `/assistant` — **new unified workflow**: choose document or résumé
- `/auto` — **automated kiosk interview**: one click, hands-free, auto-prints at the end
- `/` — the manual resume interview (talk, then click **Generate resume**)
- `/docs-assistant` — ask for a stored document by voice
- `/test` — isolated STT / TTS testing tools

### The unified flow (`/assistant`)

The first screen asks **I want a document** or **I want a résumé** and offers English,
Hindi, and Punjabi conversation modes. Every answer can be typed or spoken. Voice capture
is submitted after three seconds of silence or immediately with **Stop listening & send
answer**.

Document branch:

```
description -> search stored PDFs
            -> match: save/open -> print only if printer is ready
            -> no match: collect required application details -> PDF + Word
            -> official record request: refuse instead of fabricating it
```

Generated applications are based only on details supplied by the user. Identity documents,
marksheets, certificates, licences, prescriptions, and government/court-issued records are
not generated. The final application is saved even when no printer is connected.

Résumé branch:

```
name -> target role -> education -> experience -> projects -> skills
     -> achievements -> contact -> location/PIN -> PDF + Word -> optional print
```

Both branches show progress at each stage. A red/green status bar at the bottom reports the
current default printer. If it says **Printer: Not connected**, the PDF is saved and
downloaded rather than treated as an interview failure.

### The automated flow (`/auto`)

Click **Start interview** once; everything after that is hands-free:

```
greet -> listen -> answer -> next question -> ... -> auto-conclude
      -> extract profile -> apply facts -> PDF + Word -> send to printer
```

The server tracks which of nine resume topics have been answered (`TOPICS` in
`routes_auto.py`) and tells the model what is still missing each turn. That is what stops
the interview looping on a question, and it guarantees the interview ends by itself —
either when every topic is covered or after `MAX_TURNS` (22) as a hard stop.

Conversation state lives on the **server** and is written to disk every turn, so a refresh
or crash never loses an interview. Errors surface as popups, and printer failures are
non-fatal: the resume is always saved and downloadable even if printing fails.

## Stop the project and test tunnel

The normal way to stop everything is to press **Ctrl+C** once in each terminal where
`uvicorn`, `ollama serve`, or `cloudflared tunnel` is running.

If those terminals are no longer available, stop all three processes from a new terminal:

```bash
# Stop the public Cloudflare test link
pkill -f "cloudflared tunnel"

# Stop the TePMA web/voice server
pkill -f "uvicorn server:app"

# Unload the LLM, then stop the Ollama server
ollama stop qwen3:8b 2>/dev/null || true
pkill -f "ollama serve"
```

Check that nothing is left running:

```bash
pgrep -l -f "cloudflared tunnel|uvicorn server:app|ollama serve" || echo "all stopped"
```

Once `cloudflared` stops, the temporary `trycloudflare.com` URL stops working. A new
tunnel will create a new URL the next time it is started.

If `ollama serve` ever says `address already in use`, it is already running — don't start
it twice. To free ~5 GB RAM without stopping the server: `ollama stop qwen3:8b`.

## Configuration

Settings live in `.env` (loaded by `engines.py`, with safe defaults if the file is absent):

```
OLLAMA_URL   = "http://127.0.0.1:11434"
LLM_MODEL    = "qwen3:8b"
WHISPER_MODEL= "small"
TTS_VOICE    = "af_heart"
TTS_RATE     = 24000
```

`TEPMA_UNICODE_FONT` may optionally point to a local Unicode `.ttf` font used by generated
documents. On macOS the app automatically uses Arial Unicode when it is available.

To use a fine-tuned model, change `LLM_MODEL` — no code edits needed.

Email sending reads `SMTP_USER` / `SMTP_PASS` from the environment (never commit these).

## Accuracy: the facts layer

`facts.py` deterministically corrects every extracted profile — this is *not* the LLM's job:

- **Date awareness** — today's date is injected into prompts, so "two years back" resolves correctly
- **Indian institutions** — fuzzy-matched against `data/reference/universities.json`
  (`"Thapadi University"` → `"Thapar Institute of Engineering and Technology"`)
- **PIN codes** — validated and the state auto-filled from `data/reference/pincode_ranges.json`
- **Phone numbers** — normalised to `+91 XXXXXXXXXX`

Every change is recorded in the profile's `_corrections` list. To improve accuracy, add
entries to the reference JSON files — no retraining required.

## Fine-tuning (optional)

See **`finetune/GUIDE.md`** for the full course. Short version:

```bash
.venv/bin/python finetune/generate_data.py --count 200   # ~2.5 min/sample, resumable
.venv/bin/python finetune/build_dataset.py               # -> train.jsonl / eval.jsonl
# then run finetune/TePMA_finetune_colab.ipynb on a free Colab T4 GPU
.venv/bin/python finetune/eval_model.py --model qwen3:8b # baseline to beat
```

## Where data is saved

- `data/sessions/<timestamp>/` — per-interview `transcript.json`, `profile.json`, `resume.pdf`
- `data/documents/` — the document library (drop any PDF here; it becomes voice-searchable)
- `data/generated_documents/<timestamp>/` — generated application workflow, PDF, Word, JSON

## Run automated tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover stored-document matching, safe printer fallback, guided application
generation, PDF/DOCX rendering, refusal of official-document generation, and the new page.

## Code map

- `server.py` — FastAPI app, serves pages and mounts routers
- `engines.py` — model layer: Whisper, Kokoro, Ollama client (all config at the top)
- `routes_interview.py` — `/interview/*`: chat, speak, listen (WS), finish, resume download
- `routes_test.py` — `/test/*`: isolated STT/TTS endpoints
- `routes_documents.py` — `/documents/*`: list, LLM match, view, print
- `routes_assistant.py` — `/assistant/api/*`: unified document matching/generation workflow
- `document_render.py` — controlled application JSON → PDF and editable Word
- `ws_stt.py` — shared live-transcription WebSocket handler
- `profile_schema.py` — resume JSON schema + extraction prompt
- `resume_pdf.py` — profile JSON → resume PDF
- `make_sample_docs.py` — generates test documents
- `static/` — frontend pages including `assistant.html`; `voice.js` is the shared mic/TTS engine
- `tests/test_assistant.py` — unified workflow and renderer tests
