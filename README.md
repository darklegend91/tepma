# TePMA — Voice-Operated Resume Builder

A fully local, voice-operated platform: an LLM interviews you over voice, extracts a
structured profile, and generates a resume PDF. Also includes a voice document assistant.

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
    faster-whisper kokoro soundfile httpx fpdf2

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

Then open <http://localhost:8000> in your browser (use a real browser and allow the
microphone). The first interview reply is slow while models load; after that it's fast.

Pages:

- `/auto` — **automated kiosk interview**: one click, hands-free, auto-prints at the end
- `/` — the manual resume interview (talk, then click **Generate resume**)
- `/docs-assistant` — ask for a stored document by voice
- `/test` — isolated STT / TTS testing tools

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

## Stop the project

```bash
# Stop the voice server: press Ctrl+C in its terminal, or:
pkill -f "uvicorn server:app"
```

```bash
# Stop Ollama: press Ctrl+C in its terminal, or:
pkill -f "ollama serve"
```

Check nothing is left running:

```bash
pgrep -l -f "uvicorn|ollama" || echo "all stopped"
```

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

## Code map

- `server.py` — FastAPI app, serves pages and mounts routers
- `engines.py` — model layer: Whisper, Kokoro, Ollama client (all config at the top)
- `routes_interview.py` — `/interview/*`: chat, speak, listen (WS), finish, resume download
- `routes_test.py` — `/test/*`: isolated STT/TTS endpoints
- `routes_documents.py` — `/documents/*`: list, LLM match, view, print
- `ws_stt.py` — shared live-transcription WebSocket handler
- `profile_schema.py` — resume JSON schema + extraction prompt
- `resume_pdf.py` — profile JSON → resume PDF
- `make_sample_docs.py` — generates test documents
- `static/` — frontend: `index.html` (interview), `documents.html`, `test.html`, `voice.js` (shared mic/TTS engine), `style.css`
