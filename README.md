# TePMA — Voice Document and Resume Assistant

A fully local, voice-operated platform. The new unified assistant first asks whether the
user wants a document or a résumé. It can select a stored PDF, collect details to generate
a missing application, or run a complete résumé interview and create PDF and Word files.

Everything runs on your own machine — no cloud APIs:

| Piece | Model / tech |
|---|---|
| Speech-to-text | Whisper `large-v3` via MLX (Apple Silicon GPU); faster-whisper `small` elsewhere |
| Text-to-speech | Kokoro-82M |
| LLM (live interview questions) | Qwen3-4B-Instruct via Ollama |
| LLM (profile extraction) | Qwen3-8B via Ollama |
| Backend | FastAPI (Python 3.11) |
| PDF generation | fpdf2 |

## First-time setup

Install Ollama, then run one command:

```bash
brew install ollama        # macOS. Linux: curl -fsSL https://ollama.com/install.sh | sh
```

```bash
./scripts/tepma.sh setup
```

That creates the virtual environment, installs the Python dependencies, pulls both Ollama
models, downloads the Whisper and Kokoro weights, and pre-synthesizes every line the kiosk
speaks so the first visitor is greeted instantly rather than waiting on a cold model. It is
safe to run again at any time - everything it does is cached, so a second run just verifies
the machine is ready.

Expect ~10 GB of downloads the first time: qwen3:8b (5.2 GB), qwen3:4b-instruct (2.5 GB),
Whisper (3 GB `large-v3` on Apple Silicon, 500 MB `small` elsewhere) and Kokoro (330 MB).
After that the whole thing runs offline.

Setup also copies `.env.example` to `.env` if you do not have one. Every setting has a
working default, so you only need to edit it to change models or to enable email.

```bash
# Optional: regenerate the sample documents for the document assistant
.venv/bin/python make_sample_docs.py
```

## Start the project

One command starts both the LLM and the web server and waits until each is answering:

```bash
./scripts/tepma.sh start
```

```bash
./scripts/tepma.sh stop
```

`stop` also frees the models from memory (~7 GB). If Ollama was already running before you
started - yours, or another project's - it is left alone and only TePMA's models are
evicted; if this script started it, it is stopped too. To free the memory without stopping
the server, use `./scripts/tepma.sh unload`.

`status` shows what is installed, what is running, and which models are loaded; `restart`
does both. The port defaults to 8000 — override it with `PORT=8080 ./scripts/tepma.sh start`.
`stop` is scoped to the same host and port, so a second checkout running on another port is
left untouched. Both services log to `logs/`. Starting twice is safe: an already-running
Ollama is reused rather than launched again (a second `ollama serve` would fail with
`address already in use`).

Then open <http://localhost:8000/assistant> in your browser (use a real browser and allow
the microphone). The first voice reply is slow while models load; after that it is faster.

<details>
<summary>Starting the two services by hand instead</summary>

```bash
# 1. Start the LLM server (leave running)
ollama serve
```

```bash
# 2. In another terminal, start the voice server
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

```bash
# Free the models from memory without stopping anything else
ollama stop qwen3:8b && ollama stop qwen3:4b-instruct
```

</details>

Pages:

- `/assistant` — **new unified workflow**: choose document or résumé
- `/auto` — **automated kiosk interview**: one click, hands-free, auto-prints at the end
- `/` — the manual resume interview (talk, then click **Generate resume**)
- `/docs-assistant` — ask for a stored document by voice
- `/test` — isolated STT / TTS testing tools

### The unified flow (`/assistant`)

Nothing is typed anywhere: the whole session is spoken. One tap on the idle screen (browsers
refuse microphone access and audio playback until the page has been interacted with once),
and after that the only control is **Stop**.

```
greet in English, Hindi and Punjabi -> "which language?" -> "résumé or a document?"
```

The greeting is scripted rather than generated, and pre-synthesized into a speech cache at
startup, so it plays from disk instead of being spoken again for every visitor. Answers are
submitted after three seconds of silence. Silence three times in a row ends the session and
returns the kiosk to its idle screen for the next person.

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

Résumé branch - a fixed sequence, because a resume has fields that are not optional:

```
name -> phone -> email -> target role -> education -> experience and current status
     -> projects  ->  gap pass  ->  PDF + Word  ->  email + print
```

The order and the progression are the **server's**, not the model's: any substantive answer
moves the interview on, and a refusal or a blank buys one re-ask before it moves on anyway.
The model only phrases each question, in the candidate's language. `SECTIONS` in
`routes_auto.py` is the list.

The **gap pass** at the end is what catches what is missing. It extracts a draft profile and
re-asks only what would print blank or malformed — a name without a surname, an email with
no `@`, a phone with the wrong number of digits. The PIN code is always asked unless the
candidate already said one, and that test reads the *transcript* rather than the extracted
profile: no section collects a location, and asked for one it never heard, the extractor
will infer a plausible city from the college name.

A Hindi or Punjabi interview still produces an **English** resume: `romanize_profile()`
transliterates names and translates the rest before the Latin-only facts layer runs.

Both branches show progress at each stage. A red/green status bar at the bottom reports the
current default printer. If it says **Printer: Not connected**, the PDF is saved and
downloaded rather than treated as an interview failure.

### The automated flow (`/auto`)

Click **Start interview** once; everything after that is hands-free:

```
greet -> listen -> answer -> next question -> ... -> auto-conclude
      -> extract profile -> apply facts -> PDF + Word -> send to printer
```

The server decides what is asked and when the interview moves on (`SECTIONS` in
`routes_auto.py`), telling the model exactly one thing to ask each turn. That is what stops
the interview looping on a question, and it guarantees the interview ends by itself — after
the fixed sections plus at most `MAX_GAP_QUESTIONS` clarifications, with `MAX_TURNS` (22) as
a hard stop. If a turn fails or times out, the question falls back to a fixed wording per
section per language rather than losing an interview that is already half-collected.

Conversation state lives on the **server** and is written to disk every turn, so a refresh
or crash never loses an interview. Errors surface as popups, and printer failures are
non-fatal: the resume is always saved and downloadable even if printing fails.

## Sharing it on a public URL (cloud tunnel)

The server only listens on `127.0.0.1`, so it is not reachable from another machine. A
tunnel gives it a temporary public `https://` address — useful for a demo, or for running
the kiosk on one machine and opening it on a phone.

Install the tunnel client once:

```bash
brew install cloudflared
```

Then start everything and publish it in one step:

```bash
./scripts/tepma.sh tunnel
```

It prints the address to open, for example:

```
PUBLIC URL:  https://random-words-here.trycloudflare.com/auto
```

The URL is temporary and anonymous: it lasts only while `cloudflared` runs, and stopping
the tunnel invalidates it. Starting a new tunnel always produces a **new** URL, so it
cannot be bookmarked or printed on signage.

> **Before you share the link, note what it exposes.** Anyone who has the URL can run an
> interview, generate documents, read the document library at `/docs-assistant`, and
> **send jobs to your default printer**. There is no login. Microphone access also
> requires `https`, which the tunnel provides — a plain `http://` LAN address will not
> work in most browsers. Treat the link as a live session, not a deployment: share it
> with the people in the room and stop the tunnel when you are done.

<details>
<summary>Doing it manually, or with ngrok instead</summary>

```bash
# cloudflared, in its own terminal, after the server is already running
cloudflared tunnel --url http://127.0.0.1:8000
```

```bash
# ngrok is an alternative and needs a free account + authtoken
ngrok http 8000
```

</details>

## Stop the project and test tunnel

The quickest way to stop everything — server, LLM, and tunnel — is:

```bash
./scripts/tepma.sh stop
```

Otherwise, press **Ctrl+C** once in each terminal where `uvicorn`, `ollama serve`, or
`cloudflared tunnel` is running.

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
LLM_MODEL    = "qwen3:8b"          # profile extraction, once per interview
TURN_MODEL   = "qwen3:4b-instruct" # live interview questions
STT_BACKEND  = "auto"              # auto | mlx | faster-whisper
MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
WHISPER_MODEL= "small"             # faster-whisper (CPU) fallback only
TTS_VOICE    = "af_heart"
TTS_RATE     = 24000
```

### Speech-to-text: why two backends

`faster-whisper` uses CTranslate2, which has **no Metal backend** — it runs on the CPU
even on an M-series Mac. That caps you at the `small` model, which cannot transcribe
Hindi: in testing it returned the **wrong year** (2020 for 2027) and mangled every phone
number. MLX runs `large-v3` on the GPU instead, and is both more accurate *and faster*
than CPU `small`:

| backend / model | spoken numbers correct | per utterance |
|---|---|---|
| faster-whisper `small` (CPU) | wrong year, garbled phone | 4.9 s |
| faster-whisper `large-v3` (CPU) | good | 16.6 s |
| **MLX `large-v3` (GPU)** | **5 of 6** | **3.5 s** |
| MLX `large-v3-turbo` (GPU) | 2 of 6 | 2.5 s |

`large-v3-turbo` is tempting but **not** safe here: it drops digits from phone numbers
while still producing a plausible 10-digit result, which `normalise_phone` then accepts
silently. A wrong number that looks right is worse than a slow one.

`STT_BACKEND=auto` picks MLX when `mlx-whisper` imports and falls back to faster-whisper
otherwise, so the project still runs on Intel Macs and Linux with no config change.

MLX has no voice-activity filter, so `engines.py` adds two guards: an RMS floor
(`SILENCE_RMS`) that skips the model entirely on silence, and a repetition check that
discards Whisper's noise output (`ॐ ॐ ॐ`). Both are deliberately conservative — a spoken
PIN legitimately repeats digits, and dropping a real answer is unrecoverable.

The two models are deliberately different. Extraction runs once, nobody is waiting on it,
and accuracy decides what gets printed — so it uses the larger model. The interview turn
runs up to 22 times with a candidate listening, and **must** be a non-thinking *instruct*
model: `qwen3:8b` is a hybrid reasoning model, and the `"think": False` needed to keep it
fast enough for a voice call makes it leak its topic bookkeeping into the spoken reply
(4 of 9 replies unusable in testing). Leaving thinking enabled fixes the quality but costs
~35 s per turn. An instruct model has no thinking mode to suppress, so it is both correct
and faster here.

`TEPMA_UNICODE_FONT` may optionally point to a local Unicode `.ttf` font used by generated
documents. On macOS the app automatically uses Arial Unicode when it is available.

To use a fine-tuned model, change `LLM_MODEL` — no code edits needed.

Email sending reads `SMTP_USER` / `SMTP_PASS` from the environment (never commit these).

## Accuracy: the facts layer

`facts.py` deterministically corrects every extracted profile — this is *not* the LLM's job:

- **Date awareness** — today's date is injected into prompts, so "two years back" resolves correctly
- **Indian institutions** — fuzzy-matched against `data/reference/universities.json`
  (`"Thapadi University"` → `"Thapar Institute of Engineering and Technology"`)
- **PIN codes** — matched locally against an indexed Department of Posts snapshot; API
  lookup is used only when the local snapshot is absent or does not contain the PIN
- **Phone numbers** — normalised to `+91 XXXXXXXXXX`

Every change is recorded in the profile's `_corrections` list. A PIN that is valid but
disagrees with the spoken city is never merged in silently — it is reported separately in
`_validation_warnings`, and returned by `/auto/profile` as `warnings`.

To improve accuracy, add entries to the reference files — no retraining required.

Both reference files live under the git-ignored `data/` directory, so a fresh checkout has
neither. `universities.json` is required for institution correction; without it names are
left exactly as the candidate said them (the interview still completes). Copy both files
with the deployment, or rebuild the PIN database on the target machine.

Build or refresh the local PIN database once during setup:

```bash
.venv/bin/python scripts/build_pincode_db.py
```

This creates `data/reference/india_post_pincodes.sqlite3`. The file is intentionally under
the ignored `data/` directory; copy it with the deployment or run the builder on the target
machine. Set `PINCODE_DB_PATH` only if you store it elsewhere.

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
- `profile_schema.py` — resume JSON schema, extraction prompt, and `romanize_profile()`
- `assistant_script.py` — every scripted line the kiosk speaks, in all three languages
- `mailer.py` — best-effort emailing of the finished resume
- `scripts/tepma.sh` — setup / start / stop / unload / status / tunnel
- `scripts/prepare_models.py` — downloads the models and warms the speech cache
- `resume_pdf.py` — profile JSON → resume PDF
- `make_sample_docs.py` — generates test documents
- `static/` — frontend pages including `assistant.html`; `voice.js` is the shared mic/TTS engine
- `tests/test_assistant.py` — unified workflow and renderer tests
