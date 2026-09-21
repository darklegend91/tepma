# TePMA User Guide

TePMA is a voice kiosk that interviews a job seeker by talking to them — in English, Hindi
or Punjabi — and prints a finished resume at the end. Nothing is typed, and nothing is sent
to the internet: the speech recognition, the voice, and the AI all run on this computer.

This guide has two parts: **setting it up** (once, by whoever owns the machine) and
**using it** (every day, by the people who walk up to it).

---

## Part 1 — Setting it up

### What you need

| | |
|---|---|
| Computer | A Mac with Apple Silicon (M1 or newer) |
| Memory | 16 GB RAM minimum |
| Disk | About 15 GB free |
| Microphone | Built-in is fine; a USB mic works better in a noisy room |
| Browser | Google Chrome (recommended) or Safari |
| Printer | Optional. Set it as the Mac's **default** printer |
| Internet | Only for the one-time setup, to download the models |

### Step 1 — Install the tools (once)

Open **Terminal** and install [Homebrew](https://brew.sh) if you do not already have it, then:

```bash
brew install python@3.11 ollama
```

Optional, but recommended — a database that keeps a searchable record of every interview:

```bash
brew tap mongodb/brew && brew install mongodb-community
```

TePMA works without MongoDB; it just keeps its records as files instead.

### Step 2 — Set up TePMA (once)

In Terminal, go to the TePMA folder and run:

```bash
./scripts/tepma.sh setup
```

This downloads about **10 GB** of AI models and prepares everything. Expect 20–40 minutes
on a normal connection. It is safe to run again — a second run finishes in seconds and just
confirms the machine is ready.

### Step 3 — Start it

```bash
./scripts/tepma.sh start
```

Wait until you see `voice server ready`. Then open Chrome and go to:

```
http://localhost:8000
```

The first time, Chrome asks for **microphone** permission — click **Allow**.

Leave that browser window open. That screen is the kiosk.

### Step 4 — Stop it at the end of the day

```bash
./scripts/tepma.sh stop
```

This shuts TePMA down and frees the ~7 GB of memory the AI models use.

---

## Part 2 — Using the kiosk

Give these instructions to the people using it.

1. **Tap anywhere on the screen.** (The browser needs one tap before it is allowed to use
   the microphone and speakers. After this, you do not touch anything.)
2. **Listen to the greeting.** It is spoken in English, then Hindi, then Punjabi.
3. **Say the language you want** — "English", "Hindi" or "Punjabi".
4. **Answer each question out loud**, one at a time. The assistant asks for:
   - your full name
   - your mobile number
   - your email address
   - the job you are applying for
   - your education
   - your work experience, and whether you are working, studying or looking for work
   - a project or work you can show
5. **Pause for about three seconds** when you finish an answer. That is how it knows you are done.
6. It may ask **one or two short follow-up questions** at the end — usually your city and
   6-digit PIN code.
7. **Wait while your resume is prepared** — up to a minute. It is then printed, and it tells
   you where to collect it.

### Tips for a good resume

- **Spell out your name and email address** letter by letter. Speech recognition gets names
  and emails wrong more than anything else.
- **Say numbers clearly**, digit by digit for your phone number and PIN code.
- **Mention numbers in your experience** if you have them — how many people, how much
  improvement, how many years.
- If you **do not have something** (no experience yet, no projects), just say "no" or
  "nahi". It moves on.
- Speak in a **quiet place**, a normal distance from the microphone.
- To **cancel**, press the red **Stop** button. It is the only button on the screen.

The finished resume is always written **in English**, whichever language you speak — that
is what employers in India expect.

---

## For the operator

### Everyday commands

| Command | What it does |
|---|---|
| `./scripts/tepma.sh start` | Start everything |
| `./scripts/tepma.sh stop` | Stop everything and free memory |
| `./scripts/tepma.sh status` | What is running, and whether the models are installed |
| `./scripts/tepma.sh unload` | Free the AI memory but keep the kiosk page up |
| `./scripts/tepma.sh restart` | Stop, then start |

### Where the resumes are

- **Files:** `data/sessions/<date-time-id>/` — the transcript, `profile.json`,
  `resume.pdf` and `resume.docx` for each interview.
- **Database:** the `tepma` MongoDB database, `sessions` collection (if MongoDB is installed).

These contain people's names, phone numbers, email addresses and home cities. Treat them as
personal data: do not share the folder, and delete old interviews you no longer need.

### Printing

TePMA prints to the Mac's **default printer**. Set it in **System Settings → Printers &
Scanners**. The bar at the bottom of the kiosk shows *Printer: <name>* when it is ready. If
it says **Not connected**, interviews still work — the resume is saved and shown on screen,
and the assistant tells the person it was not printed.

### Emailing resumes to candidates (optional)

Edit the `.env` file in the TePMA folder and fill in:

```
SMTP_USER = "your-address@gmail.com"
SMTP_PASS = "your Gmail App Password"
```

For Gmail this must be an **App Password** (Google Account → Security → App passwords), not
your normal password. Restart TePMA afterwards. Without this, TePMA simply does not email —
and does not claim to.

### Using a different port

```bash
PORT=8080 ./scripts/tepma.sh start
```

Then open `http://localhost:8080`.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| Nothing happens after tapping | Chrome blocked the microphone. Click the camera/mic icon in the address bar → Allow, then reload. |
| No sound | Check the Mac's volume and output device. |
| The first question takes a long time | Normal after starting — the AI is loading into memory. Later sessions are faster. |
| It keeps saying "I did not hear anything" | Move closer to the microphone, or check the right input device is selected in System Settings → Sound. |
| It heard the wrong words | Speak a little slower. For names and emails, spell them letter by letter. |
| `address already in use` | TePMA is already running. Run `./scripts/tepma.sh status`. |
| `voice server did not come up` | Look in `logs/server.log` for the error. |
| "Printer: Not connected" | Set a default printer in System Settings → Printers & Scanners. |
| Punjabi was not understood when choosing a language | Say "Punjabi" in English, or say "ਪੰਜਾਬੀ" clearly. After three tries it continues in English. |

---

## Keeping it safe

- **Keep it local.** By default TePMA only accepts connections from this computer. Do not
  change that unless you know why.
- **The public tunnel (`./scripts/tepma.sh tunnel`) is for demos only.** Anyone with that
  link can run interviews and send jobs to your printer, and there is no login. Stop it when
  the demo ends.
- **Do not browse other websites on the kiosk machine** while TePMA is running.
- **Never share the `.env` file** — it can contain your email password.
