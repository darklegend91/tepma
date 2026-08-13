# Fine-Tuning for TePMA — A Practical Course

Read this before running anything. The code in this folder only makes sense once you know
*which problem each piece solves*.

---

## Lesson 1: The single most important idea

You asked for five things. They are **not the same kind of problem**:

| What you want | Right tool | Why |
|---|---|---|
| Interview well, ask one question at a time | **Fine-tuning** | It's a *behaviour* |
| Output perfect resume JSON | **Fine-tuning** | It's a *format/skill* |
| Handle English / Hindi / Punjabi | **Fine-tuning** (+ right base model) | It's a *behaviour* |
| Know today's date | **Prompt injection** | It's a *fact that changes* |
| Know Indian universities & pincodes | **Lookup table (RAG)** | It's a *large, exact, changing fact set* |

**Fine-tuning teaches skills, not facts.**

This is the #1 mistake beginners make. If you train a 1.7B model on a list of 1,100 Indian
universities and 19,300 pincodes, you will *not* get a model that knows them. You will get a
model that has learned *"confidently produce things that look like Indian university names"* —
which is called **hallucination**, and it's worse than not knowing, because it sounds right.

### Proof from your own app

Look at the real profile TePMA extracted from your interview earlier today:

```json
"institution": "Thapadi University"     // the candidate said "Thapar University"
"company": "Chitra University"          // their email said chitkara.edu -> "Chitkara University"
```

Whisper mis-heard, and the LLM faithfully wrote down the mis-hearing. **No amount of
fine-tuning reliably fixes this**, because the model has no way to know which of the
infinite possible garbled strings maps to which real institution.

A 40-line fuzzy-matching lookup against a real university list fixes it *deterministically,
100% of the time*:

```
"Thapadi University"  --fuzzy-->  "Thapar Institute of Engineering and Technology"
```

That is why this folder has **two** subsystems, not one:

```
        ┌──────────────────────────────────────┐
        │  Layer 1: FACTS (no training)        │
        │  • today's date -> injected in prompt│
        │  • university list -> fuzzy match    │
        │  • pincode DB -> validate + fill     │
        │  Always correct. Updateable in 1 min.│
        └──────────────────────────────────────┘
                          ▲
                          │ corrects / augments
                          │
        ┌──────────────────────────────────────┐
        │  Layer 2: BEHAVIOUR (fine-tuned)     │
        │  • interview style, one Q at a time  │
        │  • en / hi / pa conversation         │
        │  • transcript -> JSON extraction     │
        │  Small + fast. 1.7B beats 8B here.   │
        └──────────────────────────────────────┘
```

---

## Lesson 2: What fine-tuning actually does

A language model predicts the next token. Training = showing it text and nudging its weights
so its predictions match that text more closely.

**Fine-tuning** = doing that on *your* data, starting from an already-trained model.

You show it thousands of pairs:

```
INPUT  (context it sees):   system prompt + conversation so far
OUTPUT (what it should say): the ideal next interviewer question
```

After enough examples, the *shape* of your task gets burned into the weights. The payoff:
you no longer need a giant model to infer the behaviour from a long prompt — a small model
just *does* it.

**This is why fine-tuning makes things fast:** you're trading model size for task
specialization. A 1.7B model that has seen 2,000 TePMA interviews will out-interview an
8B model that is reading your instructions for the first time — while using ~1/4 the memory
and running 3-5x faster.

### Distillation: where the training data comes from

You don't have 2,000 real interviews. So we use **distillation**:

```
Qwen3-8B (teacher, on your Mac)  ──generates──>  synthetic interviews + gold JSON
                                                            │
                                                            ▼
                                        Qwen3-1.7B (student) learns to imitate
```

The student inherits the teacher's task ability, compressed. This is exactly how most
small production models are made.

---

## Lesson 3: LoRA and QLoRA (why this fits a free GPU)

Full fine-tuning updates all 1.7 billion parameters. That needs ~30 GB of GPU memory.
Colab's free T4 has 15 GB. So we use two tricks:

**LoRA (Low-Rank Adaptation)**
Freeze the entire original model. Inject tiny trainable matrices ("adapters") next to the
important layers. Train only those.

```
     original weight W (frozen, 1.7B params)
                │
      input ────┼────────────────> output
                │                    ▲
                └──> A ──> B ────────┘
                     (trainable, ~20M params = 1.2%)
```

Because `A` starts at zero, the model begins as an exact copy of the original and *learns a
small correction*. You train 1% of the parameters and get most of the benefit.

**QLoRA = Quantized LoRA**
Additionally store those frozen weights in 4 bits instead of 16. The base model shrinks from
~3.4 GB to ~1 GB in memory. Adapters stay full precision, so quality barely moves.

Together: a 1.7B model trains comfortably on a free T4 in under an hour.

### The knobs you'll actually touch

| Knob | Meaning | Our value | When to change |
|---|---|---|---|
| `r` (rank) | Adapter capacity | 16 | 32 if it underfits a complex task |
| `lora_alpha` | Adapter influence | 32 | keep at `2*r` |
| `learning_rate` | Step size | 2e-4 | lower to 1e-4 if loss is unstable |
| `num_train_epochs` | Passes over data | 2 | 3 if underfitting; never 10 |
| `max_seq_length` | Context window | 4096 | must fit longest transcript+JSON |

---

## Lesson 4: Reading the training curve (this is the skill)

Watch two numbers: **training loss** and **eval loss**.

```
 loss
   │  ╲
   │   ╲___ train                    HEALTHY:
   │    ╲___                         both fall, then flatten.
   │  ╲___╲___ eval                  Stop here.
   └──────────────> steps

 loss
   │  ╲          ___/  eval          OVERFITTING:
   │   ╲     ___/                    eval turns UP while train keeps falling.
   │    ╲___/                        The model is memorising your 150 samples.
   │     ╲____ train                 Fix: fewer epochs, more data, lower r.
   └──────────────> steps
```

**Loss is not accuracy.** A model can have great loss and still produce broken JSON.
That's why `eval_model.py` exists — it measures the thing you actually care about:
*did the right facts land in the right fields?*

---

## Lesson 5: Multilingual (English / Hindi / Punjabi)

Three separate components must each support the language. **Your model is only as
multilingual as your weakest component.**

| Component | English | Hindi | Punjabi |
|---|---|---|---|
| **Whisper STT** (`small`) | excellent | good | weak — use `large-v3` |
| **Qwen3 LLM** | excellent | good | weak, needs fine-tuning |
| **Kokoro TTS** | ✅ voices | ✅ `lang_code="h"` | ❌ **no Punjabi voice** |

**The honest constraint:** Kokoro has no Punjabi voice today. Options:
1. Speak Punjabi replies with the Hindi voice (Gurmukhi→Devanagari transliteration).
   Accent is off but intelligible. **Recommended for v1.**
2. Use a different TTS for Punjabi (Meta MMS-TTS has `pan`), at the cost of a second engine.

Also plan for **code-switching** — real Indian speech mixes languages in one sentence:

> "मेरा naam Aditya है, I'm doing B.Tech from Chitkara"

Your training data must contain this, or the model will handle it badly. `generate_data.py`
produces code-switched samples deliberately.

---

## Lesson 6: The facts layer (how we actually get accuracy)

Implemented in `facts.py`, applied to every extracted profile **after** the LLM runs.

**1. Date awareness — injection, never training**
```python
system_prompt += f"\nToday's date is {date.today():%d %B %Y}."
```
The model can now resolve "I graduated two years back" → 2024. A trained-in date would be
wrong the day after training. **Never fine-tune a fact that changes.**

**2. University correction — fuzzy match**
Every extracted institution is matched against a real list using `difflib`. Above a
similarity threshold, it's replaced with the canonical name; below, it's kept as-is
(never invent).

**3. Pincode → city/state**
A 6-digit pincode's first digits determine the region. We validate the code and auto-fill
state, so a candidate saying "one four zero four one three" gets a correct Punjab address.

**Key principle: the facts layer is *deterministic*.** It's testable, debuggable, and
updating it means editing a JSON file — not retraining a model.

---

## The workflow

Everything runs on Colab except the final latency check. Two notebooks, because the steps
need different things from the runtime:

```
 ── TePMA_data_colab.ipynb   (T4 GPU runtime, Ollama + Qwen3-8B teacher) ──
 1. GENERATE   finetune/generate_data.py --count 200 --concurrency 8
                  teacher invents Indian interviews in en/hi/pa   -> Drive/tepma/raw/
 2. BUILD      finetune/build_dataset.py
                  -> train.jsonl / eval.jsonl / eval_extraction.jsonl
 3. BASELINE   finetune/eval_model.py --model qwen3:8b | qwen3:4b | qwen3:1.7b
                  the numbers the fine-tune has to beat  <- do NOT skip this

 ── TePMA_finetune_colab.ipynb   (T4 GPU runtime, Unsloth) ──
 4. TRAIN      QLoRA, ~40 min   -> Drive/tepma/tepma.gguf  (~1.1 GB)

 ── back to TePMA_data_colab.ipynb, Step 6 ──
 5. SCORE      finetune/eval_model.py --model tepma
                  same eval set as the baselines, directly comparable

 ── on the kiosk Mac ──
 6. MEASURE    ollama create tepma -f Modelfile
               .venv/bin/python finetune/eval_model.py --model tepma
```

**Why step 6 cannot be on Colab.** A T4's tokens/sec tells you nothing about your Mac, and
reply latency on the Mac is the entire reason for shrinking the model. Colab settles
accuracy; only the kiosk settles speed.

**Why step 3 cannot be skipped.** Without baselines you cannot tell a successful fine-tune
from a wasted afternoon — and if the *untuned* 4B already scores near the 8B, the right move
is to ship that and not train anything.

### Colab realities the scripts now handle

| Problem | Handled by |
|---|---|
| Disconnect wipes the VM | `TEPMA_RAW_DIR` / `TEPMA_DATA_DIR` point at mounted Drive |
| Session cap is shorter than a 400-sample run | resume-by-skipping existing samples; re-run the cell |
| One request at a time leaves the GPU idle | `--concurrency 8` + `OLLAMA_NUM_PARALLEL=8` |
| Training dies at epoch 2 | `save_strategy="epoch"` checkpoints to Drive |

**Success criterion:** the 1.7B scores within a few points of the 8B on field accuracy,
while being several times faster *on the Mac*. If it doesn't, in this order: more data →
3 epochs → r=32. If it still doesn't, fine-tune Qwen3-4B instead of 1.7B.

---

## Things that will go wrong (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| Eval loss rises after epoch 1 | Overfitting | Fewer epochs / more data |
| Model outputs `<think>` tags | Trained with thinking on | `enable_thinking=False` in the template |
| Broken JSON | Not enough extraction examples | Keep Ollama's `format:` schema — it *forces* valid JSON |
| Hindi replies in English | Too few Hindi samples | Rebalance the language mix |
| Invented universities | You tried to train facts | Use the facts layer |
| Colab disconnects | Free-tier limit | Save checkpoints to Drive; smaller `--count` |

---

## The one-paragraph summary

Fine-tune the **behaviour** (interviewing, JSON, three languages) into a small fast model
using QLoRA on distilled data from your 8B. Handle **facts** (date, universities, pincodes)
with a deterministic lookup layer that runs after the model. Measure with a real accuracy
script, not vibes. That combination gets you something both faster *and* more accurate than
the 8B you're running today.
