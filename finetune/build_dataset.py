"""Convert raw samples into fine-tuning datasets (chat-messages JSONL).

Produces two tasks in one dataset:
  - interviewer: conversation-so-far -> next interviewer line
  - extraction:  transcript -> gold profile JSON

Run:  .venv/bin/python finetune/build_dataset.py
Outputs: finetune/data/train.jsonl, finetune/data/eval.jsonl,
         finetune/data/eval_extraction.jsonl (held-out transcripts for accuracy testing)
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from profile_schema import EXTRACTOR_PROMPT
from routes_interview import INTERVIEWER_PROMPT

# Overridable so Colab can read/write mounted Drive instead of the ephemeral VM disk.
RAW_DIR = Path(os.getenv("TEPMA_RAW_DIR") or Path(__file__).parent / "raw")
OUT_DIR = Path(os.getenv("TEPMA_DATA_DIR") or Path(__file__).parent / "data")

EVAL_FRACTION = 0.1
MAX_INTERVIEWER_TURNS_PER_SAMPLE = 5


def transcript_text(turns):
    return "\n".join(
        f"{'Interviewer' if t['speaker'] == 'interviewer' else 'Candidate'}: {t['text']}"
        for t in turns
    )


def interviewer_examples(turns):
    """Each interviewer turn (after the first) becomes: history -> next question."""
    examples = []
    history = [{"role": "user", "content":
                "(The candidate has joined the voice call. Greet them and begin the interview.)"}]
    for t in turns:
        role = "assistant" if t["speaker"] == "interviewer" else "user"
        if role == "assistant" and len(history) > 1:
            examples.append({"messages": [
                {"role": "system", "content": INTERVIEWER_PROMPT},
                *history,
                {"role": "assistant", "content": t["text"]},
            ]})
        history.append({"role": role, "content": t["text"]})
    random.shuffle(examples)
    return examples[:MAX_INTERVIEWER_TURNS_PER_SAMPLE]


def extraction_example(sample):
    return {"messages": [
        {"role": "system", "content": EXTRACTOR_PROMPT},
        {"role": "user", "content": f"Interview transcript:\n\n{transcript_text(sample['transcript'])}"},
        {"role": "assistant", "content": json.dumps(sample["profile"], ensure_ascii=False)},
    ]}


def main():
    random.seed(7)
    samples = []
    for f in sorted(RAW_DIR.glob("sample_*.json")):
        try:
            samples.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print("skipping corrupt", f.name)
    random.shuffle(samples)
    n_eval = max(2, int(len(samples) * EVAL_FRACTION))
    eval_samples, train_samples = samples[:n_eval], samples[n_eval:]

    def rows(subset):
        out = []
        for s in subset:
            out.append(extraction_example(s))
            out.extend(interviewer_examples(s["transcript"]))
        random.shuffle(out)
        return out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", train_samples), ("eval", eval_samples)]:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in rows(subset):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {sum(1 for _ in path.open())} examples from {len(subset)} interviews")

    # held-out raw transcripts + gold profiles for end-to-end accuracy testing
    with (OUT_DIR / "eval_extraction.jsonl").open("w") as fh:
        for s in eval_samples:
            fh.write(json.dumps({"transcript": transcript_text(s["transcript"]),
                                 "gold_profile": s["profile"]}, ensure_ascii=False) + "\n")
    print(f"{OUT_DIR / 'eval_extraction.jsonl'}: {len(eval_samples)} held-out interviews")


if __name__ == "__main__":
    main()
