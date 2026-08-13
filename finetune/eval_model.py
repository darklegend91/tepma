"""Measure how well a model does TePMA's real job, on held-out interviews.

Loss tells you training is progressing. THIS tells you whether users get a correct resume.
Two things are scored:
  1. extraction accuracy - did the right facts land in the right fields?
  2. speed              - median seconds per interviewer reply

Run:
  .venv/bin/python finetune/eval_model.py --model qwen3:8b     # teacher baseline
  .venv/bin/python finetune/eval_model.py --model tepma        # your fine-tuned student
"""
import argparse
import asyncio
import difflib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engines
from facts import apply_facts, date_context
from profile_schema import EXTRACTOR_PROMPT, PROFILE_SCHEMA
from routes_interview import INTERVIEWER_PROMPT

DATA_DIR = Path(os.getenv("TEPMA_DATA_DIR") or Path(__file__).parent / "data")
EVAL_FILE = DATA_DIR / "eval_extraction.jsonl"


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()


def score_profile(got: dict, gold: dict) -> dict:
    """Field-level accuracy. Strings are fuzzy-matched (wording may differ legitimately);
    lists are scored by overlap."""
    scores = {}

    for field in ("name", "target_role", "email", "phone"):
        g, w = gold.get(field, ""), got.get(field, "")
        scores[field] = 1.0 if (not g and not w) else similar(g, w) >= 0.85

    # education: did we get the right institution and degree?
    gold_edu = gold.get("education", [])
    got_edu = got.get("education", [])
    if gold_edu:
        hits = 0
        for ge in gold_edu:
            hits += any(similar(ge.get("institution", ""), we.get("institution", "")) >= 0.7
                        for we in got_edu)
        scores["education"] = hits / len(gold_edu)
    else:
        scores["education"] = 1.0

    # experience: right companies, and did we keep the bullets?
    gold_exp = gold.get("experience", [])
    got_exp = got.get("experience", [])
    if gold_exp:
        hits = sum(any(similar(ge.get("company", ""), we.get("company", "")) >= 0.7
                       for we in got_exp) for ge in gold_exp)
        scores["experience"] = hits / len(gold_exp)
    else:
        scores["experience"] = 1.0 if not got_exp else 0.0  # inventing jobs is a failure

    # skills: overlap
    gs = {s.lower().strip() for s in gold.get("skills", [])}
    ws = {s.lower().strip() for s in got.get("skills", [])}
    scores["skills"] = len(gs & ws) / len(gs) if gs else 1.0

    # hallucination check: did it invent experience entries that were never mentioned?
    extra = max(0, len(got_exp) - len(gold_exp))
    scores["no_hallucination"] = 1.0 if extra == 0 else max(0.0, 1 - extra / max(1, len(gold_exp)))

    scores["_overall"] = statistics.mean(float(v) for k, v in scores.items() if not k.startswith("_"))
    return scores


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. tepma or qwen3:8b")
    parser.add_argument("--limit", type=int, default=0, help="only evaluate the first N cases")
    args = parser.parse_args()

    if not EVAL_FILE.exists():
        sys.exit(f"missing {EVAL_FILE} - run finetune/build_dataset.py first")

    engines.LLM_MODEL = args.model  # point the shared engine at the model under test
    cases = [json.loads(l) for l in EVAL_FILE.open() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]

    print(f"model: {args.model}   held-out interviews: {len(cases)}\n")

    per_field, overall, extract_times = {}, [], []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        try:
            got = await engines.llm_extract(
                [{"role": "user", "content": f"Interview transcript:\n\n{case['transcript']}"}],
                EXTRACTOR_PROMPT + date_context(),
                PROFILE_SCHEMA,
            )
        except Exception as e:
            print(f"  [{i}] extraction FAILED: {e}")
            overall.append(0.0)
            continue
        extract_times.append(time.perf_counter() - t0)
        got = apply_facts(got)
        s = score_profile(got, case["gold_profile"])
        overall.append(s["_overall"])
        for k, v in s.items():
            if not k.startswith("_"):
                per_field.setdefault(k, []).append(float(v))
        print(f"  [{i}/{len(cases)}] {s['_overall']:.0%}  {got.get('name', '?')}")

    # interviewer latency: how long to produce one spoken reply
    reply_times = []
    convo = [{"role": "user", "content": "(The candidate has joined the voice call. Greet them and begin the interview.)"}]
    for _ in range(3):
        t0 = time.perf_counter()
        reply = await engines.llm_chat(convo, INTERVIEWER_PROMPT + date_context())
        reply_times.append(time.perf_counter() - t0)
        convo += [{"role": "assistant", "content": reply},
                  {"role": "user", "content": "I am a B.Tech student from Chitkara University."}]

    print("\n" + "=" * 58)
    print(f"RESULTS for {args.model}")
    print("=" * 58)
    for field, vals in sorted(per_field.items()):
        print(f"  {field:18} {statistics.mean(vals):6.1%}")
    print(f"  {'OVERALL':18} {statistics.mean(overall):6.1%}")
    print("-" * 58)
    print(f"  extraction time    {statistics.median(extract_times):5.1f}s median")
    print(f"  interview reply    {statistics.median(reply_times):5.1f}s median  <- users feel this")
    print("=" * 58)
    if os.getenv("COLAB_RELEASE_TAG"):
        print("NOTE: accuracy above is valid, but these timings are Colab GPU timings.\n"
              "      Re-run this on the kiosk Mac to get the latency users actually feel.")


if __name__ == "__main__":
    asyncio.run(main())
