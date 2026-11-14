#!/usr/bin/env python3
"""Run the judge over the eval set and save verdicts.

Usage: python3 src/run.py [--limit N] [--out results.jsonl]
Env: GEMINI_API_KEY (required), JUDGE_MODEL (optional)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from judge import judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run first N examples (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "results.jsonl"))
    args = ap.parse_args()

    context = (ROOT / "data" / "context.md").read_text()
    examples = []
    with open(ROOT / "data" / "evalset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.limit:
        examples = examples[: args.limit]

    print(f"running judge over {len(examples)} examples (model: {os.environ.get('JUDGE_MODEL', 'default')})")
    done = 0
    with open(args.out, "w") as out:
        for ex in examples:
            for attempt in range(2):  # one retry on transient parse errors
                try:
                    r = judge(context, ex["claim"])
                    rec = {
                        "id": ex["id"], "claim": ex["claim"], "gold": ex["label"],
                        "judge_verdict": r["verdict"], "confidence": r["confidence"],
                        "reason": r["reason"], "latency_ms": r["latency_ms"],
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    done += 1
                    mark = "✓" if r["verdict"] == ex["label"] else "✗"
                    print(f"[{done}/{len(examples)}] {mark} {ex['id']} "
                          f"gold={ex['label']} judge={r['verdict']} conf={r['confidence']:.2f}")
                    break
                except Exception as e:
                    if attempt == 1:
                        rec = {"id": ex["id"], "claim": ex["claim"], "gold": ex["label"],
                               "judge_verdict": "error", "confidence": 0.0,
                               "reason": str(e)[:200], "latency_ms": 0}
                        out.write(json.dumps(rec) + "\n")
                        print(f"[{done + 1}/{len(examples)}] ! {ex['id']} ERROR {e}")
                    else:
                        time.sleep(3)
            time.sleep(1.2)  # pace for the free tier

    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
