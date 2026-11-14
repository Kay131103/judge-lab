# 🧪 Judge Lab

**"Your eval set is lying to you" — the essay, made real with numbers.**

An LLM-as-judge calibration harness. A judge model verifies 48 claims against a real
README (grounding verification); a human-labeled gold set catches it; the report shows
exactly **when the judge is lying** — and how confidently.

Zero dependencies (pure Python stdlib). The CI **re-runs the whole evaluation on every
push** and republishes this report: https://evals.krithiuk.space

## What it measures
| Metric | Why it matters |
|---|---|
| Accuracy vs human labels | can you trust the judge at all? |
| Cohen's kappa | agreement beyond chance |
| **Expected Calibration Error (ECE)** | does stated confidence match reality? |
| Calibration curve | overconfidence made visible |
| The lies table | every miss, sorted by confidence — the overconfident wrong answers first |
| Latency p50/p95 | what the judge costs at runtime |

## Design decisions
- **Task:** grounding verification — "does the context support this claim?" It's the
  exact failure mode of RAG systems, and it's checkable with binary gold labels.
- **Literal judging:** the system prompt forbids outside knowledge. The context is the
  only truth — this is what makes wrong answers informative.
- **Confidence is the product:** the verdict alone is cheap; the *calibration* of the
  judge's stated confidence is what this lab publishes.
- ** stdlib-only:** urllib + json. No eval framework, no vendor lock-in.

## Files
```
data/context.md        the grounding document (a real project README)
data/evalset.jsonl     48 claims with gold labels
data/human_labels.csv  the same labels as editable CSV
src/judge.py           Gemini call: verdict + confidence + reason (retries, pacing)
src/run.py             runs the eval set → results.jsonl
src/calibrate.py       metrics + report/index.html
```

## Run it yourself
```bash
export GEMINI_API_KEY=...   # free tier is enough
python3 src/run.py
python3 src/calibrate.py    # → report/index.html
```

Live report: **https://evals.krithiuk.space** · part of the [krithiuk.space](https://portfolio.krithiuk.space) lab.
