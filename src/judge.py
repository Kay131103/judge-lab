#!/usr/bin/env python3
"""LLM judge for grounding verification. Stdlib-only (urllib + json).

Given a context document and a claim, the judge answers:
  - verdict: "supported" | "unsupported"
  - confidence: 0.0-1.0 (how sure the judge is in its verdict)
  - reason: one sentence

Everything is configurable via env vars:
  GEMINI_API_KEY   required
  JUDGE_MODEL      default gemini-3.5-flash-lite (works on the free tier)
  JUDGE_TIMEOUT    default 30s
"""
import json
import os
import time
import urllib.request
import urllib.error

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.5-flash-lite")
TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "30"))

SYSTEM = """You are a strict fact-checking judge. You will be given a CONTEXT document \
and a CLAIM. Decide whether the CONTEXT explicitly supports the CLAIM.

Rules:
- "supported": the claim is directly stated in the context, or is a faithful, \
unambiguous paraphrase of something stated there.
- "unsupported": the claim adds information, changes details (numbers, names, \
features), contradicts the context, or goes beyond what is written.
- If the context does not mention the topic at all, answer "unsupported".
- Be literal. Do not use outside knowledge, even if you believe the claim is true \
in the real world. The context is the only source of truth.
- Confidence: 1.0 = absolutely certain, 0.5 = genuinely torn, 0.0 = pure guess.

Answer ONLY with minified JSON: {"verdict": "...", "confidence": 0.0, "reason": "..."}"""


def judge(context: str, claim: str) -> dict:
    """Returns {verdict, confidence, reason, latency_ms, raw}. Raises on hard failure."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200,
                             "responseMimeType": "application/json"},
    }
    url = f"{API_BASE}/{MODEL}:generateContent?key={key}"

    t0 = time.time()
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    last_err = None
    for attempt in range(3):  # retry on 503 / rate limits
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code in (429, 500, 503) and attempt < 2:
                time.sleep(4 * (attempt + 1))
                continue
            raise RuntimeError(f"judge call failed: {last_err}")
        except Exception as e:  # network
            last_err = str(e)
            if attempt < 2:
                time.sleep(3)
                continue
            raise RuntimeError(f"judge call failed: {last_err}")

    latency_ms = int((time.time() - t0) * 1000)
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    verdict_obj = json.loads(text)
    verdict = str(verdict_obj.get("verdict", "")).lower()
    if verdict not in ("supported", "unsupported"):
        raise RuntimeError(f"judge returned invalid verdict: {text[:200]}")
    confidence = max(0.0, min(1.0, float(verdict_obj.get("confidence", 0.5))))
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(verdict_obj.get("reason", ""))[:300],
        "latency_ms": latency_ms,
        "model": MODEL,
    }
