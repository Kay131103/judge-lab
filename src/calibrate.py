#!/usr/bin/env python3
"""Calibrate the judge against the human gold labels and publish a report.

Reads:   data/human_labels.csv (gold), results.jsonl (judge output)
Writes:  report/index.html — a self-contained static report (inline SVG, no deps)

Metrics:
  - accuracy + confusion matrix
  - Cohen's kappa (chance-corrected agreement)
  - calibration curve: judge confidence vs empirical accuracy (binned) + ECE
  - "the lies table": every misclassified example with the judge's confidence and reason
"""
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    gold = {}
    with open(ROOT / "data" / "human_labels.csv") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = row["label"].strip().lower()
    results = []
    p = ROOT / "results.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return gold, results


def metrics(gold, results):
    judged = [r for r in results if r["judge_verdict"] in ("supported", "unsupported")]
    pairs = [(r, gold.get(r["id"], r["gold"])) for r in judged]

    n = len(pairs)
    correct = sum(1 for r, g in pairs if r["judge_verdict"] == g)
    acc = correct / n if n else 0.0

    tp = sum(1 for r, g in pairs if r["judge_verdict"] == "supported" and g == "supported")
    fp = sum(1 for r, g in pairs if r["judge_verdict"] == "supported" and g == "unsupported")
    fn = sum(1 for r, g in pairs if r["judge_verdict"] == "unsupported" and g == "supported")
    tn = sum(1 for r, g in pairs if r["judge_verdict"] == "unsupported" and g == "unsupported")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Cohen's kappa
    p_supported_true = (tp + fn) / n if n else 0
    p_supported_pred = (tp + fp) / n if n else 0
    p_e = p_supported_true * p_supported_pred + (1 - p_supported_true) * (1 - p_supported_pred)
    kappa = (acc - p_e) / (1 - p_e) if p_e < 1 else 0.0

    # calibration: bin by confidence, compare to empirical accuracy
    bins = []
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    for lo, hi in zip(edges, edges[1:]):
        grp = [(r, g) for r, g in pairs if lo <= r["confidence"] < hi]
        if grp:
            c = sum(1 for r, g in grp if r["judge_verdict"] == g)
            bins.append({
                "range": f"{lo:.1f}–{hi:.1f}", "n": len(grp),
                "avg_conf": statistics.mean(r["confidence"] for r, _ in grp),
                "accuracy": c / len(grp),
            })
    ece = sum((b["n"] / n) * abs(b["accuracy"] - b["avg_conf"]) for b in bins) if n else 0.0

    # the lies: every miss, sorted by confidence (overconfident mistakes first)
    misses = [{"id": r["id"], "claim": r["claim"], "gold": g, "judge": r["judge_verdict"],
               "confidence": r["confidence"], "reason": r["reason"]}
              for r, g in pairs if r["judge_verdict"] != g]
    misses.sort(key=lambda m: -m["confidence"])

    latencies = [r["latency_ms"] for r in judged if r.get("latency_ms")]
    lat = {
        "p50": int(statistics.median(latencies)) if latencies else 0,
        "p95": int(sorted(latencies)[int(0.95 * len(latencies))]) if latencies else 0,
        "mean": int(statistics.mean(latencies)) if latencies else 0,
    }
    errors = sum(1 for r in results if r["judge_verdict"] == "error")

    return {
        "n": n, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "kappa": kappa,
        "bins": bins, "ece": ece, "misses": misses, "latency": lat, "errors": errors,
    }


SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


def reliability_chart(bins):
    """calibration curve: judge confidence (x) vs empirical accuracy (y). perfect = diagonal."""
    w, h, pad = 420, 300, 40
    def X(v): return pad + v * (w - 2 * pad)
    def Y(v): return h - pad - v * (h - 2 * pad)
    parts = [f'<svg {SVG_NS} viewBox="0 0 {w} {h}" style="max-width:100%">']
    parts.append(f'<rect width="{w}" height="{h}" fill="#FFFDF9"/>')
    # perfect-calibration diagonal + grid
    parts.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#1a1a1a" stroke-dasharray="5 4" stroke-width="1.6"/>')
    for g in (0.25, 0.5, 0.75):
        parts.append(f'<line x1="{X(g)}" y1="{Y(0)}" x2="{X(g)}" y2="{Y(1)}" stroke="#e7e0d6" stroke-width="1"/>')
        parts.append(f'<line x1="{X(0)}" y1="{Y(g)}" x2="{X(1)}" y2="{Y(g)}" stroke="#e7e0d6" stroke-width="1"/>')
    # line through bin centers
    if bins:
        pts = " ".join(f"{X(min(b['avg_conf'], 1))},{Y(min(b['accuracy'], 1))}" for b in bins)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563EB" stroke-width="2.6"/>')
        for b in bins:
            parts.append(
                f'<circle cx="{X(min(b["avg_conf"], 1))}" cy="{Y(min(b["accuracy"], 1))}" r="5.5" '
                f'fill="#2563EB" stroke="#1a1a1a" stroke-width="1.6"/>')
    # axes labels
    parts.append(f'<text x="{w/2}" y="{h-8}" text-anchor="middle" font-family="monospace" font-size="11" fill="#8a8a8a">judge confidence</text>')
    parts.append(f'<text x="12" y="{h/2}" text-anchor="middle" font-family="monospace" font-size="11" fill="#8a8a8a" transform="rotate(-90 12 {h/2})">empirical accuracy</text>')
    parts.append('</svg>')
    return "".join(parts)


def conf_matrix(m):
    rows = [("judge: supported", m["tp"], m["fp"]), ("judge: unsupported", m["fn"], m["tn"])]
    body = ""
    for name, a, b in rows:
        c1 = "#DCFCE7" if a else "#FEE2E2"
        c2 = "#DCFCE7" if b else "#FEE2E2"
        body += f'<tr><td>{name}</td><td style="background:{c1}"><b>{a}</b> gold=supported</td><td style="background:{c2}"><b>{b}</b> gold=unsupported</td></tr>'
    return body


def lies_table(m):
    if not m["misses"]:
        return "<p class='ok'>no misses. (the judge was right about every claim this run.)</p>"
    rows = "".join(
        f"<tr><td>{x['id']}</td><td>{x['claim'][:110]}…</td>"
        f"<td><b>{x['gold']}</b></td><td class='{'over' if x['confidence'] >= 0.85 else ''}'>{x['judge']} @ {x['confidence']:.2f}</td>"
        f"<td>{x['reason'][:120]}</td></tr>"
        for x in m["misses"])
    return (f"<table><thead><tr><th>id</th><th>claim</th><th>gold</th>"
            f"<th>judge verdict</th><th>judge reason</th></tr></thead><tbody>{rows}</tbody></table>")


def build_html(m):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    grade = "well-calibrated" if m["ece"] < 0.08 else ("mildly overconfident" if m["ece"] < 0.2 else "overconfident")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>The judge is lying to you — LLM-as-judge calibration report</title>
<style>
  :root {{ --bg:#FAF8F5; --paper:#FFFDF9; --ink:#1A1A1A; --soft:#4A4A4A; --faint:#8A8A8A;
          --elec:#2563EB; --coral:#F87171; --sage:#86EFAC; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink); font:15px/1.6 "DM Sans",system-ui,sans-serif; padding:40px 18px; }}
  .wrap {{ max-width:880px; margin:0 auto; }}
  h1 {{ font-family:"Plus Jakarta Sans"; font-weight:800; font-size:clamp(28px,5vw,42px); line-height:1.15; }}
  .meta {{ font-family:"JetBrains Mono",monospace; font-size:12px; color:var(--faint); margin:10px 0 26px; }}
  h2 {{ font-family:"JetBrains Mono",monospace; font-size:13px; letter-spacing:.14em; color:var(--elec); margin:34px 0 12px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }}
  .card {{ background:var(--paper); border:2px solid var(--ink); border-radius:12px; padding:14px; box-shadow:3px 3px 0 var(--ink); }}
  .card .v {{ font-family:"Plus Jakarta Sans"; font-weight:800; font-size:26px; }}
  .card .l {{ font-family:"JetBrains Mono",monospace; font-size:10.5px; color:var(--faint); }}
  p {{ color:var(--soft); margin:8px 0; }}
  table {{ width:100%; border-collapse:collapse; background:var(--paper); border:2px solid var(--ink); font-size:13px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid #e7e0d6; text-align:left; }}
  th {{ font-family:"JetBrains Mono",monospace; font-size:11px; color:var(--faint); }}
  td.over {{ color:#b91c1c; font-weight:700; }}
  .ok {{ color:#166534; font-weight:600; }}
  .verdict {{ background:var(--paper); border:2px solid var(--ink); border-radius:12px; padding:16px 18px; box-shadow:3px 3px 0 var(--ink); margin-top:14px; }}
  .verdict b {{ font-family:"JetBrains Mono",monospace; }}
  footer {{ margin-top:40px; font-family:"JetBrains Mono",monospace; font-size:11px; color:var(--faint); text-align:center; }}
  a {{ color:var(--elec); }}
</style></head><body><div class="wrap">
  <h1>The judge is lying to you</h1>
  <div class="meta">LLM-as-judge calibration report · {m['n']} claims · generated {now} · task: grounding verification against a real README</div>

  <h2>HEADLINE NUMBERS</h2>
  <div class="cards">
    <div class="card"><div class="v">{m['accuracy']*100:.1f}%</div><div class="l">accuracy vs human labels</div></div>
    <div class="card"><div class="v">{m['f1']*100:.0f}%</div><div class="l">F1 (supported class)</div></div>
    <div class="card"><div class="v">{m['kappa']:.2f}</div><div class="l">Cohen's kappa</div></div>
    <div class="card"><div class="v">{m['ece']*100:.1f}%</div><div class="l">expected calibration error</div></div>
  </div>

  <h2>VERDICT</h2>
  <div class="verdict">
    The judge agrees with human labels <b>{m['accuracy']*100:.1f}%</b> of the time (kappa {m['kappa']:.2f}).
    Its stated confidence is <b>{grade}</b>: expected calibration error is <b>{m['ece']*100:.1f}%</b>.
    {('When it is wrong, it is wrong <b>confidently</b> — see the misses below, sorted by confidence.' if m['misses'] else 'Zero misses this run.')}
  </div>

  <h2>CALIBRATION CURVE</h2>
  <p>If the judge were perfectly calibrated, every point would sit on the dashed diagonal
  (stated confidence = observed accuracy). Points below the diagonal = overconfidence.</p>
  {reliability_chart(m['bins'])}

  <h2>CONFUSION MATRIX</h2>
  <table><tbody>{conf_matrix(m)}</tbody></table>

  <h2>THE LIES ({len(m['misses'])} misses, overconfident first)</h2>
  {lies_table(m)}

  <h2>LATENCY</h2>
  <div class="cards">
    <div class="card"><div class="v">{m['latency']['p50']} ms</div><div class="l">p50</div></div>
    <div class="card"><div class="v">{m['latency']['p95']} ms</div><div class="l">p95</div></div>
    <div class="card"><div class="v">{m['errors']}</div><div class="l">transport/API errors</div></div>
  </div>

  <h2>METHODOLOGY</h2>
  <p><b>Task:</b> grounding verification. A context document (the README of a real open-source
  project) and 48 claims about it — some faithful paraphrases, some with altered numbers,
  invented features, or subtle negations. Gold labels are human-authored.</p>
  <p><b>Judge:</b> a single LLM call per claim with a strict system prompt (literal reading,
  no outside knowledge), temperature 0, JSON output containing verdict + confidence.
  The confidence — not just the verdict — is what this report calibrates.</p>
  <p><b>Reproduce:</b> clone the repo, set <code>GEMINI_API_KEY</code>, run
  <code>python3 src/run.py && python3 src/calibrate.py</code>. The CI re-runs the whole
  evaluation on every push and republishes this report.</p>

  <footer>built with a stdlib-only Python harness (no eval framework) · part of the krithiuk.space lab</footer>
</div></body></html>"""


def main():
    gold, results = load()
    if not results:
        print("no results.jsonl found — run src/run.py first")
        return
    m = metrics(gold, results)
    out = ROOT / "report" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(m))
    print(f"report → {out}")
    print(f"accuracy {m['accuracy']*100:.1f}% · kappa {m['kappa']:.2f} · ECE {m['ece']*100:.1f}% · misses {len(m['misses'])}")


if __name__ == "__main__":
    main()
