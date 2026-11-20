"""Run eval/questions.jsonl against a running instance and print a pass/fail table.
Usage: python scripts/eval.py --url http://localhost:8000   (or your Render URL)
Uses ~18 LLM calls (the 2 guardrail cases are blocked before the model is called);
on the 50/day free tier, run it sparingly."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def ask(url: str, message: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps({"message": message}).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"text": "", "meta": {}, "model": ""}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        ev = None
        for raw in r:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                d = json.loads(line[5:])
                if ev == "meta":
                    out["meta"] = d
                elif ev == "model":
                    out["model"] = d["model"]
                elif ev == "delta":
                    out["text"] += d["text"]
    out["secs"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--sleep", type=float, default=4.0, help="pause between questions (free-tier RPM)")
    a = ap.parse_args()
    cases = [json.loads(l) for l in (Path(__file__).parent.parent / "eval" / "questions.jsonl").read_text().splitlines() if l]
    passed = 0
    for c in cases:
        r = ask(a.url, c["q"])
        text, ok = r["text"], True
        if "expect_any" in c:
            ok &= any(k.lower() in text.lower() for k in c["expect_any"])
        if c.get("expect_listings"):
            ok &= bool(r["meta"].get("listings"))
        ok &= "language model is unavailable" not in text
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {r['secs']:5.1f}s  {r['model'][:28]:<28}  {c['q'][:55]}")
        if not ok:
            print("      ->", text[:300].replace("\n", " "))
        time.sleep(a.sleep)
    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    main()
