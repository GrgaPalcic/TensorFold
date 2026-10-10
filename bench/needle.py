"""Needle-recall smoke: a long filler prompt with one buried access code, answered at temperature 0.

Usage (against the running server):  python needle.py [filler_blocks] [needle_at_block]
Default: 400 filler blocks (~90K tokens), needle after block 300. Prints prompt tokens, prefill
throughput, and whether the exact code was recalled. Run on a rank-0 host (worker-2).
"""
import json, sys, time, urllib.request

filler = "The harbor lights flickered over the water while merchants argued about grain prices. " * 15
blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 400
at = int(sys.argv[2]) if len(sys.argv) > 2 else 300
needle = "\n\nIMPORTANT: the access code is PINEAPPLE-47. Remember it.\n\n"
prompt = filler * at + needle + filler * (blocks - at)
prompt += "\n\nWhat is the access code? Answer with just the code."

body = {"model": "redsnow-serve", "temperature": 0, "max_tokens": 16,
        "messages": [{"role": "user", "content": prompt}]}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=900))
ptk = d["usage"]["prompt_tokens"]
text = d["choices"][0]["message"]["content"] or ""
ok = "PINEAPPLE-47" in text
print(f"prompt_tokens={ptk} prefill_s={d['tensorfold']['prefill_s']:.2f} "
      f"({ptk / d['tensorfold']['prefill_s']:.0f} tok/s) needle_exact={ok} answer={text.strip()!r}")
