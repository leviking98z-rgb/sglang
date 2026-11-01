import argparse
import concurrent.futures
import json
import math
import time
import urllib.request
from pathlib import Path

from runtime import MODEL, MODEL_NAME
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--workers", nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--filler-tokens", type=int, default=8000)
args = parser.parse_args()
assert args.filler_tokens >= 8000
tokenizer = AutoTokenizer.from_pretrained(MODEL)
filler_text = "A notebook records ordinary observations about weather, trees, books, and streets. "
unit_tokens = len(tokenizer.encode(filler_text, add_special_tokens=False))
repetitions = math.ceil(args.filler_tokens / unit_tokens) + 1
filler = tokenizer.encode(filler_text * repetitions, add_special_tokens=False)
while len(filler) < args.filler_tokens:
    # Token boundaries can merge between repeated units.
    repetitions *= 2
    filler = tokenizer.encode(filler_text * repetitions, add_special_tokens=False)
filler = filler[: args.filler_tokens]
assert len(filler) == args.filler_tokens
expected = "SKY7319"
cases = []
for fraction in [0.1, 0.5, 0.9]:
    index = int(len(filler) * fraction)
    prompt = (
        tokenizer.decode(filler[:index])
        + "\nMEMORY: The access code for package Orion is SKY7319.\n"
        + tokenizer.decode(filler[index:])
        + "\nWhat is the access code for package Orion? Reply with only the code, no formatting."
    )
    cases.append(
        {
            "fraction": fraction,
            "prompt": prompt,
            "text_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        }
    )


def check(url):
    results = []
    for case in cases:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
        content = data["choices"][0]["message"]["content"].strip()
        usage = data["usage"]
        passed = (
            content.strip("`\"' .\n") == expected
            and args.filler_tokens <= usage["prompt_tokens"] <= args.filler_tokens + 512
        )
        results.append(
            {
                "needle_fraction": case["fraction"],
                "expected": expected,
                "content": content,
                "passed": passed,
                "usage": usage,
                "seconds": time.monotonic() - started,
            }
        )
    result = {"url": url, "cases": results, "passed": all(x["passed"] for x in results)}
    print(json.dumps(result), flush=True)
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.workers)) as pool:
    results = list(pool.map(check, args.workers))
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as f:
    json.dump(
        {
            "workers": results,
            "filler_tokens": args.filler_tokens,
            "created_unix": time.time(),
        },
        f,
        indent=2,
    )
assert all(x["passed"] for x in results), "long context smoke response mismatch"
