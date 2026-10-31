import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

from runtime import MODEL_NAME

parser = argparse.ArgumentParser()
parser.add_argument("--workers", nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def request(url, path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def check(url):
    request(url, "/health", timeout=5)
    models = json.loads(request(url, "/v1/models", timeout=20))
    assert MODEL_NAME in [x["id"] for x in models["data"]], models
    server_info = json.loads(request(url, "/server_info", timeout=20))
    cases = [
        ("Calculate 17 * 19. Reply with only the integer.", "323"),
        ("Reply with exactly: SGLANG_OK", "SGLANG_OK"),
        (
            "What is the capital of France? Reply with only the English city name.",
            "Paris",
        ),
    ]
    results = []
    for prompt, expected in cases:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.monotonic()
        data = json.loads(request(url, "/v1/chat/completions", payload))
        content = data["choices"][0]["message"]["content"].strip()
        passed = content.rstrip(".") == expected
        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "content": content,
                "passed": passed,
                "seconds": time.monotonic() - start,
                "usage": data.get("usage"),
            }
        )
    return {
        "url": url,
        "models": models,
        "server_info": server_info,
        "cases": results,
        "passed": all(x["passed"] for x in results),
    }


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.workers)) as pool:
    for future in concurrent.futures.as_completed(
        [pool.submit(check, url) for url in args.workers]
    ):
        result = future.result()
        results.append(result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k in ["url", "cases", "passed"]}
            ),
            flush=True,
        )
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as f:
    json.dump({"started_check_unix": time.time(), "workers": results}, f, indent=2)
assert all(x["passed"] for x in results), (
    "smoke response mismatch; inspect saved results"
)
