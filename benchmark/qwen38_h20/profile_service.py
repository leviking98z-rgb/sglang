"""Capture a short, separate prefill/decode trace through SGLang's native API."""

import argparse
import json
import time
import urllib.request

from runtime import PROJECT

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("--url", required=True)
parser.add_argument("--concurrency", type=int, default=64)
parser.add_argument("--input-len", type=int, default=1024)
parser.add_argument("--output-len", type=int, default=256)
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
directory = PROJECT / "results" / args.attempt
directory.mkdir(exist_ok=False)


def post(path, data):
    request = urllib.request.Request(
        args.url + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = response.read()
        assert response.status == 200
        return body


sampling = {"temperature": 0, "max_new_tokens": args.output_len, "ignore_eos": True}
prompts = [
    [1000 + (i * 7919 + j * 2003 + 20261031) % 200000 for i in range(args.input_len)]
    for j in range(args.concurrency)
]
post("/generate", {"input_ids": prompts[0], "sampling_params": sampling})
post("/flush_cache?timeout=60", {})
post(
    "/start_profile",
    {
        "output_dir": str(directory),
        "profile_id": args.attempt,
        "num_steps": 6,
        "activities": ["CPU", "GPU"],
        "profile_by_stage": True,
        "with_stack": False,
        "record_shapes": False,
        "merge_profiles": False,
    },
)
started = time.time()
try:
    responses = json.loads(
        post("/generate", {"input_ids": prompts, "sampling_params": sampling})
    )
    assert len(responses) == args.concurrency
    metadata = [response["meta_info"] for response in responses]
    assert all(item["prompt_tokens"] == args.input_len for item in metadata)
    assert all(item["completion_tokens"] == args.output_len for item in metadata)
    assert all(item["num_retractions"] == 0 for item in metadata)
    result = {
        "started_unix": started,
        "ended_unix": time.time(),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "requests": metadata,
        "note": "Profiling run; excluded from throughput comparisons.",
    }
    (directory / "profile_workload.json").write_text(json.dumps(result, indent=2))
except BaseException:
    try:
        post("/stop_profile", {})
    except OSError:
        pass
    raise
# num_steps automatically stops and exports the traces during generation.
# A second stop would return an error because profiling already finished.
traces = list(directory.glob("*.trace.json.gz"))
assert traces, "no profiler traces were exported"
print(
    json.dumps(
        {
            "directory": str(directory),
            "completed": args.concurrency,
            "traces": len(traces),
        }
    )
)
