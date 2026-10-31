"""Derive a request-timed benchmark from a checksum-verified main source file."""

import hashlib
import json

from runtime import PROJECT

UPSTREAM_COMMIT = "a4a6eddd25343905245db7a8b54bcf24d9483c8b"
SOURCE_SHA256 = "852c7d3076f58fc1e96f65021ce26b3a1dfd2b5839f1487f20fa18a8019b9935"


def prepare():
    source = PROJECT / "workdir/sglang/python/sglang/benchmark/serving.py"
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == SOURCE_SHA256
    text = original.decode()
    timer = "    benchmark_duration = time.perf_counter() - benchmark_start_time\n"
    anchor = "    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)\n"
    assert text.count(timer) == text.count(anchor) == 1
    patched = text.replace(timer, "").replace(anchor, anchor + timer).encode()
    target = PROJECT / "scripts/serving_request_timed.py"
    if target.exists():
        assert target.read_bytes() == patched
    else:
        with target.open("xb") as stream:
            stream.write(patched)
    provenance = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "generated_sha256": hashlib.sha256(patched).hexdigest(),
        "change": "Stop the request timer immediately after asyncio.gather completes, before the post-run server_info query.",
    }
    sidecar = target.with_suffix(".provenance.json")
    if sidecar.exists():
        assert json.loads(sidecar.read_text()) == provenance
    else:
        with sidecar.open("x") as stream:
            json.dump(provenance, stream, indent=2)
    return target, provenance


if __name__ == "__main__":
    target, provenance = prepare()
    print(json.dumps({"target": str(target), **provenance}))
