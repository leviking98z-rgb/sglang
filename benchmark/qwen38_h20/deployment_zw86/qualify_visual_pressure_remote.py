"""Synthetic multi-page visual requests matching observed benchmark dimensions."""

import base64
import concurrent.futures
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

project = Path(PROJECT)
urls = WORKERS
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
fixtures = []
for page in range(47):
    image = Image.new("RGB", (1275, 1650), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 300, 1000, 1300), fill=(220, 20, 20))
    draw.text((30, 30), f"Synthetic page {page + 1}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    fixtures.append(dict(
        page=page, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
        uri="data:image/png;base64," + base64.b64encode(raw).decode(),
    ))


def check(item):
    url, count = item
    content = []
    if count == 24:
        content.append({"type": "text", "text": "Read this filler silently.\n" + " x" * 49000})
    content.extend({"type": "image_url", "image_url": {"url": fixture["uri"]}}
                   for fixture in fixtures[:count])
    content.append({"type": "text", "text": "What color is the large rectangle on every page? Reply with only the lower-case color name."})
    body = dict(
        model="Qwen/Qwen3.8-Flash-Next", messages=[{"role":"user","content":content}],
        temperature=0, max_tokens=65536, chat_template_kwargs={"enable_thinking":False},
    )
    encoded = json.dumps(body).encode()
    request = urllib.request.Request(url + "/v1/chat/completions", data=encoded,
                                     headers={"Content-Type":"application/json"})
    begin = time.monotonic()
    try:
        with opener.open(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        return dict(url=url, images=count, passed=False, status=exc.code,
                    error=exc.read().decode()[:2500], seconds=time.monotonic()-begin)
    answer = result["choices"][0]["message"].get("content", "").strip()
    assert answer.lower().strip('`"\' .\n') == "red", result
    return dict(
        url=url, images=count, passed=True, status=200, answer=answer,
        usage=result["usage"], seconds=time.monotonic()-begin, request_bytes=len(encoded),
        requested_max_tokens=65536, actual_65k_generation_tested=False,
        total_source_pixels=count*1275*1650, approximate_text_filler_tokens=49000 if count==24 else 0,
    )


started = time.time()
# Two simultaneous requests per worker: 24 pages with a 49K filler and 47 pages.
# This is16 client requests across eight independent TP4/EP4 workers.
with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)*2) as pool:
    results = list(pool.map(check, [(url, count) for url in urls for count in [24,47]]))
receipt = dict(
    passed=all(j["passed"] for j in results), started_unix=started, completed_unix=time.time(),
    source_dimensions=[1275,1650], source_mode="RGB", source_format="PNG",
    source_fixtures=[{k:v for k,v in f.items() if k!="uri"} for f in fixtures],
    client_concurrency=len(urls)*2, results=results,
    scope="Synthetic valid images and text only; does not replay historical failed questions or guarantee arbitrary future image loads",
)
(project/"results/visual-pressure-qualification.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"passed":receipt["passed"],"requests":len(results),"results":results}),flush=True)
assert receipt["passed"], "Synthetic visual pressure request failed; inspect recorded original result"
