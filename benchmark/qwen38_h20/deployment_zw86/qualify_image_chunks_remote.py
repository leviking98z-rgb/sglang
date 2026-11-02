"""Compare original and bounded Qwen GPU preprocessing on a serving H20."""
import hashlib
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

project = Path(PROJECT)
helper_namespace = {}
exec(HELPER_SOURCE, helper_namespace)
install = helper_namespace["install_cpu_transport_image_chunks"]
model = Path("/root/.cache") / ("qwen38-runtime-" + project.name) / "model/Qwen3.8-Flash-Next"
processor = AutoProcessor.from_pretrained(str(model))
torch.cuda.set_device(3)
device = "cuda:3"
images = []
hashes = []
for index in range(47):
    size = (1275, 1650) if index % 4 else (630, 525)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 50, size[0] - 20, size[1] - 20), fill=(220, index * 5 % 256, 20))
    draw.text((8, 8), f"Distinct qualification page {index}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    hashes.append(hashlib.sha256(buffer.getvalue()).hexdigest())
    images.append(image)

def process(instance, count):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.mem_get_info()[0]
    start = time.monotonic()
    with torch.cuda.use_mem_pool(torch.cuda.MemPool(), device=device):
        result = instance(images=images[:count], return_tensors="pt", device=device)
        result = {key: value.cpu() for key, value in result.items()}
    torch.cuda.synchronize()
    return result, {"count": count, "seconds": time.monotonic() - start,
                    "peak_torch_bytes": torch.cuda.max_memory_allocated(),
                    "free_before_bytes": before, "free_after_bytes": torch.cuda.mem_get_info()[0]}

reference, reference_memory = process(processor.image_processor, 12)
install(processor.image_processor, 4)
chunked, chunked_memory = process(processor.image_processor, 12)
for key in reference:
    assert torch.equal(reference[key], chunked[key]), (key, (reference[key] - chunked[key]).abs().max())
assert reference.keys() == chunked.keys()
assert chunked_memory["peak_torch_bytes"] < reference_memory["peak_torch_bytes"]
large, large_memory = process(processor.image_processor, 47)
assert large["image_grid_thw"].shape == (47, 3)
offset = 0
for index in range(47):
    single = processor.image_processor(images=[images[index]], return_tensors="pt", device=device)
    grid = single["image_grid_thw"].cpu()
    pixels = single["pixel_values"].cpu()
    assert torch.equal(grid[0], large["image_grid_thw"][index])
    assert torch.equal(pixels, large["pixel_values"][offset:offset + len(pixels)]), index
    offset += len(pixels)
    del single, pixels, grid
assert offset == len(large["pixel_values"])
result = {"passed": True, "original_12": reference_memory, "chunked_12": chunked_memory,
          "chunked_47": large_memory, "chunk_size": 4, "images": 47,
          "fixture_sha256": hashes, "grid_shape": list(large["image_grid_thw"].shape),
          "pixel_shape": list(large["pixel_values"].shape), "comparisons":
          "12 mixed-size images bitwise equal to original batch; all 47 images bitwise equal to original per-image results in order",
          "helper_sha256": hashlib.sha256(HELPER_SOURCE.encode()).hexdigest(), "created_unix": time.time()}
(project / "results/image-chunks-qualification-08.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
