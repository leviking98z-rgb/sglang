"""Full original/candidate comparison on an otherwise idle allocated H20."""
import hashlib
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

p = Path(PROJECT)
helper_namespace = {}
exec(HELPER_SOURCE, helper_namespace)
model = Path('/root/.cache') / ('qwen38-runtime-' + p.name) / 'model/Qwen3.8-Flash-Next'
torch.cuda.set_device(0)
assert torch.cuda.mem_get_info()[0] > 80 * 1024**3, 'Qualification requires the stopped pilot GPU'
processor = AutoProcessor.from_pretrained(str(model))
original = processor.image_processor._preprocess
helper_namespace['install_cpu_transport_image_chunks'](processor.image_processor, 4)
bounded = processor.image_processor._preprocess
fixtures = []
hashes = []
for page in range(47):
    image = Image.new('RGB', (1275, 1650), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 300, 1000, 1300), fill=(220, 20, 20))
    draw.text((30, 30), f'Synthetic page {page + 1}', fill='black')
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    hashes.append(hashlib.sha256(buffer.getvalue()).hexdigest())
    fixtures.append(image)
mixed = [image.resize((630, 525)) if index % 4 == 0 else image for index, image in enumerate(fixtures)]

def run(images, method):
    processor.image_processor._preprocess = method
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.mem_get_info()[0]; start = time.monotonic()
    with torch.cuda.use_mem_pool(torch.cuda.MemPool(), device='cuda:0'):
        result = processor.image_processor(images=images, return_tensors='pt', device='cuda:0')
        result = {key: value.cpu() for key, value in result.items()}
    torch.cuda.synchronize()
    return result, {'seconds':time.monotonic()-start, 'peak_torch_bytes':torch.cuda.max_memory_allocated(), 'free_before_bytes':before, 'free_after_bytes':torch.cuda.mem_get_info()[0]}

cases = []
for name, images in [('fixed24', fixtures[:24]), ('fixed47', fixtures), ('mixed47', mixed)]:
    reference, old_memory = run(images, original)
    candidate, new_memory = run(images, bounded)
    assert reference.keys() == candidate.keys()
    details = {}
    for key, value in reference.items():
        other = candidate[key]
        assert torch.equal(value, other), key
        assert value.dtype == other.dtype and value.shape == other.shape and value.stride() == other.stride(), key
        details[key] = {'shape':list(value.shape), 'dtype':str(value.dtype), 'stride':list(value.stride()), 'bitwise_equal':True, 'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    assert new_memory['peak_torch_bytes'] < old_memory['peak_torch_bytes']
    cases.append({'case':name,'images':len(images),'passed':True,'original':old_memory,'chunked':new_memory,'outputs':details,'image_tokens':int(candidate['image_grid_thw'].prod(-1).sum().item())//4})
    del reference, candidate
receipt = {'passed':True,'chunk_size':4,'cases':cases,'fixed_fixture_sha256':hashes,'helper_sha256':hashlib.sha256(HELPER_SOURCE.encode()).hexdigest(),'created_unix':time.time(),'scope':'Same 24/47 synthetic fixtures and mixed sizes, original full GPU batch versus bounded GPU transforms with CPU concat; no resize or dtype changes'}
(p/'results/image-chunks-full-qualification-09.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
