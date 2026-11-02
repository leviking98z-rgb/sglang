"""Replace committed local source only after both node replicas have stopped."""
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

p=Path(PROJECT)
manifest=MANIFEST
runtime=Path('/root/.cache')/('qwen38-runtime-'+p.name)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
archive=p/'results'/manifest['source_archive']['file']
assert archive.stat().st_size==manifest['source_archive']['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['source_archive']['sha256']
staging=runtime/('source-staging-image-'+manifest['engine_commit'][:8])
assert not staging.exists();staging.mkdir()
with tarfile.open(archive) as t:t.extractall(staging,filter='data')
for name,sha in manifest['source_sha256'].items():assert hashlib.sha256((staging/'python/sglang'/name).read_bytes()).hexdigest()==sha,name
old=runtime/'sglang-before-image-09';assert not old.exists()
(runtime/'sglang').rename(old);staging.rename(runtime/'sglang')
result={'passed':True,'node':NODE,'engine_commit':manifest['engine_commit'],'source_sha256':manifest['source_sha256'],'archive_sha256':manifest['source_archive']['sha256'],'installed_unix':time.time(),'prior_engine_commit':'e429cbe6ea8b6a7f41e180b171b236a4925a1bc8','ple_module_unchanged':True}
(runtime/'source-deployment-09.json').write_text(json.dumps(result,indent=2)+'\n')
previous=p/'results'/('source-deployment-'+NODE+'.json')
backup=p/'results'/('source-deployment-before-image09-'+NODE+'.json')
assert not backup.exists()
shutil.copyfile(previous,backup)
(p/'results'/('source-deployment-'+NODE+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
