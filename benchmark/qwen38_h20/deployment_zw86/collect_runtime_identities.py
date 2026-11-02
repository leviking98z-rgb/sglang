"""Fingerprint the actual local source, model metadata, and live worker identities."""
import concurrent.futures
import json
import os
from pathlib import Path
from native_run import execute, json_result

root = Path(__file__).parent
plan = json.loads((root / "task.json").read_text())
assert os.environ["CB_SID"] == plan["session"]
ns = {"__file__": str(root / "controller.py")}
exec(compile((root / "controller.py").read_text().split('\ntry:\n    if not plan.get("storage"):')[0], "controller.py", "exec"), ns)

def collect(node):
    code = f"""import hashlib,json,os,time
from pathlib import Path
p=Path({plan['project']!r});runtime=Path('/root/.cache')/('qwen38-runtime-'+p.name);m=runtime/'model/Qwen3.8-Flash-Next';source=runtime/'sglang/python/sglang'
deployment=json.loads((p/'results'/('source-deployment-'+{node['node']!r}+'.json')).read_text());assert deployment['passed'] and deployment['engine_commit']=={plan['software_commit']!r}
expected={ns['SOURCE_HASHES']!r};actual={{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in expected}};assert actual==expected
metadata={{f.name:{{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}} for f in m.iterdir() if f.is_file() and f.suffix in ['.json','.jinja','.txt','.model','.tiktoken']}}
cache=json.loads((m/'node-cache-manifest.json').read_text());assert cache['passed'] and cache['phase']=='verified'
workers=[]
for rep in {node['replicas']!r}:
 record=json.loads((p/'logs'/(rep['attempt']+'.process.json')).read_text());proc=Path('/proc')/str(record['pid']);fields=(proc/'stat').read_text().rsplit(')',1)[1].split();cmd=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))]
 assert fields[0]!='Z' and cmd==(record['command'][2:] if record['command'][:2]==['/usr/bin/numactl','--interleave=all'] else record['command']) and os.getpgid(record['pid'])==record['pgid']
 workers.append({{'attempt':rep['attempt'],'url':rep['url'],'pid':record['pid'],'pgid':record['pgid'],'start_ticks':int(fields[19]),'command':cmd}})
result={{'passed':True,'node':{node['node']!r},'engine_commit':{plan['software_commit']!r},'local_source':str(source),'local_model':str(m),'source_sha256':actual,'model_metadata':metadata,'worker_identities':workers,'created_unix':time.time(),'supersedes':'Earlier asset-fingerprints source fields describe shared staging; this record fingerprints actual node-local deployed source'}}
(p/'results'/('runtime-identity-'+{node['node']!r}+'-04.json')).write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
"""
    result = execute(["cb", "alloc", plan["allocation"], "run", "--node", node["node"],
                      "--timeout", "90", "--", "python3", "-c", code],
                     key="runtime-identity-04-" + str(plan["nodes"].index(node)), timeout=135)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-2000:]
    return json_result(result.stdout)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    identities = list(pool.map(collect, plan["nodes"]))
reference = {k: v for k, v in identities[0]["model_metadata"].items()
             if k != "node-cache-manifest.json"}
for identity in identities:
    assert {k: v for k, v in identity["model_metadata"].items()
            if k != "node-cache-manifest.json"} == reference
(root / "runtime-identities-04.json").write_text(json.dumps(identities, indent=2) + "\n")
print(json.dumps({"passed": True, "nodes": len(identities), "workers": 8,
                  "metadata_hashes": reference}))
