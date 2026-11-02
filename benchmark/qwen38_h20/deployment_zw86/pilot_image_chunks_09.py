"""Stop one node, qualify the candidate, install committed source, start its replicas."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from native_run import execute, json_result

root = Path(__file__).parent
plan = json.loads((root/'task.json').read_text())
assert os.environ['CB_SID'] == plan['session']
node = plan['nodes'][0]
manifest = json.loads((root/'deployment-source-09.json').read_text())
assert manifest['engine_commit'] == subprocess.check_output(['git','-C',str(root/'sglang'),'rev-parse','HEAD'],text=True).strip()

def command(code, key, timeout=900):
    result = execute(['cb','alloc',plan['allocation'],'run','--node',node['node'],'--timeout',str(timeout),'--','python3','-c',code], key=key, timeout=timeout+45)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-5000:]
    return json_result(result.stdout)

try:
    stopped=command(f"""import json,os,subprocess,sys,time
from pathlib import Path
p=Path({plan['project']!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'));from runtime import PYTHON,environment
records=[]
for rep in {node['replicas']!r}:
 rec=json.loads((p/'logs'/(rep['attempt']+'.process.json')).read_text());proc=Path('/proc')/str(rec['pid']);fields=(proc/'stat').read_text().rsplit(')',1)[1].split();cmd=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))]
 assert fields[0]!='Z' and os.getpgid(rec['pid'])==rec['pgid'] and cmd==rec['command'][2:]
 r=subprocess.run([PYTHON,str(p/'scripts/manage_service.py'),'stop','--attempt',rep['attempt']],env=environment(),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
 j=json.loads(r.stdout);assert not j['alive'];j.pop('tail',None);records.append(j)
deadline=time.time()+60
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
 assert time.time()<deadline;time.sleep(2)
print(json.dumps({{'stopped':records}}))
""", 'stop-image-pilot-09')
    (root/'pilot-stopped-09.json').write_text(json.dumps(stopped,indent=2)+'\n')
    print(json.dumps({'phase':'pilot node stopped; qualifying exact output'}),flush=True)
    helper=(root/'sglang/python/sglang/srt/multimodal/processors/qwen_vl_image_chunks.py').read_text()
    code=(root/'qualify_image_chunks_full_remote.py').read_text().replace('PROJECT',repr(plan['project'])).replace('HELPER_SOURCE',repr(helper))
    outer=f"""import os,subprocess,sys
from pathlib import Path
p=Path({plan['project']!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'));from runtime import PYTHON,environment
r=subprocess.run([PYTHON,'-c',{code!r}],env=environment(),capture_output=True,text=True,timeout=600);(p/'logs/image-chunks-full-qualification-09.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout[-3000:]+r.stderr[-5000:];print(r.stdout)
"""
    proof=command(outer,'image-chunks-full-09');assert proof['passed']
    (root/'image-chunks-full-qualification-09.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'phase':'original/candidate full24/47 and mixed bitwise qualification passed','cases':proof['cases']}),flush=True)
    code=(root/'install_source_remote_09.py').read_text().replace('PROJECT',repr(plan['project'])).replace('MANIFEST',repr(manifest)).replace('NODE',repr(node['node']))
    installed=command(code,'install-image-pilot-09');assert installed['passed']
    (root/'pilot-source-09.json').write_text(json.dumps(installed,indent=2)+'\n')
    replicas=[]
    for i,rep in enumerate(node['replicas']):
        new=dict(rep,attempt='zw86-image04-n0-r'+str(i)+'-09')
        code=f"""import json,os,subprocess,sys
from pathlib import Path
p=Path({plan['project']!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'));from runtime import PYTHON,environment
env=environment();env['SGLANG_QWEN4_IMAGE_PREPROCESS_CHUNK_SIZE']='4';env['SGLANG_MOE_CONFIG_DIR']=str(p/'data/moe-tuned-plain-tp4ep4');env['SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED']='0';env['MAX_JOBS']='8'
args=['start','--attempt',{new['attempt']!r},'--tp','4','--ep','4','--base-gpu',{str(rep['base_gpu'])!r},'--port',{str(rep['port'])!r},'--max-running','64','--graph-bs','64','--chunk-size','8192','--mem-fraction','0.82','--ssm-dtype','bfloat16','--interleave-memory','--extra','--model-loader-extra-config','{{"enable_multithread_load":true}}','--linear-attn-prefill-backend','flashinfer','--speculative-algorithm','NEXTN','--speculative-num-steps','3','--speculative-num-draft-tokens','4','--speculative-eagle-topk','1','--linear-attn-verify-backend','triton']
r=subprocess.run([PYTHON,str(p/'scripts/manage_service.py'),*args],env=env,capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr;j=json.loads(r.stdout);j['node']={node['node']!r};j['url']={new['url']!r};j['image_preprocess_chunk_size']=4;j['process_start_ticks']=int((Path('/proc')/str(j['pid'])/'stat').read_text().rsplit(')',1)[1].split()[19]);print(json.dumps(j))
"""
        j=command(code,'start-image-pilot-09-r'+str(i));replicas.append({'replica':new,'record':j})
    (root/'pilot-replicas-09.json').write_text(json.dumps(replicas,indent=2)+'\n')
    print(json.dumps({'phase':'pilot replicas launched','replicas':replicas}),flush=True)
except BaseException as error:
    (root/'pilot-image-chunks-09.error.json').write_text(json.dumps({'error':repr(error),'unix':time.time()},indent=2)+'\n')
    raise
