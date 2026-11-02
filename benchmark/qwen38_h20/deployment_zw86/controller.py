"""Deploy additional replicas and validate the combined H20 service."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from native_run import json_result

ROOT = Path(__file__).parent
TASK = ROOT / "task.json"
plan = json.loads(TASK.read_text())
assert os.environ["CB_SID"] == plan["session"]
PROJECT = plan["project"]
BUNDLE = Path(str(ROOT / "bundle"))
SCRIPT_HASHES = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in BUNDLE.glob("*.py")
}
SOURCE = Path(str(ROOT / "sglang/python/sglang"))
SOURCE_FILES = [
    "srt/environ.py",
    "srt/utils/common.py", "srt/multimodal/processors/base_processor.py",
    "srt/models/qwen4_exp.py", "srt/models/qwen4_exp_mtp.py",
    "srt/models/qwen4_exp_ple_table.py", "srt/configs/qwen4_exp.py",
    "srt/managers/scheduler.py", "srt/model_loader/loader.py",
    "srt/speculative/eagle_worker_v2.py", "srt/speculative/eagle_worker_common.py",
    "srt/speculative/eagle_info.py", "srt/speculative/eagle_utils.py",
    "srt/speculative/eagle_post_verify_graphs.py",
    "kernels/jit/csrc/elementwise/grouped_gemma_rmsnorm.cuh",
    "srt/managers/tp_worker.py", "srt/managers/utils.py",
    "srt/managers/tokenizer_manager.py", "srt/runtime_context.py",
]
SOURCE_HASHES = {
    name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    for name in SOURCE_FILES
}
NEW_URLS = [rep["url"] for node in plan["nodes"] for rep in node["replicas"]]
ALL_URLS = plan["existing_worker_urls"] + NEW_URLS
BASE_URL = f"http://{plan['router_ip']}:{plan['router_port']}/v1"
NEW_WORKERS = len(NEW_URLS)
ALL_WORKERS = len(ALL_URLS)


def save(**changes):
    plan.update(changes, updated_unix=time.time())
    tmp = TASK.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=2) + "\n")
    tmp.replace(TASK)
    print(json.dumps({k: plan.get(k) for k in ["status", "ready_workers", "error"]}), flush=True)


def node_command(node_id, code, timeout=90):
    key = "qwen38-native-" + str(uuid.uuid4())
    argv = [
        "cb", "alloc", plan["allocation"], "run", "--node", node_id,
        "--timeout", str(timeout), "--idempotency-key", key, "--", "python3", "-c", code,
    ]
    (ROOT / (key + ".argv.json")).write_text(json.dumps(argv))
    for attempt in range(3):
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout + 45)
        except subprocess.TimeoutExpired as exc:
            fragments = [exc.stdout or "", exc.stderr or ""]
            (ROOT / (key + ".receipt.txt")).write_text(
                "\n".join(x.decode(errors="replace") if isinstance(x, bytes) else x for x in fragments)
            )
            raise RuntimeError(f"Native command outcome unknown; inspect or reuse exact argv/key {key}") from exc
        before_submission = (
            result.returncode == 3
            and "authenticated heartbeat has expired" in result.stderr
            and "submitted" not in result.stderr
            and "command=ncmd-" not in result.stderr
            and not result.stdout.strip()
        )
        if before_submission and attempt < 2:
            time.sleep(15)
            continue
        break
    if result.returncode:
        (ROOT / (key + ".receipt.txt")).write_text(result.stdout + result.stderr)
        raise RuntimeError(node_id + ": " + result.stdout[-3000:] + result.stderr[-4000:])
    return json_result(result.stdout)


def prepare_storage():
    return node_command(plan["router_node"], f"""
import hashlib,json,shutil,time
from pathlib import Path
p=Path({PROJECT!r});source=Path({plan['source_project']!r});scripts=Path({plan['source_scripts']!r})
mounts=[line for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' /apdcephfs_zwfy8/share_305110755 ' in line]
assert mounts and all('fuse.dop-fuse' in line for line in mounts),mounts
assert source.is_dir() and scripts.is_dir()
p.mkdir(exist_ok=True)
for name in ['scripts','logs','results','wandb']:(p/name).mkdir(exist_ok=True)
for name in ['workdir','checkpoints','data']:
 target=p/name
 if target.is_symlink():assert target.resolve()==(source/name).resolve()
 else:assert not target.exists();target.symlink_to(source/name,target_is_directory=True)
expected={SCRIPT_HASHES!r}
for name,sha in expected.items():
 original=scripts/name;assert hashlib.sha256(original.read_bytes()).hexdigest()==sha,name
 target=p/'scripts'/name
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==sha,name
 else:shutil.copyfile(original,target)
marker=p/'results/shared-read-marker.json'
if not marker.exists():marker.write_text(json.dumps({{'project':str(p),'unix':time.time()}}))
print(json.dumps({{'project':str(p),'mountinfo':mounts,'script_sha256':expected,'shared_marker':hashlib.sha256(marker.read_bytes()).hexdigest()}}))
""")


def preflight(node):
    profile = r'''
import hashlib,importlib.metadata as md,json
from pathlib import Path
import torch,sglang
source=Path(sglang.__file__).parent;expected=EXPECTED
for relative,sha in expected.items():assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==sha,relative
assert torch.cuda.is_available() and torch.cuda.device_count()==8
devices=[]
for i in range(8):
 with torch.cuda.device(i):
  x=torch.ones((16,16),device='cuda');assert (x@x)[0,0].item()==16;del x
 devices.append({'index':i,'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i)})
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'packages':{k:md.version(k) for k in ['flashinfer-python','sglang-kernel','triton','transformers','sglang-router']},'source':str(source),'source_sha256':expected,'devices':devices}))
'''.replace("EXPECTED", repr(SOURCE_HASHES))
    return node_command(node["node"], f"""
import hashlib,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
p=Path({PROJECT!r});mounts=[line for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' /apdcephfs_zwfy8/share_305110755 ' in line]
assert mounts and all('fuse.dop-fuse' in line for line in mounts)
assert hashlib.sha256((p/'results/shared-read-marker.json').read_bytes()).hexdigest()=={plan['storage']['shared_marker']!r}
deadline=time.monotonic()+60
while True:
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
 if not apps.strip():break
 assert time.monotonic()<deadline,'CUDA processes still active before launch: '+apps
 time.sleep(2)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader'],text=True)
for line in gpu.splitlines():assert int(line.split(',')[-1].strip().split()[0])==0,line
for port in [30000,29000,31000,31001]:
 with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('0.0.0.0',port))
model=Path('/root/.cache')/('qwen38-runtime-'+p.name)/'model/Qwen3.8-Flash-Next'
cache=json.loads((model/'node-cache-manifest.json').read_text());assert cache['passed'] and cache['phase']=='verified' and cache['node']=={node['node']!r}
assert len([r for r in cache['verified_files'] if r['expected_lfs_sha256'] is not None])==131
index=json.loads((model/'model.safetensors.index.json').read_text());shards=sorted(set(index['weight_map'].values()))
assert len(shards)==131 and all((model/x).is_file() for x in shards)
revision=(model/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
assert revision=='de4b8e4d43b917e7706784d8bb445c9af86a3540'
table=p/'data/moe-tuned-plain-tp4ep4/configs/triton_3_8_0/E=128,N=640,device_name=NVIDIA_H20.json'
assert hashlib.sha256(table.read_bytes()).hexdigest()=='2d788a1da86fbd8d74529745396b9abc3794fbbdff979cfd1e9101e050077286'
os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'))
from runtime import PYTHON,environment
env=environment();env['MAX_JOBS']='8';env['SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED']='0';env['SGLANG_MOE_CONFIG_DIR']=str(p/'data/moe-tuned-plain-tp4ep4')
r=subprocess.run([PYTHON,'-c',{profile!r}],env=env,text=True,capture_output=True,timeout=140)
assert r.returncode==0,r.stdout[-2000:]+r.stderr[-3000:]
result=json.loads(r.stdout.strip().splitlines()[-1]);result.update(node={node['node']!r},ip={node['ip']!r},prelaunch_cuda_apps=apps,gpu=gpu,mountinfo=mounts,model_revision=revision,weight_shards=len(shards),weight_bytes=sum((model/x).stat().st_size for x in shards),moe_config_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),verified_unix=time.time())
(p/'results'/('preflight-'+{node['node']!r}+'.json')).write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
""", timeout=240)


def start_script(node_id, name, args):
    return node_command(node_id, f"""
import json,os,subprocess,sys
from pathlib import Path
p=Path({PROJECT!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'))
from runtime import PYTHON,environment
env=environment();env['MAX_JOBS']='8';env['SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED']='0';env['SGLANG_MOE_CONFIG_DIR']=str(p/'data/moe-tuned-plain-tp4ep4')
args={args!r};attempt=args[args.index('--attempt')+1];record=p/'logs'/(attempt+'.process.json')
if record.exists():
 assert args[0]=='start';args[0]='status'
r=subprocess.run([PYTHON,str(p/'scripts'/{name!r}),*args],env=env,text=True,capture_output=True,timeout=90)
assert r.returncode==0,r.stdout[-2000:]+r.stderr[-3000:]
info=json.loads(r.stdout);info.pop('tail',None)
proc=Path('/proc')/str(info['pid']);assert proc.exists()
fields=(proc/'stat').read_text().rsplit(')',1)[1].split();assert fields[0]!='Z' and os.getpgid(info['pid'])==info['pgid']
cmd=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))]
if {name!r}=='manage_service.py':assert cmd==(info['command'][2:] if info['command'][:2]==['/usr/bin/numactl','--interleave=all'] else info['command'])
else:assert int(fields[19])==info['process_start_ticks']
info['process_start_ticks']=int(fields[19]);info['node']={node_id!r};print(json.dumps(info))
""", timeout=120)


def launch_node(node):
    services = []
    for rep in node["replicas"]:
        info = start_script(node["node"], "manage_service.py", [
            "start", "--attempt", rep["attempt"], "--tp", "4", "--ep", "4",
            "--base-gpu", str(rep["base_gpu"]), "--port", str(rep["port"]),
            "--max-running", "64", "--graph-bs", "64", "--chunk-size", "8192",
            "--mem-fraction", "0.82", "--ssm-dtype", "bfloat16", "--interleave-memory", "--extra",
            "--model-loader-extra-config", '{"enable_multithread_load":true}',
            "--linear-attn-prefill-backend", "flashinfer",
            "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
            "--speculative-num-draft-tokens", "4", "--speculative-eagle-topk", "1",
            "--linear-attn-verify-backend", "triton",
        ])
        info["url"] = rep["url"]
        services.append(info)
    return services


def check_node(node):
    return node_command(node["node"], f"""
import concurrent.futures,json,os,urllib.request
from pathlib import Path
p=Path({PROJECT!r});opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
def check(rep):
 info=json.loads((p/'logs'/(rep['attempt']+'.process.json')).read_text());proc=Path('/proc')/str(info['pid']);alive=proc.exists()
 if alive:
  fields=(proc/'stat').read_text().rsplit(')',1)[1].split();cmd=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))]
  alive=fields[0]!='Z' and cmd==(info['command'][2:] if info['command'][:2]==['/usr/bin/numactl','--interleave=all'] else info['command']) and os.getpgid(info['pid'])==info['pgid']
 result={{'attempt':rep['attempt'],'node':{node['node']!r},'url':rep['url'],'alive':alive,'ready':False,'log_tail':Path(info['log']).read_text(errors='replace')[-1300:]}}
 if alive:
  try:
   with opener.open(rep['url']+'/health',timeout=5) as response:result['ready']=response.status==200
  except Exception as exc:result['health_error']=str(exc)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(check,{node['replicas']!r}))
print(json.dumps(result))
""", timeout=45)


def functional_validation():
    return node_command(plan["router_node"], f"""
import concurrent.futures,json,time,urllib.request
from pathlib import Path
p=Path({PROJECT!r});opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}));model='Qwen/Qwen3.8-Flash-Next'
def chat(url,payload):
 req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={{'Content-Type':'application/json'}})
 with opener.open(req,timeout=240) as response:return json.load(response)
def check(url):
 records=[]
 for prompt,expected in [('Calculate 17 * 19. Reply with only the integer.','323'),('Reply with exactly: SGLANG_OK','SGLANG_OK'),('What is the capital of France? Reply with only the English city name.','Paris')]:
  start=time.monotonic();j=chat(url,{{'model':model,'messages':[{{'role':'user','content':prompt}}],'temperature':0,'max_tokens':64,'chat_template_kwargs':{{'enable_thinking':False}}}})
  content=j['choices'][0]['message']['content'].strip();assert content.rstrip('.')==expected,j
  records.append({{'expected':expected,'content':content,'passed':True,'usage':j.get('usage'),'seconds':time.monotonic()-start}})
 return {{'url':url,'cases':records,'passed':True}}
with concurrent.futures.ThreadPoolExecutor(max_workers={NEW_WORKERS}) as pool:short=list(pool.map(check,{NEW_URLS!r}))
result={{'short_workers':short,'created_unix':time.time()}}
(p/'results/new-worker-functional.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
""", timeout=300)


def long_validation():
    code = r'''
import concurrent.futures,json,math,time,urllib.request
from pathlib import Path
from transformers import AutoTokenizer
p=Path(PROJECT);tok=AutoTokenizer.from_pretrained(str(p/'checkpoints/Qwen3.8-Flash-Next'))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));unit='A notebook records ordinary observations about weather, trees, books, and streets. '
ids=tok.encode(unit*3000,add_special_tokens=False)[:32768];assert len(ids)==32768
prompt=tok.decode(ids[:16384])+'\nMEMORY: The access code for package Orion is SKY7319.\n'+tok.decode(ids[16384:])+'\nWhat is the access code for package Orion? Reply with only the code, no formatting.'
def check(url):
 body={'model':'Qwen/Qwen3.8-Flash-Next','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});start=time.monotonic()
 with opener.open(req,timeout=300) as response:j=json.load(response)
 content=j['choices'][0]['message']['content'].strip();assert content.strip('`"\' .\n')=='SKY7319' and 32768<=j['usage']['prompt_tokens']<=33280,j
 return {'url':url,'content':content,'passed':True,'usage':j['usage'],'seconds':time.monotonic()-start}
with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as pool:result=list(pool.map(check,URLS))
j={'workers':result,'created_unix':time.time(),'filler_tokens':32768}
(p/'results/new-worker-32k.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
'''.replace("PROJECT", repr(PROJECT)).replace("URLS", repr(NEW_URLS))
    return node_command(plan["router_node"], f"""
import json,os,subprocess,sys
from pathlib import Path
p=Path({PROJECT!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'))
from runtime import PYTHON,environment
r=subprocess.run([PYTHON,'-c',{code!r}],env=environment(),text=True,capture_output=True,timeout=350)
assert r.returncode==0,r.stdout[-2500:]+r.stderr[-3500:];print(r.stdout)
""", timeout=390)


def router_validation():
    return node_command(plan["router_node"], f"""
import concurrent.futures,json,time,urllib.request
from pathlib import Path
p=Path({PROJECT!r});base='http://127.0.0.1:{plan['router_port']}';opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}));urls={ALL_URLS!r}
def get(url,path):
 with opener.open(url+path,timeout=20) as response:return json.load(response)
deadline=time.monotonic()+90
while True:
 try:
  workers=get(base,'/workers');assert len(workers['workers'])=={ALL_WORKERS}
  assert {{w['url'].rstrip('/') for w in workers['workers'] if w['is_healthy']}}==set(urls);break
 except Exception:
  assert time.monotonic()<deadline;time.sleep(2)
models=get(base,'/v1/models');assert 'Qwen/Qwen3.8-Flash-Next' in [m['id'] for m in models['data']]
configs=[]
for url in urls:
 info=get(url,'/server_info');model_info=get(url,'/model_info')
 for key,value in {{'tp_size':4,'ep_size':4,'mamba_ssm_dtype':'bfloat16','speculative_algorithm':'EAGLE','speculative_num_steps':3,'speculative_num_draft_tokens':4,'speculative_eagle_topk':1,'linear_attn_verify_backend':'triton','ple_offload_embedding':True,'chunked_prefill_size':8192,'mem_fraction_static':0.82,'max_running_requests':64,'max_req_input_len':262138,'allow_auto_truncate':False}}.items():assert info.get(key)==value,(url,key,info.get(key))
 assert model_info['served_model_name']=='Qwen/Qwen3.8-Flash-Next'
 configs.append({{'url':url,'server_info':info,'model_info':model_info}})
def check(i):
 stream=i%2==1;body={{'model':'Qwen/Qwen3.8-Flash-Next','messages':[{{'role':'user','content':'Reply with exactly: SGLANG_OK'}}],'temperature':0,'max_tokens':64,'stream':stream,'chat_template_kwargs':{{'enable_thinking':False}}}}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={{'Content-Type':'application/json'}})
 with opener.open(req,timeout=90) as response:
  if stream:
   parts=[];done=False
   for line in response:
    raw=line.decode().strip()
    if not raw.startswith('data: '):continue
    raw=raw[6:]
    if raw=='[DONE]':done=True;break
    choices=json.loads(raw).get('choices',[])
    if choices:parts.append(choices[0].get('delta',{{}}).get('content') or '')
   content=''.join(parts).strip();assert done
  else:content=json.load(response)['choices'][0]['message']['content'].strip()
 assert content=='SGLANG_OK',content
 return {{'request':i,'stream':stream,'content':content,'passed':True}}
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:checks=list(pool.map(check,range(32)))
result={{'base_url':{BASE_URL!r},'workers':workers,'configs':configs,'models':models,'routed_checks':checks,'verified_unix':time.time()}}
(p/'results/combined-router-validation.json').write_text(json.dumps(result,indent=2)+'\\n')
summary={{'base_url':{BASE_URL!r},'healthy_workers':{ALL_WORKERS},'routed_checks_passed':len(checks),'capacity':[{{'url':x['url'],'max_req_input_len':x['server_info']['max_req_input_len'],'slots':[s.get('effective_max_running_requests_per_dp') for s in x['server_info']['internal_states']],'token_capacity':[s.get('memory_usage',{{}}).get('token_capacity') for s in x['server_info']['internal_states']]}} for x in configs]}}
print(json.dumps(summary))
""", timeout=240)


def final_process_check(node):
    expected_starts = {
        s["attempt"]: s["process_start_ticks"]
        for s in plan.get("services", []) if s["node"] == node["node"]
    }
    return node_command(node["node"], f"""
import hashlib,json,os,re,subprocess,time
from pathlib import Path
p=Path({PROJECT!r});records=[];checks=[];expected_starts={expected_starts!r}
for rep in {node['replicas']!r}:
 info=json.loads((p/'logs'/(rep['attempt']+'.process.json')).read_text());proc=Path('/proc')/str(info['pid']);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
 cmd=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))]
 assert fields[0]!='Z' and cmd==(info['command'][2:] if info['command'][:2]==['/usr/bin/numactl','--interleave=all'] else info['command']) and os.getpgid(info['pid'])==info['pgid']
 if rep['attempt'] in expected_starts:assert int(fields[19])==expected_starts[rep['attempt']]
 records.append({{'attempt':rep['attempt'],'pid':info['pid'],'pgid':info['pgid'],'start_ticks':int(fields[19]),'port':info['port']}})
 raw=Path(info['log']).read_bytes();text=raw.decode(errors='replace')
 events=len(re.findall(r'Scheduler hit an exception|Traceback|CUDA error:|RuntimeError:',text));retractions=sum('#retracted_reqs:' in line for line in text.splitlines())
 assert events==0 and retractions==0,(rep['attempt'],events,retractions)
 checks.append({{'attempt':rep['attempt'],'runtime_error_events':events,'retraction_lines':retractions,'compiler_data_race_warnings':text.count('Warning: Data race detected:'),'cc_state_query_messages':text.count('Error querying CC state: Function Not Found'),'log_sha256':hashlib.sha256(raw).hexdigest()}})
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
result={{'node':{node['node']!r},'processes':records,'logs':checks,'gpu_apps':apps,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader'],text=True),'process_identity_scope':'Container service command/PID/PGID/start-time verified; NVML host PIDs are not mapped through container procfs','verified_unix':time.time()}}
(p/'results'/('final-'+{node['node']!r}+'.json')).write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
""", timeout=90)


try:
    if not plan.get("storage"):
        save(status="preparing shared project", storage=prepare_storage(), error=None)
    if not plan.get("preflight"):
        save(status="checking four nodes")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            preflights = list(pool.map(preflight, plan["nodes"]))
        save(status=f"launching {NEW_WORKERS} replicas", preflight=preflights)
    if not plan.get("services"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            services = list(pool.map(launch_node, plan["nodes"]))
        save(status=f"waiting for {NEW_WORKERS} new workers", services=[s for batch in services for s in batch])
    deadline = time.monotonic() + 3600
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(check_node, plan["nodes"]))
        health = [h for batch in batches for h in batch]
        save(health=health, ready_workers=sum(h["ready"] for h in health))
        assert all(h["alive"] for h in health), health
        if all(h["ready"] for h in health):
            break
        assert time.monotonic() < deadline, "Worker startup deadline exceeded"
        time.sleep(30)
    if not plan.get("functional"):
        save(status=f"validating {NEW_WORKERS} new workers", functional=functional_validation())
    if not plan.get("long_context"):
        save(status=f"validating 32K inputs on {NEW_WORKERS} new workers", long_context=long_validation())
    router = start_script(plan["router_node"], "manage_router.py", [
        "start", "--attempt", plan["router_attempt"], "--port", str(plan["router_port"]),
        "--prometheus-port", str(plan["router_metrics_port"]), "--policy", "round_robin",
        "--workers", *ALL_URLS,
    ])
    save(status=f"validating combined {ALL_WORKERS}-worker router", router=router)
    validation = router_validation()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        final_checks = list(pool.map(final_process_check, plan["nodes"]))
    save(status=f"ready; {plan['total_gpus']}H20 service left running", combined_validation=validation,
         final_checks=final_checks, ready_workers=ALL_WORKERS, completed_unix=time.time(), error=None)
    service = {
        "status": "running", "base_url": BASE_URL, "model": "Qwen/Qwen3.8-Flash-Next",
        "model_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "cluster": plan["cluster"], "gpus": plan["total_gpus"], "nodes": len(plan["nodes"]), "replicas": ALL_WORKERS,
        "layout": {"tp": 4, "ep": 4, "mtp_steps": 3, "draft_tokens": 4,
                   "dtype": "bfloat16", "chunked_prefill_size": 8192, "mem_fraction_static": 0.82},
        "new_allocation": plan["allocation"], "new_allocation_owner": plan["session"],
        "expires_at": plan["expires_at"], "software_commit": plan["software_commit"],
        "software_difference_from_bj87": "GPU JPEG RGB decode and opt-in bounded PLE cudaHostRegister; 256 MiB registrations under NUMA interleave=all",
        "project": PROJECT, "workers": ALL_URLS, "router": router,
        "validation": validation, "performance_note": "Deployment verification only; no new aggregate benchmark claim",
    }
    (ROOT / "service.json").write_text(json.dumps(service, indent=2) + "\n")
except BaseException as error:
    save(status="failed; inspect running services before recovery", error=repr(error))
    raise
