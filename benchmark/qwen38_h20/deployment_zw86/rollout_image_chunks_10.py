"""Gate full rollout on live pilot pressure, then complete deployment qualification."""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from native_run import execute, json_result

root=Path(__file__).parent
plan=json.loads((root/'task.json').read_text())
assert os.environ['CB_SID']==plan['session']
manifest=json.loads((root/'deployment-source-09.json').read_text())

def command(node,code,key,timeout=900):
    result=execute(['cb','alloc',plan['allocation'],'run','--node',node,'--timeout',str(timeout),'--','python3','-c',code],key=key,timeout=timeout+45)
    assert result.returncode==0,result.stdout[-2500:]+result.stderr[-5000:]
    return json_result(result.stdout)

def wait_health(nodes):
    deadline=time.monotonic()+2400
    while True:
        code=f"""import concurrent.futures,json,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
def health(url):
 try:
  with opener.open(url+'/health',timeout=5) as r:return {{'url':url,'ready':r.status==200}}
 except Exception as e:return {{'url':url,'ready':False,'error':str(e)}}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:print(json.dumps(list(pool.map(health,{[rep['url'] for n in nodes for rep in n['replicas']]!r}))))
"""
        result=command(plan['router_node'],code,'image10-health-'+str(time.time_ns()),300)
        print(json.dumps({'phase':'worker readiness','result':result}),flush=True)
        if all(r['ready'] for r in result):return
        assert time.monotonic()<deadline
        time.sleep(30)

def update_node(node):
    i=plan['nodes'].index(node)
    replicas=[dict(rep,attempt=f'zw86-image04-n{i}-r{r}-10') for r,rep in enumerate(node['replicas'])]
    install=(root/'install_source_remote_09.py').read_text().replace('PROJECT',repr(plan['project'])).replace('MANIFEST',repr(manifest)).replace('NODE',repr(node['node']))
    code=f"""import json,os,subprocess,sys,time
from pathlib import Path
p=Path({plan['project']!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'));from runtime import PYTHON,environment
env=environment();env['SGLANG_QWEN4_IMAGE_PREPROCESS_CHUNK_SIZE']='4';env['SGLANG_MOE_CONFIG_DIR']=str(p/'data/moe-tuned-plain-tp4ep4');env['SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED']='0';env['MAX_JOBS']='8'
for rep in {node['replicas']!r}:
 rec=json.loads((p/'logs'/(rep['attempt']+'.process.json')).read_text());proc=Path('/proc')/str(rec['pid']);fields=(proc/'stat').read_text().rsplit(')',1)[1].split();cmd=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))];assert fields[0]!='Z' and cmd==rec['command'][2:] and os.getpgid(rec['pid'])==rec['pgid']
 r=subprocess.run([PYTHON,str(p/'scripts/manage_service.py'),'stop','--attempt',rep['attempt']],env=env,capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr;assert not json.loads(r.stdout)['alive']
deadline=time.time()+60
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
 assert time.time()<deadline;time.sleep(2)
exec({install!r})
started=[]
for rep in {replicas!r}:
 args=['start','--attempt',rep['attempt'],'--tp','4','--ep','4','--base-gpu',str(rep['base_gpu']),'--port',str(rep['port']),'--max-running','64','--graph-bs','64','--chunk-size','8192','--mem-fraction','0.82','--ssm-dtype','bfloat16','--interleave-memory','--extra','--model-loader-extra-config','{{"enable_multithread_load":true}}','--linear-attn-prefill-backend','flashinfer','--speculative-algorithm','NEXTN','--speculative-num-steps','3','--speculative-num-draft-tokens','4','--speculative-eagle-topk','1','--linear-attn-verify-backend','triton']
 r=subprocess.run([PYTHON,str(p/'scripts/manage_service.py'),*args],env=env,capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr;record=json.loads(r.stdout);record.update(node={node['node']!r},url=rep['url'],image_preprocess_chunk_size=4,process_start_ticks=int((Path('/proc')/str(record['pid'])/'stat').read_text().rsplit(')',1)[1].split()[19]));started.append({{'replica':rep,'record':record}})
print(json.dumps({{'node':{node['node']!r},'replicas':started}}))
"""
    spin=subprocess.run(['cb','spin',node['node'],'off'],capture_output=True,text=True,timeout=90)
    (root/f'image10-spin-n{i}.receipt.txt').write_text(spin.stdout+spin.stderr)
    assert spin.returncode==0,spin.stderr
    result=command(node['node'],code,'image10-rollout-n'+str(i))
    (root/f'image10-rollout-n{i}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

try:
    deadline=time.monotonic()+2400
    while not (root/'pilot-replicas-09.json').exists():
        assert not (root/'pilot-image-chunks-09.error.json').exists(),(root/'pilot-image-chunks-09.error.json').read_text() if (root/'pilot-image-chunks-09.error.json').exists() else ''
        assert time.monotonic()<deadline;time.sleep(15)
    pilot=json.loads((root/'pilot-replicas-09.json').read_text())
    pilot_node=dict(plan['nodes'][0],replicas=[j['replica'] for j in pilot])
    wait_health([pilot_node])
    pressure=(root/'qualify_visual_pressure_remote.py').read_text().replace('PROJECT',repr(plan['project'])).replace('WORKERS',repr([r['url'] for r in pilot_node['replicas']])).replace('visual-pressure-qualification.json','visual-pressure-pilot-image09.json')
    result=command(plan['router_node'],pressure,'visual-pressure-pilot-image09',1200)
    assert result['passed'];(root/'visual-pressure-pilot-image09.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'pilot passed4 combined24/47-page requests'}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:updates=list(pool.map(update_node,plan['nodes'][1:]))
    plan['nodes'][0]=pilot_node
    services=[j['record'] for j in pilot]
    for update in updates:
        node=next(n for n in plan['nodes'] if n['node']==update['node'])
        node['replicas']=[j['replica'] for j in update['replicas']];services.extend(j['record'] for j in update['replicas'])
    # Pilot launch records may predate adding ticks to the launcher; collect them independently.
    ticks=command(plan['router_node'],f"""import json
from pathlib import Path
print(json.dumps({{str(pid):int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]) for pid in {[s['pid'] for s in services if s['node']==plan['router_node']]!r}}}))
""",'pilot-process-ticks-image10',300)
    for s in services:
        if s['node']==plan['router_node']:s['process_start_ticks']=ticks[str(s['pid'])]
    plan.update(services=services,software_commit=manifest['engine_commit'],qualification_round='image10',source_manifest=manifest,deployment_scripts_commit=manifest['engine_commit'],fully_qualified=False,status='new image engine launched; completing all qualifications')
    (root/'task.json').write_text(json.dumps(plan,indent=2)+'\n')
    wait_health(plan['nodes'])
    ns={'__file__':str(root/'controller.py')};source=(root/'controller.py').read_text();exec(compile(source.split('\ntry:\n    if not plan.get("storage"):')[0],'controller.py','exec'),ns)
    ns['node_command']=lambda node,code,timeout=90:command(node,code,'image10-native-'+str(time.time_ns()),max(timeout,300))
    plan=ns['plan']
    plan['functional']=ns['functional_validation']();plan['long_context']=ns['long_validation']();plan['combined_validation']=ns['router_validation']()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:plan['final_checks']=list(pool.map(ns['final_process_check'],plan['nodes']))
    plan.update(status='ready;32H20 basic verification passed, final protocol pending',ready_workers=8,error=None)
    (root/'task.json').write_text(json.dumps(plan,indent=2)+'\n')
    backup_code=f"""import shutil
from pathlib import Path
p=Path({plan['project']!r})
for name in ['visual-pressure-qualification.json','new-worker-vision-tools-budget.json']:
 source=p/'results'/name;dest=p/'results'/('before-image10-'+name);assert not dest.exists();shutil.copyfile(source,dest)
print('{{"backed_up":true}}')
"""
    command(plan['router_node'],backup_code,'image10-preserve-old-results',300)
    (root/'visual_memory_observer10.py').write_text((root/'visual_memory_observer.py').read_text().replace('04','10'))
    observer=(root/'start_visual_memory_observers.py').read_text().replace('04','10').replace('visual_memory_observer.py','visual_memory_observer10.py')
    exec(compile(observer,'start-new-observers','exec'),{'__file__':str(root/'start_visual_memory_observers.py')})
    with (root/'runtime-identities-image10.log').open('xb') as log:r=subprocess.run(['python3',str(root/'collect_runtime_identities.py')],stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0
    with (root/'final-pipeline-image10.log').open('xb') as log:r=subprocess.run(['python3',str(root/'final_pipeline.py')],stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0
    (root/'image10-rollout-complete.json').write_text(json.dumps({'passed':True,'unix':time.time()},indent=2)+'\n')
except BaseException as error:
    (root/'image10-rollout-error.json').write_text(json.dumps({'error':repr(error),'unix':time.time()},indent=2)+'\n')
    raise
