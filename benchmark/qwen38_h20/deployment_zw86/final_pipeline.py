"""Finish service qualifications and prepare export after all replicas are ready."""
import json
import os
import subprocess
import time
import concurrent.futures
from pathlib import Path
from native_run import execute, json_result
ROOT=Path(__file__).parent
plan=json.loads((ROOT/'task.json').read_text())
assert os.environ['CB_SID']==plan['session']
validation_source=json.loads((ROOT/'validation-source-05.json').read_text())
plan['validation_scripts_commit']=validation_source['commit']
try:
    deadline=time.monotonic()+9000
    while True:
        plan=json.loads((ROOT/'task.json').read_text())
        if plan['status'].startswith('ready;'):break
        assert not plan['status'].startswith('failed;'),plan.get('error')
        assert time.monotonic()<deadline
        time.sleep(30)
    plan['validation_scripts_commit']=validation_source['commit']
    print(json.dumps({'phase':'basic service qualification passed','workers':plan.get('ready_workers')}),flush=True)
    r=subprocess.run(['python3',str(ROOT/'qualify_new_workers.py')],capture_output=True,text=True,timeout=450)
    (ROOT/'functional-qualification-01.receipt.txt').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout[-3000:]+r.stderr[-3000:]
    print(json.dumps({'phase':'vision/tools/history/output-budget passed'}),flush=True)
    urls=[r['url'] for n in plan['nodes'] for r in n['replicas']]
    pressure=(ROOT/'qualify_visual_pressure_remote.py').read_text().replace('PROJECT',repr(plan['project'])).replace('WORKERS',repr(urls))
    r=execute(['cb','alloc',plan['allocation'],'run','--node',plan['router_node'],'--timeout','1200','--','python3','-c',pressure],key='visual-pressure-local-ple04',timeout=1260)
    (ROOT/'visual-pressure-local-ple04.receipt.txt').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout[-3500:]+r.stderr[-3500:]
    print(json.dumps({'phase':'24/47-page synthetic visual pressure passed'}),flush=True)
    code=(ROOT/'qualify_protocol_remote.py').read_text().replace('PROJECT',repr(plan['project'])).replace('WORKERS',repr(urls))
    outer=f"""import json,os,subprocess,sys
from pathlib import Path
p=Path({plan['project']!r});os.environ['QWEN38_PROJECT']=str(p);sys.path.insert(0,str(p/'scripts'));from runtime import PYTHON,environment
r=subprocess.run([PYTHON,'-c',{code!r}],env=environment(),capture_output=True,text=True,timeout=1600)
(p/'logs/protocol-qualification-01.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout[-3500:]+r.stderr[-3500:];print(r.stdout)
"""
    r=execute(['cb','alloc',plan['allocation'],'run','--node',plan['router_node'],'--timeout','1800','--','python3','-c',outer],key='protocol-local-ple04',timeout=1860)
    (ROOT/'protocol-qualification-01.receipt.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout[-3500:]+r.stderr[-3500:]
    summary=json_result(r.stdout);(ROOT/'protocol-qualification-01.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'phase':'full benchmark protocol passed','summary':summary}),flush=True)
    ns={'__file__':str(ROOT/'controller.py')}
    exec(compile((ROOT/'controller.py').read_text().split('\ntry:\n    if not plan.get("storage"):')[0],str(ROOT/'controller.py'),'exec'),ns)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:post_checks=list(pool.map(ns['final_process_check'],plan['nodes']))
    (ROOT/'post-protocol-process-checks.json').write_text(json.dumps(post_checks,indent=2)+'\n')
    plan['post_protocol_final_checks']=post_checks
    (ROOT/'task.json').write_text(json.dumps(plan,indent=2)+'\n')
    code=(ROOT/'export_final_remote.py').read_text().replace('PROJECT',repr(plan['project'])).replace('PLAN',repr(plan))
    r=execute(['cb','alloc',plan['allocation'],'run','--node',plan['router_node'],'--timeout','180','--','python3','-c',code],key='export-local-ple04',timeout=240)
    (ROOT/'evidence-export-01.receipt.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout[-2500:]+r.stderr[-2500:]
    manifest=json_result(r.stdout);(ROOT/'evidence-export-01.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'phase':'fully qualified; ready for benchmark handoff','export':manifest}),flush=True)
    (ROOT/'final-pipeline-complete.json').write_text(json.dumps({'passed':True,'created_unix':time.time(),'export':manifest},indent=2)+'\n')
except BaseException as exc:
    (ROOT/'final-pipeline-error.json').write_text(json.dumps({'error':repr(exc),'created_unix':time.time()},indent=2)+'\n')
    raise
