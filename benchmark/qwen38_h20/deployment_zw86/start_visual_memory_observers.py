"""Start one bounded CPU memory observer per serving node, with exact receipts."""
import concurrent.futures
import json
import os
from pathlib import Path
from native_run import execute, json_result

root = Path(__file__).parent
plan = json.loads((root / "task.json").read_text())
assert os.environ["CB_SID"] == plan["session"]
source = (root / "visual_memory_observer.py").read_text()

def start(node):
    code = source.replace("PROJECT", repr(plan["project"])).replace("NODE", repr(node["node"]))
    outer = f"""import json,os,subprocess,sys,time
from pathlib import Path
p=Path({plan['project']!r});record=p/'logs'/('visual-memory-'+{node['node']!r}+'-04.process.json')
assert not record.exists(),'Observer already has a launch record; inspect it'
log=p/'logs'/('visual-memory-'+{node['node']!r}+'-04.log')
with log.open('xb') as f: child=subprocess.Popen([sys.executable,'-c',{code!r}],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
result={{'pid':child.pid,'pgid':os.getpgid(child.pid),'start_ticks':int(fields[19]),'node':{node['node']!r},'started_unix':time.time(),'max_seconds':1800,'log':str(log)}}
record.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
"""
    result = execute(["cb", "alloc", plan["allocation"], "run", "--node", node["node"],
                      "--timeout", "60", "--", "python3", "-c", outer],
                     key="visual-memory-04-" + str(plan["nodes"].index(node)), timeout=105)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-2000:]
    return json_result(result.stdout)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    result = list(pool.map(start, plan["nodes"]))
(root / "visual-memory-observers-04.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
