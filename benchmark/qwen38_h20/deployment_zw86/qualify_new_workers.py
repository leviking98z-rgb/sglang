"""Check vision, tools, and the evaluator's output budget on new replicas."""

import json
import subprocess
from pathlib import Path
from native_run import execute, json_result

ROOT = Path(__file__).parent
plan = json.loads((ROOT / "task.json").read_text())
assert plan["status"].startswith("ready;")
urls = [r["url"] for n in plan["nodes"] for r in n["replicas"]]
code = r'''
import base64,concurrent.futures,hashlib,io,json,time,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(PROJECT);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
image=Image.new('RGB',(1024,768),'white');ImageDraw.Draw(image).rectangle((250,180,770,590),fill='red')
buf=io.BytesIO();image.save(buf,format='PNG');png=buf.getvalue()
image_url='data:image/png;base64,'+base64.b64encode(png).decode()
model='Qwen/Qwen3.8-Flash-Next'
def post(url,path,body):
 req=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=180) as response:return json.load(response)
def check(url):
 messages=[{'role':'user','content':[{'type':'text','text':'What color is the rectangle? Reply with only the lower-case color name.'},{'type':'image_url','image_url':{'url':image_url}}]}]
 tokenized=post(url,'/tokenize',{'model':model,'messages':messages,'chat_template_kwargs':{'enable_thinking':False}})
 vision=post(url,'/v1/chat/completions',{'model':model,'messages':messages,'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}})
 content=vision['choices'][0]['message']['content'].strip();assert content.lower().strip('`"\' .\n')=='red',vision
 assert vision['usage']['prompt_tokens']>tokenized['count'],(vision['usage'],tokenized['count'])
 tools=[{'type':'function','function':{'name':'get_weather','description':'Get the weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
 user={'role':'user','content':'Call get_weather with city=Beijing.'}
 forced=post(url,'/v1/chat/completions',{'model':model,'messages':[user],'tools':tools,'tool_choice':{'type':'function','function':{'name':'get_weather'}},'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False}})
 calls=forced['choices'][0]['message'].get('tool_calls',[]);assert len(calls)==1 and calls[0]['function']['name']=='get_weather',forced
 assert json.loads(calls[0]['function']['arguments'])['city']=='Beijing',calls
 history=[user,{'role':'assistant','content':None,'tool_calls':calls},{'role':'tool','tool_call_id':calls[0]['id'],'content':'{"city":"Beijing","weather":"sunny"}'},{'role':'user','content':'According to the tool result, what is the city? Reply with only its English name.'}]
 continued=post(url,'/v1/chat/completions',{'model':model,'messages':history,'tools':tools,'tool_choice':'none','temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}})
 answer=continued['choices'][0]['message']['content'].strip();assert answer.strip('`"\' .\n')=='Beijing',continued
 budget=post(url,'/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':'Reply with exactly: SGLANG_OK'}],'temperature':0,'max_tokens':65536,'chat_template_kwargs':{'enable_thinking':False}})
 assert budget['choices'][0]['message']['content'].strip()=='SGLANG_OK',budget
 return {'url':url,'passed':True,'vision':{'content':content,'usage':vision['usage'],'template_token_count':tokenized['count'],'fixture_sha256':hashlib.sha256(png).hexdigest(),'fixture_size':[1024,768]},'forced_tool':{'calls':calls,'usage':forced['usage']},'tool_history':{'content':answer,'usage':continued['usage']},'output_budget':{'requested_max_tokens':65536,'usage':budget['usage'],'finish_reason':budget['choices'][0]['finish_reason'],'scope':'65,536 output budget accepted for a short prompt; actual65K generation not tested'}}
with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as pool:results=list(pool.map(check,URLS))
j={'created_unix':time.time(),'workers':results,'passed':all(x['passed'] for x in results),'scope':'New replicas individually verified for image, forced tool, tool history and short prompt with65K output budget; near196K prompt qualification remains separate'}
(p/'results/new-worker-vision-tools-budget.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
'''.replace("PROJECT", repr(plan["project"])).replace("URLS", repr(urls))
compile(code, "remote-qualification", "exec")
r = execute([
    "cb", "alloc", plan["allocation"], "run", "--node", plan["router_node"],
    "--timeout", "300", "--", "python3", "-c", code,
], key="functional-local-ple04", timeout=345)
(ROOT / "new-worker-qualification-01.receipt.txt").write_text(r.stdout + r.stderr)
if r.returncode:
    raise RuntimeError(r.stdout[-4000:] + r.stderr[-4000:])
result = json_result(r.stdout)
(ROOT / "new-worker-qualification-01.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
