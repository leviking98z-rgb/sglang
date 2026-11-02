"""Qualify frozen benchmark protocol and full context admission on every worker."""
import base64,concurrent.futures,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw
from transformers import AutoTokenizer
P=Path(PROJECT)
URLS=WORKERS
MODEL='Qwen/Qwen3.8-Flash-Next'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(url,path,body,timeout=600):
 req=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=timeout) as response:return json.load(response)
def base(messages,**kwargs):
 return {'model':MODEL,'messages':messages,'temperature':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False},**kwargs}
fixtures=[]
for mode,color,fmt,size in [('RGB',(220,20,20),'JPEG',(630,525)),('L',128,'JPEG',(630,525)),('L',128,'JPEG',(630,630)),('L',128,'JPEG',(640,640)),('L',128,'JPEG',(512,640)),('RGB',(220,20,20),'PNG',(630,525))]:
 img=Image.new(mode,size,color);buf=io.BytesIO();img.save(buf,format=fmt);raw=buf.getvalue()
 fixtures.append({'mode':mode,'format':fmt,'source_size':list(size),'sha256':hashlib.sha256(raw).hexdigest(),'uri':'data:image/'+fmt.lower()+';base64,'+base64.b64encode(raw).decode()})
def qualify(url):
 vision=[]
 for fixture in fixtures:
  messages=[{'role':'user','content':[{'type':'text','text':'Describe the color of this image in one short sentence.'},{'type':'image_url','image_url':{'url':fixture['uri']}}]}]
  count=post(url,'/tokenize',{'model':MODEL,'messages':messages,'chat_template_kwargs':{'enable_thinking':False}})['count']
  result=post(url,'/v1/chat/completions',base(messages,max_tokens=65536 if fixture['mode']=='L' else 128));answer=result['choices'][0]['message'].get('content')
  assert answer and result['usage']['prompt_tokens']>count,result
  assert ('gray' in answer.lower() or 'grey' in answer.lower()) if fixture['mode']=='L' else 'red' in answer.lower(),result
  vision.append({k:v for k,v in fixture.items() if k!='uri'}|{'content':answer,'usage':result['usage'],'template_count':count,'usage_counter':'chat_usage_1'})
 thinking=post(url,'/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':'What is 17 times 19? Give the integer as your final answer.'}],'temperature':0,'max_tokens':1024})
 message=thinking['choices'][0]['message'];assert '323' in message.get('content','') and message.get('reasoning_content'),thinking
 return {'url':url,'vision':vision,'thinking':{'message':message,'usage':thinking.get('usage')},'passed':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as pool:features=list(pool.map(qualify,URLS))
(P/'results/vision-thinking-qualification.json').write_text(json.dumps({'workers':features,'passed':True,'time':time.time()},indent=2)+'\n')
# Count the complete chat template; the visual expansion count is qualified separately.
N=196604
n=N-64
def messages(n):return [{'role':'user','content':'Read the following filler silently.\n'+' x'*n+'\nReply with exactly: SGLANG_OK'}]
for _ in range(8):
 request={'model':MODEL,'messages':messages(n),'chat_template_kwargs':{'enable_thinking':False}}
 counted=post(URLS[0],'/tokenize',request)
 delta=N-counted['count']
 if delta==0:break
 n+=delta
else:raise AssertionError('Unable to create exact context boundary')
assert counted['count']==N
payload=base(messages(n),max_tokens=65536)
prompt_sha=hashlib.sha256(json.dumps(payload['messages'],ensure_ascii=False).encode()).hexdigest()
def boundary(url):
 started=time.monotonic();result=post(url,'/v1/chat/completions',payload,timeout=1200)
 usage=result['usage'];assert usage['prompt_tokens']==N,result
 answer=result['choices'][0]['message']['content'].strip();assert answer=='SGLANG_OK',result
 # One extra requested completion token exceeds the context after MTP reservation.
 rejected=False;error=None
 try:post(url,'/v1/chat/completions',base(payload['messages'],max_tokens=65537))
 except urllib.error.HTTPError as e:
  error=e.read().decode();rejected=e.code==400 and '262144' in error and ('262145' in error or 'maximum context length' in error)
 assert rejected,(url,error)
 return {'url':url,'passed':True,'prompt_tokens':N,'requested_max_tokens':65536,'mtp_reserved_tokens':4,'context_len':262144,'content':answer,'usage':usage,'finish_reason':result['choices'][0]['finish_reason'],'seconds':time.monotonic()-started,'overflow_rejected':True,'overflow_error':error,'prompt_sha256':prompt_sha,'scope':'Full prompt admitted and generated a short answer; not a65536-token generation test'}
with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as pool:boundaries=list(pool.map(boundary,URLS))
result={'passed':True,'workers':boundaries,'created_unix':time.time()}
(P/'results/context-boundary-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'features_passed':len(features),'context_boundaries_passed':len(boundaries),'passed':True,'result_paths':['results/vision-thinking-qualification.json','results/context-boundary-qualification.json']}))
