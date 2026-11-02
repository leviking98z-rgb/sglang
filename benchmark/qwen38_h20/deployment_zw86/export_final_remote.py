import concurrent.futures,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
p=Path(PROJECT)
plan=PLAN
checks=[]
for node in plan['nodes']:
 result=json.loads((p/'results'/('runtime-cache-'+node['node']+'.json')).read_text());assert result['passed'];checks.append(result)
for node in plan['nodes']:
 cache=json.loads((p/'results'/('model-cache-'+node['node']+'.json')).read_text());assert cache['passed'] and cache['phase']=='published';checks.append(cache)
 source=json.loads((p/'results'/('source-deployment-'+node['node']+'.json')).read_text());assert source['passed'] and source['engine_commit']==plan['software_commit']
 ple=json.loads((p/'results'/('ple-source02-n'+str(plan['nodes'].index(node))+'.json')).read_text());assert ple['passed'] and len(ple['ranks'])==8
 for rank in ple['ranks']:
  for name in ['srt/models/qwen4_exp.py','srt/models/qwen4_exp_ple_table.py']:
   assert rank['source_sha256'][name]==plan['source_manifest']['source_sha256'][name],(rank,name)
runtime_sha=hashlib.sha256((p/'scripts/runtime.py').read_bytes()).hexdigest()
qualification={'passed':True,'created_unix':time.time(),'engine_commit':plan['software_commit'],'deployment_scripts_commit':plan['deployment_scripts_commit'],'model_revision':'de4b8e4d43b917e7706784d8bb445c9af86a3540','software_difference_from_bj87':['Default GPU JPEG decode explicitly produces RGB channels','Opt-in contiguous private PLE host storage with 256MiB bounded cudaHostRegister'],'ple_registration_chunk_mb':256,'numa_policy':'interleave=all','runtime_difference':'Same dependency versions; source and full131-shard checkpoints use verified node-local caches','runtime_sha256':runtime_sha,'cache_manifests':checks,'context_boundary':{'input':196604,'output_budget':65536,'mtp_reserved':4,'total':262144},'evidence':['asset-fingerprints.json','jpeg-regression-tests.json','new-worker-functional.json','new-worker-32k.json','combined-router-validation.json','new-worker-vision-tools-budget.json','vision-thinking-qualification.json','context-boundary-qualification.json','deployment-source-02.json',*[f'ple-source02-n{i}.json' for i in range(4)]]}
for filename in qualification['evidence']:assert (p/'results'/filename).is_file(),filename
pressure=json.loads((p/'results/visual-pressure-qualification.json').read_text());assert pressure['passed'];qualification['visual_pressure']={k:pressure[k] for k in ['source_dimensions','client_concurrency','results','scope']};qualification['evidence'].append('visual-pressure-qualification.json')
qualification['validation_scripts_commit']=plan['validation_scripts_commit']
for node in plan['nodes']:
 identity_name='runtime-identity-'+node['node']+'-04.json'
 identity=json.loads((p/'results'/identity_name).read_text())
 assert identity['passed'] and identity['engine_commit']==plan['software_commit']
 qualification['evidence'].append(identity_name)
 memory_name='visual-memory-'+node['node']+('-10.json' if plan.get('qualification_round')=='image10' else '-04.json')
 memory=json.loads((p/'results'/memory_name).read_text())
 assert memory['sample_count']>0 and memory['reason']=='visual qualification result published'
 qualification['evidence'].append(memory_name)
qualification['actual_runtime_identity_evidence']='runtime-identity-<NodeId>-04.json supersedes earlier shared-staging source fields in asset-fingerprints.json'
qualification['software_difference_from_bj87'].append('Opt-in four-image GPU transforms with CPU feature concatenation, exact original outputs')
qualification['image_preprocess_chunk_size']=4
qualification['image_chunk_qualification']=json.loads((p/'results/image-chunks-full-qualification-09.json').read_text())
assert [fixture['sha256'] for fixture in pressure['source_fixtures']]==qualification['image_chunk_qualification']['fixed_fixture_sha256']
assert all(case['passed'] for case in qualification['image_chunk_qualification']['cases'])
qualification['evidence'].append('image-chunks-full-qualification-09.json')
(p/'results/final-qualification-summary.json').write_text(json.dumps(qualification,indent=2)+'\n')
archive=p/'results/deployment-evidence-32h20-01.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for f in sorted((p/'results').glob('*.json')):t.add(f,arcname=f.name)
manifest={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'base_url':'http://'+plan['router_ip']+':30000/v1','created_unix':time.time()}
(p/'results/evidence-export-01.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
