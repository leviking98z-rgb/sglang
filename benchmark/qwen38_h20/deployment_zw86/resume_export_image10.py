"""Resume only evidence export after already successful final qualifications."""
import json
import os
import time
from pathlib import Path

from export_evidence import run_export

root = Path(__file__).parent
plan = json.loads((root / 'task.json').read_text())
assert os.environ['CB_SID'] == plan['session']
assert plan['qualification_round'] == 'image10'
assert not plan['fully_qualified']
protocol = json.loads((root / 'protocol-qualification-01.json').read_text())
assert protocol['passed'] and protocol['features_passed'] == 8 and protocol['context_boundaries_passed'] == 8
assert len(plan['post_protocol_final_checks']) == 4
manifest = run_export(plan, 'image10')
plan.update(fully_qualified=True, status='ready;32H20 fully qualified; awaiting benchmark handoff',
            completed_unix=time.time())
(root / 'task.json').write_text(json.dumps(plan, indent=2) + '\n')
service = json.loads((root / 'service.json').read_text())
service.update(status='running; fully qualified', software_commit=plan['software_commit'],
               validation_scripts_commit=plan['validation_scripts_commit'],
               export_orchestration_commit=plan['export_orchestration_commit'],
               new_allocation_owner=plan['session'], validation=plan['combined_validation'],
               image_preprocess_chunk_size=4, evidence_export=manifest)
(root / 'service.json').write_text(json.dumps(service, indent=2) + '\n')
complete = {'passed': True, 'created_unix': time.time(), 'export': manifest}
(root / 'final-pipeline-complete.json').write_text(json.dumps(complete, indent=2) + '\n')
print(json.dumps(complete), flush=True)
