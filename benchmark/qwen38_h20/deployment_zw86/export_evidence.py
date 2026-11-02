"""Transfer export inputs as an ordinary file, then submit a short native command."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from native_run import execute, json_result

ROOT = Path(__file__).parent


def run_export(plan, tag):
    bundle = ROOT / f'export-input-{tag}.tar.gz'
    inputs = {
        'plan.json': (json.dumps(plan, indent=2) + '\n').encode(),
        'export.py': (ROOT / 'export_final_remote.py').read_bytes(),
    }
    if bundle.exists():
        with tarfile.open(bundle, 'r:gz') as archive:
            for name, content in inputs.items():
                assert archive.extractfile(name).read() == content, 'Export inputs changed'
    else:
        with tarfile.open(bundle, 'w:gz') as archive:
            for name, content in inputs.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o600
                archive.addfile(member, io.BytesIO(content))
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    remote = str(Path(plan['project']) / 'scripts' / bundle.name)
    argv = ['cb', 'file', 'put', str(bundle), remote,
            '--allocation', plan['allocation'], '--node', plan['router_node'],
            '--timeout-seconds', '1800', '--wait', '--json',
            '--idempotency-key', f'export-input-upload-{tag}']
    record = ROOT / f'export-input-upload-{tag}.argv.json'
    if record.exists():
        assert json.loads(record.read_text()) == argv
    else:
        record.write_text(json.dumps(argv, indent=2) + '\n')
    result = subprocess.run(argv, capture_output=True, text=True, timeout=240)
    (ROOT / f'export-input-upload-{tag}.receipt.txt').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    code = f'''import hashlib,json,tarfile
from pathlib import Path
bundle=Path({remote!r})
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=={digest!r}
with tarfile.open(bundle,'r:gz') as archive:
    plan=json.loads(archive.extractfile('plan.json').read())
    source=archive.extractfile('export.py').read()
assert plan['project']=={plan['project']!r}
snapshot=Path(plan['project'])/'results/final-plan-{tag}.json'
snapshot.write_text(json.dumps(plan,indent=2)+'\\n')
exec(compile(source,'export.py','exec'),{{'PROJECT':plan['project'],'PLAN':plan,'__name__':'__main__'}})
'''
    result = execute(['cb', 'alloc', plan['allocation'], 'run', '--node', plan['router_node'],
                      '--timeout', '600', '--', 'python3', '-c', code],
                     key=f'export-file-{tag}', timeout=645)
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-2500:]
    manifest = json_result(result.stdout)
    (ROOT / 'evidence-export-01.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
