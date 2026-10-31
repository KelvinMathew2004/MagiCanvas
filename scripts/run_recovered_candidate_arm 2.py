#!/usr/bin/env python3
"""Overlap the independent recovered arm of an already-running fresh comparison.

Reconstructs and verifies the original fixed inputs; preserves the parent
evaluator's manifest. Pauses the owned parent before its candidate arm to avoid
two writers. Resumes it after the independently checkpointed arm finishes.
This is throughput orchestration, not a generation-latency benchmark.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from threading import Thread,Event
import argparse,json,os,signal,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def count_rows(path):
    if not path.exists():return 0
    lines=path.read_text().splitlines();rows={}
    for line in lines:
        try:r=json.loads(line)
        except json.JSONDecodeError:continue  # A concurrently appended final line.
        rows[r['pair_id']]=r
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline-pid',type=int,required=True);p.add_argument('--reuse-inference-from',type=Path);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    command=subprocess.check_output(['ps','-p',str(a.baseline_pid),'-o','command='],text=True)
    if 'scripts/evaluate_trained_layouts.py --candidate-adapter models/qwen3-0.6b-latest/final' not in command:
        raise SystemExit('Refuse to control an unrelated process')
    if count_rows(a.run/'baseline.jsonl')>=140:
        raise SystemExit('Too late to overlap safely; leave the original evaluator running')
    for file,digest in manifest['engine'].items():assert sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    assert sha256((ROOT/'scripts/evaluate_trained_layouts.py').read_bytes()).hexdigest()==manifest['evaluator']
    assert sha256((ROOT/'scripts/evaluate_qwen_adapter.py').read_bytes()).hexdigest()==manifest['core_evaluator']
    if a.reuse_inference_from:
        assert sha256((ROOT/'scripts/reuse_adapter_inference.py').read_bytes()).hexdigest()==manifest['inference_reuse_code_sha256']
    adapter=ROOT/'models/qwen3-0.6b-latest/final';model=ROOT/'models/qwen3-0.6b'
    for name,folder in [('audited-0.6b-recovered',adapter)]:
        assert {f.name:sha256(f.read_bytes()).hexdigest() for f in sorted(folder.iterdir())
                if f.suffix in ('.json','.txt','.safetensors')}==manifest['adapters'][name]
    from scripts.evaluate_trained_layouts import verify_recovered
    from scripts.evaluate_qwen_adapter import read_sources,run_engine_mode
    from scripts.refresh_regression_viewer import PACK
    verify_recovered(adapter,model)
    saved_fingerprint=json.loads((PACK/'review/regression_suite/run_manifest.json').read_text())['fingerprint']
    for key in ('files_sha256','model','model_weights','adapter','layout_experiments'):
        assert saved_fingerprint.get(key)==manifest['regression_fingerprint'].get(key),'Regression snapshot changed'
    signature=sha256(json.dumps(saved_fingerprint,sort_keys=True).encode()).hexdigest()[:16]
    original=read_sources();sources={};records=[]
    for r in map(json.loads,(ROOT/'data/qwen_lora_expanded_v2/test.jsonl').read_text().splitlines()):
        r=deepcopy(r);old=r['pair_id'];key='reference:'+old.split('__to__')[0]
        sources[(r['source_split'],key)]=original(r['source_split'],old.split('__to__')[0])
        r['pair_id']='reference:'+old;records.append(r)
    for file in sorted((ROOT/'data/qwen_lora/regression_suite'/signature).glob('[0-9][0-9].json')):
        saved=json.loads(file.read_text());r=deepcopy(saved['record']);source=saved['after']['source_design']
        key='review:'+r['pair_id'].split('__to__')[0];sources[(r['source_split'],key)]=source
        r['pair_id']='review:'+r['pair_id'];records.append(r)
    records.sort(key=lambda r:not r['pair_id'].startswith('review:'))
    digest=sha256(json.dumps(dict(records=records,sources={str(k):v for k,v in sources.items()}),sort_keys=True).encode()).hexdigest()
    assert digest==manifest['inputs'] and len(records)==148,'Inputs differ from the parent evaluation'
    os.environ['MAGICANVAS_REQUIRE_QWEN_ADAPTER']='1';os.environ['MAGICANVAS_QWEN_ENABLED']='1'
    stop=Event();paused=Event()
    def monitor():
        while not stop.wait(1):
            if count_rows(a.run/'baseline.jsonl')>=140:
                try:os.kill(a.baseline_pid,signal.SIGSTOP)
                except ProcessLookupError:return
                paused.set();print('Paused owned baseline process before it can write the candidate arm',flush=True);return
    watcher=Thread(target=monitor,daemon=True);watcher.start()
    provenance=dict(qwen_enabled=True,fresh_inference=True,production_changed=False,
        source_manifest_sha256=sha256((a.run/'manifest.json').read_bytes()).hexdigest(),
        launcher_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_input_digest_verified=True,independent_processes_overlap=True,
        timing_scope='Concurrent local evaluation; arm timing is not a portable or isolated speed benchmark')
    (a.run/'parallel_arm_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    try:
        if a.reuse_inference_from:
            from scripts.reuse_adapter_inference import restore_predictions
            restore_predictions(a.reuse_inference_from,a.run,manifest,'audited-0.6b-recovered',records,sources,adapter,model)
        rows=run_engine_mode(records,lambda split,sid:sources[(split,sid)],adapter,model,a.run/'audited-0.6b-recovered.jsonl')
        if any(r.get('engine_error') for r in rows):raise RuntimeError('Candidate engine errors recorded; parent can retry them')
        print('Independent candidate arm complete:',len(rows),flush=True)
    finally:
        stop.set();watcher.join(timeout=3)
        if paused.is_set():
            try:os.kill(a.baseline_pid,signal.SIGCONT)
            except ProcessLookupError:pass
            print('Resumed original evaluator for remaining baseline/report work',flush=True)


if __name__=='__main__':main()
