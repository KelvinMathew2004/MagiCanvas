"""Reuse literal decoded predictions after an engine-only correction.

Old final layouts are never imported. Every saved prediction is replayed through
the current engine. Original completion diagnostics survive reconstruction.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json,os


GENERATION_FILES=('engine/qwen_seed.py','engine/qwen_prompt.py','engine/runtime.py',
                  'engine/source_structure.py','engine/proposal_completion.py')


def verify_compatible(old,new,arm):
    for key in ('inputs','assets'):
        if old[key]!=new[key]:raise ValueError('Cannot reuse predictions: '+key+' differ')
    for key in ('adapters','models'):
        if old[key][arm]!=new[key][arm]:raise ValueError('Cannot reuse predictions: '+key+' differ')
    if old.get('runtime_environment')!=new.get('runtime_environment'):
        raise ValueError('Cannot reuse predictions: runtime environment differs')
    for name in GENERATION_FILES:
        if name not in old['engine'] or old['engine'].get(name)!=new['engine'].get(name):
            raise ValueError('Cannot reuse predictions: generation contract changed: '+name)


def original_metadata(row,old,origin):
    row=deepcopy(row)
    for key in ('generation','raw_proposal_valid','raw_json_complete'):
        if key in old:row[key]=deepcopy(old[key])
    row['original_qwen_status']=old.get('qwen_status')
    row['raw_prediction_reused']=True
    row['fresh_prediction_origin']=str(origin.resolve())
    row['engine_replay_seconds']=row.pop('runtime')
    row['runtime']=None  # Do not add a previous generation to a new render time.
    return row


def restore_predictions(origin,dest,new_manifest,arm,records,sources,adapter,model):
    from scripts.report_recovered_adapter import load
    from scripts.replay_qwen_engine import replay
    old_manifest=json.loads((origin/'manifest.json').read_text())
    verify_compatible(old_manifest,new_manifest,arm)
    original=load(origin/(arm+'.jsonl'))
    already=load(dest/(arm+'.jsonl'));restored=0
    provenance=dest/(arm+'.inference_reuse.json')
    note=dict(origin=str(origin.resolve()),origin_manifest_sha256=sha256((origin/'manifest.json').read_bytes()).hexdigest(),
        origin_records_sha256=sha256((origin/(arm+'.jsonl')).read_bytes()).hexdigest(),
        generation_contract_unchanged=True,final_outputs_reused=False,restored_pairs=[])
    for record in records:
        key=record['pair_id'];old=original.get(key)
        if key in already or old is None or old.get('engine_error'):continue
        # Explicitly recorded identity/no-proposal outcomes are replayable too;
        # omitted model IDs are never credited as newly correct predictions.
        row=replay([record],[old],adapter,model,None,source_for=lambda split,sid:deepcopy(sources[(split,sid)]),allow_incomplete=True)[0]
        if row.get('engine_error'):
            print('Replay failed; leave for normal fresh retry:',key,flush=True);continue
        row=original_metadata(row,old,origin)
        with (dest/(arm+'.jsonl')).open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        restored+=1;note['restored_pairs'].append(key)
        provenance.write_text(json.dumps(note,indent=2)+'\n')
        print('Re-realized saved prediction',arm,restored,key,flush=True)
    return restored
