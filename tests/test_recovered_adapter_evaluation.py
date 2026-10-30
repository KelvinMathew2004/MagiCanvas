"""Recovered artifacts are evaluable without fabricated training metadata."""
from hashlib import sha256
import json
import pytest
from scripts.evaluate_trained_layouts import verify_recovered,promotion


def recovered(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.verify_qwen_adapter.verify',lambda *a:dict(prompt_version=2))
    (tmp_path/'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path='Qwen/Qwen3-0.6B')))
    (tmp_path/'adapter_model.safetensors').write_bytes(b'fixture-checked-by-mocked-base-verifier')
    files={n:sha256((tmp_path/n).read_bytes()).hexdigest() for n in ('adapter_config.json','adapter_model.safetensors')}
    (tmp_path/'recovery_manifest.json').write_text(json.dumps(dict(files_sha256=files,global_step=210,max_steps=210,completed_epochs=3,
        provenance_limit='Original environment and exact training-bundle identity unavailable')))
    return tmp_path


def test_recovered_package_records_missing_original_provenance(tmp_path,monkeypatch):
    adapter=recovered(tmp_path,monkeypatch)
    report=verify_recovered(adapter,tmp_path)
    assert report['original_training_manifest_present'] is False
    assert 'unavailable' in report['recovery']['provenance_limit']
    assert not (tmp_path/'training_manifest.json').exists()


def test_recovered_weight_change_fails_closed(tmp_path,monkeypatch):
    adapter=recovered(tmp_path,monkeypatch)
    (adapter/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):verify_recovered(adapter,tmp_path)


def test_incomplete_checkpoint_is_not_called_finished(tmp_path,monkeypatch):
    adapter=recovered(tmp_path,monkeypatch);path=adapter/'recovery_manifest.json'
    r=json.loads(path.read_text());r['global_step']=140;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='completed'):verify_recovered(adapter,tmp_path)


def test_promotion_rejects_regression_in_dense_completion():
    def row(id,valid=True):return dict(pair_id=id,source_design=dict(elements=[{}]*20),
        raw_proposal_valid=valid,raw_geometry_mae=.03,raw_relative_font_mae=.1,geometry_mae=.03,font_mae=.1,
        exact_ids=True,valid=True,flagged=False,defect_flags=[],qwen_family_win=True)
    baseline=[row('reference:a'),row('review:b')];candidate=[row('reference:a'),row('review:b',False)]
    report=promotion(baseline,candidate)
    assert not report['eligible_for_human_review']
    assert not report['checks']['raw_dense_previously_valid_preserved']


def test_paired_reporting_separates_identity_and_common_valid_raw_metrics():
    from scripts.report_recovered_adapter import pair_stats
    def row(valid,size=100):return dict(source_design=dict(canvas=dict(width=100,height=100)),target_canvas=dict(width=size,height=100),
        raw_proposal_valid=valid,raw_json_complete=valid,raw_geometry_mae=.1,raw_relative_font_mae=.2,
        geometry_mae=.04,font_mae=.2,flagged=False,defect_flags=[],qwen_family_win=False)
    before={'identity':row(False),'complete':row(True,200),'new-failure':row(True,200)}
    after={'identity':row(False),'complete':row(True,200),'new-failure':row(False,200)}
    after['complete']['raw_geometry_mae']=.05
    report=pair_stats(before,after,list(before))
    assert report['generation_tasks']==2 and report['before_raw_ids_valid']==2 and report['after_raw_ids_valid']==1
    assert report['raw_geometry_mae']['common_pairs']==1
    assert report['raw_geometry_mae']['after']==pytest.approx(.05)


def test_family_bootstrap_counts_pairs_inside_families():
    from scripts.report_recovered_adapter import family_interval
    before={'a':dict(family_id='one',geometry_mae=.1),'b':dict(family_id='one',geometry_mae=.2),
            'c':dict(family_id='two',geometry_mae=.3)}
    after={k:dict(r,geometry_mae=r['geometry_mae']-.02) for k,r in before.items()}
    report=family_interval(before,after,list(before),'geometry_mae',repeats=100)
    assert report['families']==2 and report['pairs']==3
    assert report['mean_delta']==pytest.approx(-.02)
    assert report['interval95']==pytest.approx([-.02,-.02])


def test_raw_metrics_without_a_resized_human_target_are_not_credited():
    from scripts.report_recovered_adapter import pair_stats
    row=dict(source_design=dict(canvas=dict(width=100,height=100)),target_canvas=dict(width=200,height=100),
        raw_proposal_valid=True,raw_json_complete=True,geometry_mae=None,font_mae=None,
        raw_geometry_mae=.3,raw_relative_font_mae=.4)
    report=pair_stats({'authored':row},{'authored':row},['authored'])
    assert report['before_raw_ids_valid']==1 and report['after_raw_ids_valid']==1
    assert report['raw_geometry_mae']['common_pairs']==0
    assert report['raw_relative_font_mae']['after'] is None


def test_progress_ignores_only_an_unfinished_append(tmp_path):
    from scripts.report_recovered_adapter import load
    p=tmp_path/'arm.jsonl';p.write_text('{"pair_id":"complete"}\n{"pair_id":')
    assert list(load(p,allow_incomplete_tail=True))==['complete']
    with pytest.raises(json.JSONDecodeError):load(p)
    p.write_text('{"pair_id":"complete"}\nBROKEN\n')
    with pytest.raises(json.JSONDecodeError):load(p,allow_incomplete_tail=True)


def test_missing_completed_pointer_cannot_package_partial_results(tmp_path):
    from scripts.package_adapter_review import completed_run
    assert completed_run(tmp_path/'missing.json') is None


def test_packaging_rejects_tampered_completed_report(tmp_path):
    from scripts.package_adapter_review import completed_run
    run=tmp_path/'run';run.mkdir()
    (run/'manifest.json').write_text('{}')
    (run/'paired_analysis.json').write_text('{}')
    pointer=tmp_path/'completed.json'
    pointer.write_text(json.dumps(dict(run=str(run),
        manifest_sha256=sha256((run/'manifest.json').read_bytes()).hexdigest(),
        analysis_sha256=sha256((run/'paired_analysis.json').read_bytes()).hexdigest())))
    (run/'paired_analysis.json').write_text('{"modified":true}')
    with pytest.raises(ValueError,match='changed'):completed_run(pointer)


def test_prediction_reuse_requires_identical_generation_contract_and_weights():
    from copy import deepcopy
    from scripts.reuse_adapter_inference import verify_compatible,GENERATION_FILES
    old=dict(inputs='input',assets={},adapters={'new':{'weights':'same'}},models={'new':{'base':'same'}},
        runtime_environment={},engine={name:'same' for name in GENERATION_FILES})
    new=deepcopy(old);new['engine']['engine/resize.py']='bounds-correction'
    verify_compatible(old,new,'new')
    new['engine']['engine/qwen_prompt.py']='changed'
    with pytest.raises(ValueError,match='generation contract'):verify_compatible(old,new,'new')
    new=deepcopy(old);new['adapters']['new']['weights']='different'
    with pytest.raises(ValueError,match='adapters'):verify_compatible(old,new,'new')


def test_replay_preserves_original_incomplete_inventory_and_separates_timing(tmp_path):
    from scripts.reuse_adapter_inference import original_metadata
    original=dict(generation=dict(missing_ids=['caption'],generation_seconds=5.),raw_proposal_valid=False,
        raw_json_complete=True,qwen_status='literal original outcome')
    replayed=dict(generation=dict(missing_ids=[],generation_seconds=.001),raw_proposal_valid=True,
        raw_json_complete=True,runtime=3.,qwen_status='current realization outcome',output_design={'canvas':{'width':100,'height':100}})
    result=original_metadata(replayed,original,tmp_path)
    assert result['raw_proposal_valid'] is False and result['generation']['missing_ids']==['caption']
    assert result['runtime'] is None and result['engine_replay_seconds']==3.
    assert result['generation']['generation_seconds']==5.
    assert result['output_design']==replayed['output_design']
    assert result['qwen_status']=='current realization outcome'
    assert result['original_qwen_status']=='literal original outcome'
    assert replayed['raw_proposal_valid'] is True


def test_completed_viewer_does_not_invent_font_error_for_nontext_targets():
    from scripts.enhance_adapter_viewer import metric_text,cohort
    assert metric_text(dict(geometry_mae=None),dict(geometry_mae=None))==''
    text=metric_text(dict(geometry_mae=.1,font_mae=None),dict(geometry_mae=.05,font_mae=None))
    assert 'unavailable' in text and '0.05000' in text
    assert cohort('reference:pair',dict(official_path='reference-path'))=='reference'
    assert cohort('review:pair',dict(official_path='official-path'))=='official'
    assert cohort('review:supplied:u02__to__skyscraper',dict(official_path='authored-file'))=='supplied'


def test_raw_stage_audit_keeps_missing_ids_and_does_not_fit_geometry():
    from scripts.audit_adapter_defect_stages import raw_design
    source=dict(canvas=dict(width=100,height=100),elements=[
        dict(id='text',type='text',font_size=20,letter_spacing=4,x=10,y=10,width=80,height=30),
        dict(id='omitted',type='shape',x=0,y=0,width=50,height=50)])
    predicted=dict(id='text',font_size=10,x=160,y=-5,width=80,height=20)
    result=raw_design(dict(source_design=source,target_canvas=dict(width=200,height=100),qwen_model_elements=[predicted]))
    assert [e['id'] for e in result['elements']]==['text']
    e=result['elements'][0]
    assert e['x']==160 and e['y']==-5 and e['width']==80 and e['height']==20
    assert e['font_size']==10 and e['letter_spacing']==2
    assert source['elements'][0]['font_size']==20


def test_stage_categories_do_not_call_missing_inventory_out_of_bounds():
    from scripts.audit_adapter_defect_stages import category_counts
    rows=[dict(raw_flags=['element_ids','structural_check'],
        raw_details=dict(structural_errors=['element id mismatch — missing caption'])),
        dict(raw_flags=['structural_check','glyph_overflow','text_on_text'],
        raw_details=dict(structural_errors=["element 'a' is off-canvas (100x100)"]))]
    result=category_counts(rows,'raw')
    assert result['out_of_bounds']==1 and result['text_fit']==1 and result['overlap']==1
    assert result['stretch']==0 and result['misalignment']==0
