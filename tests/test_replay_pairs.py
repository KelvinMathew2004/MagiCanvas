"""A paired evaluation must not collapse equal-size target-template IDs."""
from scripts.replay_qwen_engine import _prediction_queues


def test_equal_canvas_pairs_retain_their_own_prediction_in_record_order():
    def row(name,x):
        return dict(pair_id=name,source_design={'elements':[{'id':'same_source'}]},target_canvas={'width':1080,'height':1080},qwen_model_elements=[{'id':'same_source','x':x}])
    a,b=row('target_a',10),row('target_b',90)
    queues,by_pair=_prediction_queues([{'pair_id':'target_b'},{'pair_id':'target_a'}],[a,b])
    queue=next(iter(queues.values()))
    assert queue.popleft()['qwen_model_elements'][0]['x']==90
    assert queue.popleft()['qwen_model_elements'][0]['x']==10
    assert by_pair['target_a'] is a


def test_resumed_pair_is_not_consumed_by_the_next_equal_canvas_task():
    a=dict(pair_id='a',source_design={'elements':[{'id':'same'}]},target_canvas={'width':100,'height':100})
    b=dict(a,pair_id='b')
    queues,_=_prediction_queues([{'pair_id':'a'},{'pair_id':'b'}],[a,b],{'a'})
    assert next(iter(queues.values())).popleft()['pair_id']=='b'


def test_recorded_prompt_refusal_replays_without_loading_or_inventing_prediction():
    from pathlib import Path
    from unittest.mock import patch
    from scripts.replay_qwen_engine import replay
    source={'canvas':{'width':100,'height':100},'elements':[{'id':'a','type':'shape','x':1,'y':1,'width':10,'height':10}]}
    row={'pair_id':'dense','source_design':source,'target_canvas':{'width':160,'height':600},
         'qwen_model_elements':[],'qwen_status':'Qwen prompt exceeds the 4096-token safety limit (5000 tokens)'}
    def run(records,source_for,adapter,model,checkpoint):
        from engine import qwen_seed
        elements,note=qwen_seed.generate_seed(source,row['target_canvas'],[],diagnostics={})
        assert elements is None and note==row['qwen_status']
        return [dict(row)]
    with patch('scripts.evaluate_qwen_adapter.run_engine_mode',side_effect=run):
        result=replay([{'pair_id':'dense'}],[row],Path('unused'),Path('unused'),None,source_for=lambda *_:source,allow_incomplete=True)
    assert result[0]['raw_prediction_reused'] and result[0]['qwen_model_elements']==[]
