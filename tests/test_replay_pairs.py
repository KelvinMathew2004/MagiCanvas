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
