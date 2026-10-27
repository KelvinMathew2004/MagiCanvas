"""Human approval must refer to the exact images and current decision."""
import hashlib
import json
from scripts.train_siglip_ranker import reviewed_pairs, family_splits


def pair(tmp_path):
    a,b=tmp_path/'a.png',tmp_path/'b.png'
    a.write_bytes(b'image A');b.write_bytes(b'image B')
    return dict(pair_id='x',source_id='source',target_id='target',status='unreviewed_engine_candidate',positive_images=[str(a)],engine_image=str(b),review_hash=hashlib.sha256(a.read_bytes()+b.read_bytes()).hexdigest())


def test_preference_bound_to_pixels_and_human_choice(tmp_path):
    row=pair(tmp_path);approvals=tmp_path/'labels.jsonl'
    assert reviewed_pairs([row],approvals)==[]
    approvals.write_text(json.dumps(dict(pair_id='x',decision='prefer_candidate',review_hash=row['review_hash']))+'\n')
    result=reviewed_pairs([row],approvals)
    assert len(result)==1 and result[0][1:]==(row['engine_image'],row['positive_images'][0])
    (tmp_path/'a.png').write_bytes(b'render changed after approval')
    assert reviewed_pairs([row],approvals)==[]


def test_latest_skip_revokes_earlier_approval(tmp_path):
    row=pair(tmp_path);approvals=tmp_path/'labels.jsonl'
    approvals.write_text(''.join(json.dumps(dict(pair_id='x',decision=d,review_hash=row['review_hash']))+'\n' for d in ('accept','skip')))
    assert reviewed_pairs([row],approvals)==[]


def test_fidelity_failure_and_connected_family_split(tmp_path):
    row=pair(tmp_path);approvals=tmp_path/'labels.jsonl'
    approvals.write_text(json.dumps(dict(pair_id='x',decision='accept',review_hash=row['review_hash']))+'\n')
    row['status']='review_only_fidelity_failure'
    assert reviewed_pairs([row],approvals)==[]
    a=dict(pair_id='a',source_id='s1',target_id='shared')
    b=dict(pair_id='b',source_id='shared',target_id='t2')
    splits=family_splits([(a,None,None),(b,None,None)])
    assert splits['a']==splits['b']
