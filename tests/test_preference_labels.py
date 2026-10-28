"""Human approval must refer to the exact images and current decision."""
import hashlib
import json
from scripts.train_siglip_ranker import reviewed_pairs, family_splits, family_keys, readiness


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


def test_declared_ancestry_cannot_leak_through_disconnected_sampled_edges():
    a=dict(pair_id='a',source_id='s1',target_id='t1',family_id='shared-family')
    b=dict(pair_id='b',source_id='s2',target_id='t2',family_id='shared-family')
    items=[(a,None,None),(b,None,None)]
    assert len(set(family_keys(items).values()))==1
    assert len(set(family_splits(items,with_test=True).values()))==1
    assert family_splits(items,with_test=True)==family_splits(list(reversed(items)),with_test=True)


def test_hard_ineligible_preference_and_consistency_repeats_do_not_train(tmp_path):
    row=pair(tmp_path);approvals=tmp_path/'labels.jsonl'
    row.update(a_training_eligible=True,b_training_eligible=False)
    approvals.write_text(json.dumps(dict(pair_id='x',decision='prefer_candidate',review_hash=row['review_hash']))+'\n')
    assert reviewed_pairs([row],approvals)==[]
    approvals.write_text(json.dumps(dict(pair_id='x',decision='accept',review_hash=row['review_hash']))+'\n')
    assert len(reviewed_pairs([row],approvals))==1
    row['status']='reliability_check'
    assert reviewed_pairs([row],approvals)==[]


def test_readiness_uses_independent_test_families_not_pair_volume():
    rows=[(dict(pair_id=str(i),source_id='one-source',target_id=str(i),family_id='one-family'),None,None) for i in range(100)]
    result=readiness(rows)
    assert result['eligible_pairs']==100
    assert sum(result['family_counts'].values())==1
    assert not result['ready']


def test_export_side_mapping_and_hash_are_both_verified(tmp_path):
    from scripts.import_preference_labels import validate
    import pytest
    row=pair(tmp_path);row['display_swapped']=True
    choice=dict(pair_id='x',decision='prefer_candidate',display_choice='a',review_hash=row['review_hash'])
    assert validate([row],[choice])['x']==choice
    with pytest.raises(ValueError,match='side mapping'):
        validate([row],[dict(choice,decision='accept')])


def test_conflicting_repeated_images_quarantine_original_without_fabricating_a_vote(tmp_path):
    row=pair(tmp_path);repeat=dict(row,pair_id='repeat',status='reliability_check',reliability_repeat_of='x')
    path=tmp_path/'labels.jsonl'
    first=dict(pair_id='x',decision='accept',review_hash=row['review_hash'])
    second=dict(first,pair_id='repeat',decision='prefer_candidate')
    path.write_text(''.join(json.dumps(c)+'\n' for c in (first,second)))
    assert reviewed_pairs([row,repeat],path)==[]
    second['decision']='accept'
    path.write_text(''.join(json.dumps(c)+'\n' for c in (first,second)))
    assert len(reviewed_pairs([row,repeat],path))==1
