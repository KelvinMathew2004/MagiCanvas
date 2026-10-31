import json
from PIL import Image
from scripts.package_regression_review import repair_packet,replace_note


def test_data_driven_stats_and_featured_non_alternate(tmp_path):
    source=tmp_path/'data/donor_leaf_repair_v2';source.mkdir(parents=True)
    alternate=dict(index='001',pair_id='alternate',donor_flags=['x'],repair_flags=[],best_variant='labels',eligible_vs_incumbent=True,changed=True)
    other=dict(index='999',pair_id='dataset',donor_flags=['x'],repair_flags=[],best_variant='dividers',eligible_vs_incumbent=True,changed=True)
    summary=dict(tasks=7,cost_improved=2,donor_flagged=5,repair_flagged=3,divider_contacts_improved=1,clean_candidates_vs_incumbent=2,alternate_results=[alternate])
    (source/'summary.json').write_text(json.dumps(summary));(source/'selected.json').write_text(json.dumps([alternate,other]))
    for row in (alternate,other):
        for name in ('incumbent','donor',row['best_variant']):Image.new('RGB',(20,20)).save(source/f'{row["index"]}_{name}.png')
    dest=tmp_path/'packet';note=repair_packet(tmp_path,dest);first=(dest/'index.html').read_text()
    repair_packet(tmp_path,dest)
    assert (dest/'index.html').read_text()==first
    assert '7 tasks' in note and '5 to 3' in note
    assert 'dataset' in first and 'Production incumbent' in first
    assert 'not confirmed wins' in note


def test_note_replaced_without_duplicates(tmp_path):
    path=tmp_path/'note.md';path.write_text('Unrelated text.\n')
    replace_note(path,'experiment','Old numbers')
    replace_note(path,'experiment','New numbers')
    assert path.read_text().count('experiment:start')==1
    assert 'Old numbers' not in path.read_text() and 'Unrelated text.' in path.read_text()
