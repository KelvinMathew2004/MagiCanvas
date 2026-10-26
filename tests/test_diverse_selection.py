from copy import deepcopy
from scripts.build_diverse_dataset_viewer import choose,proposal_design


def rows():
    result=[]
    for family in range(7):
        for source in range(3):
            pid=f'f{family}s{source}__to__target'
            result.append(dict(pair_id=pid,family_id=str(family),target_canvas={'width':800 if source%2 else 300,'height':300},
                source_design={'canvas':{'width':500,'height':600},'elements':[{'type':'text'}]*(family+1)},flagged=False))
    return result


def test_diverse_selection_does_not_cherry_pick_on_defects_or_winner():
    original=rows();before,_=choose(original,[],10);changed=deepcopy(original)
    for i,r in enumerate(changed):r.update(flagged=True,geometry_mae=i*999,winner='bad',qwen_family_win=False)
    after,_=choose(changed,[],10)
    assert [r['pair_id'] for r in before]==[r['pair_id'] for r in after]


def test_reviewed_family_excluded_and_at_most_two_unique_sources_per_family():
    from collections import Counter
    picked,coverage=choose(rows(),[{'pair_id':'f0s0__to__other'}],10)
    assert all(r['family_id']!='0' for r in picked)
    assert max(Counter(r['family_id'] for r in picked).values())<=2
    assert len({r['pair_id'].split('__to__')[0] for r in picked})==len(picked)
    assert coverage['reviewed_families_excluded']


def test_raw_viewer_retains_paint_fields_without_repairing_predicted_geometry():
    source={'canvas':{'width':300,'height':250,'background':'#222'},'elements':[
        {'id':'title','type':'text','content':'Raw text','font_family':'Serif','font_size':24,'letter_spacing':3,'x':10,'y':20,'width':200,'height':30},
        {'id':'hero','type':'image','asset_id':'hero.png','fit':'contain','x':0,'y':0,'width':40,'height':50}]}
    frozen=deepcopy(source)
    predictions=[{'id':'title','x':-10,'y':7,'width':20,'height':8,'font_size':40},
                 {'id':'hero','x':5,'y':8,'width':99,'height':12}]
    raw=proposal_design(source,{'width':100,'height':100},predictions)
    assert source==frozen
    assert raw['elements'][0]['content']=='Raw text'
    assert raw['elements'][0]['x']==-10 and raw['elements'][0]['font_size']==40
    assert raw['elements'][0]['letter_spacing']==5
    assert raw['elements'][1]['asset_id']=='hero.png'
    assert raw['elements'][1]['width']==99 and raw['elements'][1]['height']==12
    assert raw['canvas']=={'width':100,'height':100,'background':'#222'}
