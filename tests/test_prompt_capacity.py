"""Dense prompt capacity retains leaf aliases and trained relation tuples."""
from copy import deepcopy
import json
from engine.qwen_prompt import messages_for
from engine.qwen_seed import _prompt_capacity,_output_budget


def source(n=24):
    return {'canvas':{'width':2000,'height':500},'elements':[
        {'id':str(i),'type':'text','x':i*70,'y':100,'width':50,'height':30,
         'content':'Item '+str(i),'font_family':'Solstice Sans','font_size':20}
        for i in range(n)]}


def relations(prompt):
    return json.loads(prompt.split('Source relationships [kind, element IDs, optional layer order]:\n')[1].split('\nReflow')[0])


def test_compaction_keeps_pair_schema_all_aliases_and_late_alignment_chain():
    s=source();before=deepcopy(s)
    old=messages_for(s,{'width':160,'height':600},version=2)[1]['content']
    new=messages_for(s,{'width':160,'height':600},version=2,relation_token_counter=len,relation_token_budget=900)[1]['content']
    a=old.split('Source relationships')[0];b=new.split('Source relationships')[0]
    assert a==b and s==before
    rs=relations(new)
    assert len(json.dumps(rs,separators=(',',':')))<=900
    assert len(rs)==23
    assert all(len(r)==3 and r[0]=='same-row' for r in rs)
    assert any('e023' in r for r in rs)
    assert messages_for(s,{'width':160,'height':600},version=2)[1]['content']==old


def test_compaction_prefers_nearest_parent_and_excludes_full_canvas_paint():
    s=source(1);s['elements'][0].update(x=100,y=120,z_index=3)
    s['elements']=[dict(id='bg',type='shape',x=0,y=0,width=2000,height=500,z_index=0),
                   dict(id='outer',type='shape',x=50,y=80,width=400,height=300,z_index=1),
                   dict(id='inner',type='shape',x=90,y=110,width=100,height=80,z_index=2),*s['elements']]
    rs=relations(messages_for(s,{'width':160,'height':600},version=2,relation_token_counter=len,relation_token_budget=900)[1]['content'])
    backing=[r for r in rs if r[0]=='backing' and r[2]=='e003']
    assert backing==[['backing','e002','e003']]
    assert all('e000' not in r for r in rs)


def test_context_budget_reserves_inventory_sized_output_and_never_exceeds_context():
    class Tokenizer:
        def __call__(self,text,**kwargs):return {'input_ids':[0]*(len(text)//4)}
    tok=Tokenizer();s=source(38)
    limit,reserve=_prompt_capacity(tok,s,2,8192)
    assert reserve>=64*38+128 and limit>4096 and limit+reserve<=8192
    assert limit+_output_budget(tok,s,2,limit,8192)<=8192
    small,_=_prompt_capacity(tok,s,2,1024)
    assert small==0


def test_zero_area_omission_recovers_without_disabling_visual_checks():
    from engine.proposal_completion import complete_missing
    from unittest.mock import patch
    s=source(9);s['elements'][-1]['width']=0
    proposals={e['id']:{k:e[k] for k in ('x','y','width','height')} for e in s['elements'][:-1]}
    before=deepcopy(s)
    with patch('engine.source_structure.groups',return_value={'backings':[],'members':[]}):
        result,notes=complete_missing(s,proposals,[],{'width':2000,'height':500})
    assert s==before and result is not None and set(result)=={e['id'] for e in s['elements']}
    assert result['8']['width']==1 and notes['8']['method']=='free-rectangle'
