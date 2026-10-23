"""Regression checks for model safety, local variants and versioned labels."""
from copy import deepcopy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engine.qwen_refine import font_variant, repair_relationships
from engine.qwen_prompt import encode_v2_completion, decode_v2_completion, messages_for
from engine.qwen_seed import _missing_proposal, _extract_json, _output_budget
from engine.resize import _qwen_contrast_safe, _select_visual_candidate, _text


def text(eid,y=10,**kwargs):
    return dict(id=eid,type='text',content='Hello world',font_family='Solstice Sans',
                font_size=20,x=10,y=y,width=150,height=25,color='rgb(0,0,0)',
                line_height=1.,**kwargs)


def units(items):
    return [dict(items=[e],source_canvas=(500,500)) for e in items]


class QwenPipelineTests(unittest.TestCase):
    def test_relative_contrast_uses_all_alternatives_and_invisible_stays_unsafe(self):
        q=dict(cost=7.5,low_contrast=['a','b'],invisible_text=[])
        alternatives=[dict(cost=1.,low_contrast=['x','y','z']),dict(cost=8.7,low_contrast=['p','q','r'])]
        self.assertTrue(_qwen_contrast_safe(q,alternatives,True))
        q['invisible_text']=['b']
        self.assertFalse(_qwen_contrast_safe(q,alternatives,True))
        self.assertFalse(_qwen_contrast_safe(dict(cost=8.,low_contrast=['a']),[dict(cost=1.,low_contrast=[])]))

    def test_qwen_prior_can_select_a_repaired_sibling(self):
        entries=[dict(candidate=dict(name=n),cost=0.,low_contrast=[])
                 for n in ['split','qwen-proposal','qwen-font-fitted']]
        r=_select_visual_candidate(entries,[1.,9.,9.2],[.8,.4,.9],True,True,False)
        self.assertEqual(r['selected_index'],2)
        self.assertEqual(r['qwen_family_band'],[1,2])
        # A large objective regression is excluded even with a high image score.
        r=_select_visual_candidate(entries,[1.,9.,90.],[.8,.4,1.],True,True,False)
        self.assertEqual(r['selected_index'],1)

    def test_font_calibration_can_grow_without_mutating_original(self):
        original=text('a');proposal=deepcopy(original);proposal['font_size']=10
        before=deepcopy(proposal)
        fitted,changed=font_variant([proposal],units([original]),500,500)
        self.assertEqual(proposal,before)
        self.assertEqual(changed,['a']);self.assertGreater(fitted[0]['font_size'],10)
        self.assertLessEqual(_text(fitted[0])[0],fitted[0]['height'])

    def test_whole_column_solving_cannot_push_into_next_member(self):
        originals=[text('a',10),text('b',50),text('c',90)]
        proposed=deepcopy(originals)
        for e,y in zip(proposed,(20,25,50)):e['y']=y
        before=deepcopy(proposed)
        result,notes=repair_relationships(proposed,units(originals),500,500)
        ordered=sorted(result,key=lambda e:e['id'])
        self.assertEqual(proposed,before)
        for a,b in zip(ordered,ordered[1:]):
            self.assertGreaterEqual(b['y'],a['y']+_text(a)[0]+2.9)
        self.assertTrue(any(n['kind']=='column-chain-solve' for n in notes))

    def test_explicit_lockup_keeps_relative_offsets_and_hierarchy(self):
        a=text('price',group_id='price-lockup');a.update(width=150,height=40,font_size=30)
        b=text('suffix',group_id='price-lockup');b.update(content='99',x=120,width=30,height=15,font_size=12)
        proposed=deepcopy([a,b]);proposed[0].update(x=100,y=100);proposed[1].update(x=270,y=140)
        result,_=repair_relationships(proposed,units([a,b]),500,500)
        scale=result[0]['font_size']/a['font_size']
        self.assertAlmostEqual(result[1]['x']-result[0]['x'],(b['x']-a['x'])*scale)
        self.assertAlmostEqual(result[1]['font_size']/result[0]['font_size'],b['font_size']/a['font_size'])

    def test_missing_superscript_uses_matched_parent_transform(self):
        a=text('price',group_id='g');b=text('suffix',group_id='g');b.update(x=120,width=25,content='99')
        proposals={'price':dict(x=100,y=100,width=300,height=50,font_size=40)}
        out,anchor=_missing_proposal(b,{'price':a,'suffix':b},proposals,b)
        self.assertEqual(anchor,'price');self.assertAlmostEqual(out['x'],320)
        self.assertEqual(out['font_size'],40)

    def test_v2_and_v3_roundtrip_and_negative_alias_rejected(self):
        source=dict(canvas=dict(width=1080,height=1080),elements=[text('a')])
        canvas=dict(width=1600,height=100)
        expected={'elements':[dict(id='a',x=12,y=8,width=700,height=50,font_size=35)]}
        for version in (2,3):
            encoded=encode_v2_completion(expected,source,canvas,version)
            decoded=decode_v2_completion(encoded,source,canvas,version)['elements'][0]
            self.assertAlmostEqual(decoded['font_size'],35,places=2)
            self.assertIn(f'format v{version}',messages_for(source,canvas,version=version)[1]['content'])
        payload={'elements':[dict(id='e-1',box=[0,0,1,1])]}
        self.assertEqual(decode_v2_completion(payload,source,canvas)['elements'],[])

    def test_partial_json_recovers_only_complete_objects(self):
        payload=_extract_json('{"elements":[{"id":"e000","box":[1,2,3,4]},{"id":"e001","box":[1,')
        self.assertEqual(len(payload['elements']),1)

    def test_output_budget_scales_with_json_token_cost_and_context(self):
        class Tokenizer:
            def __call__(self,s,**kwargs):return {'input_ids':list(s)}
        source={'elements':[text(str(i)) for i in range(20)]}
        self.assertGreater(_output_budget(Tokenizer(),source,2,1000),2048)
        self.assertEqual(_output_budget(Tokenizer(),source,2,8000),192)

    def test_inference_does_not_mutate_source_or_make_contain_images_cover(self):
        import torch
        from engine.qwen_seed import generate_seed
        source=dict(canvas=dict(width=500,height=500),elements=[
            dict(id='photo',type='image',asset_id='missing.png',x=0,y=0,width=500,height=250),
            text('copy',300)])
        payload=json.dumps({'elements':[{'id':'e000','box':[0,0,1000,1000]},
            {'id':'e001','box':[100,500,400,200],'font_scale':80}]})
        class Tokenizer:
            eos_token_id=0
            def apply_chat_template(self,*args,**kwargs):return 'prompt'
            def __call__(self,s,**kwargs):
                return {'input_ids':torch.tensor([[1]])} if kwargs.get('return_tensors') else {'input_ids':list(s)}
            def decode(self,*args,**kwargs):return payload
        class Model:
            config=SimpleNamespace(max_position_embeddings=8192)
            def generate(self,**kwargs):return torch.tensor([[1,2,3]])
        before=deepcopy(source);diagnostics={}
        with patch('engine.qwen_seed._load',return_value=(Model(),Tokenizer(),torch,'cpu',2)):
            result,_=generate_seed(source,{'width':800,'height':200},source['elements'],diagnostics)
        self.assertEqual(source,before)
        self.assertAlmostEqual(result[0]['width']/result[0]['height'],2)
        self.assertNotEqual(result[0].get('fit'),'cover')
        self.assertTrue(diagnostics['json_complete'])
        self.assertEqual(diagnostics['missing_ids'],[])
        # A source cover image may crop using a changed target frame.
        source['elements'][0]['fit']='cover'
        with patch('engine.qwen_seed._load',return_value=(Model(),Tokenizer(),torch,'cpu',2)):
            cover,_=generate_seed(source,{'width':800,'height':200},source['elements'])
        self.assertEqual(cover[0]['fit'],'cover')
        self.assertAlmostEqual(cover[0]['width']/cover[0]['height'],4)


if __name__=='__main__':unittest.main()
