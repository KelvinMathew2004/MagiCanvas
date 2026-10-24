"""Regression checks for model safety, local variants and versioned labels."""
from copy import deepcopy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engine.qwen_refine import (anchor_superscripts, contrast_variant,
                                font_variant, repair_relationships)
from engine.qwen_prompt import encode_v2_completion, decode_v2_completion, messages_for
from engine.qwen_seed import _missing_proposal, _extract_json, _output_budget
from engine.resize import (_broken_source_stack_units, _preserve_layered_proposals,
                           _adapt_with_diagnostics, _analyze, _qwen_contrast_safe,
                           _select_visual_candidate, _siglip_rerank, _text)


def text(eid,y=10,**kwargs):
    result=dict(id=eid,type='text',content='Hello world',font_family='Solstice Sans',
                font_size=20,x=10,y=y,width=150,height=25,color='rgb(0,0,0)',
                line_height=1.)
    result.update(kwargs)
    return result


def units(items):
    return [dict(items=[e],source_canvas=(500,500)) for e in items]


class QwenPipelineTests(unittest.TestCase):
    def test_source_backdrop_overlap_does_not_trigger_hero_clearance(self):
        from engine.qwen_refine import _repair_hero_clearance
        dome=dict(id='dome',type='image',x=100.,y=60.,width=300.,height=360.,z_index=0)
        title=text('title',x=150.,y=100.,width=200.,height=40.,z_index=1)
        source=[dome,title];proposal=deepcopy(source)
        result,notes=_repair_hero_clearance(proposal,units(source),500,500,[])
        self.assertEqual(result,proposal)
        self.assertEqual(notes,[])

    def test_top_header_does_not_prevent_a_side_callout_clearing_hero(self):
        from engine.qwen_refine import _repair_hero_clearance
        hero=dict(id='hero',type='image',x=150.,y=150.,width=200.,height=350.)
        header=text('header',x=140.,y=20.,width=220.,height=35.)
        callout=text('callout',x=30.,y=250.,width=100.,height=40.)
        source=[hero,header,callout];proposal=deepcopy(source);proposal[2]['x']=130.
        result,notes=_repair_hero_clearance(proposal,units(source),500,500,[])
        by_id={e['id']:e for e in result}
        self.assertLessEqual(by_id['callout']['x']+by_id['callout']['width'],hero['x']-18.)
        self.assertEqual(by_id['header']['x'],header['x'])
        self.assertTrue(any(n['kind']=='hero-column-clearance' for n in notes))

    def test_side_callout_can_shrink_as_a_group_into_a_narrow_hero_column(self):
        from engine.qwen_refine import _repair_hero_clearance
        hero=dict(id='hero',type='image',x=150.,y=150.,width=200.,height=350.)
        callout=text('callout',x=30.,y=250.,width=100.,height=40.)
        proposal=deepcopy([hero,callout]);proposal[0].update(x=130.,width=240.);proposal[1]['x']=100.
        result,notes=_repair_hero_clearance(proposal,units([hero,callout]),500,500,[])
        self.assertLess(result[1]['width'],100.)
        self.assertLessEqual(result[1]['x']+result[1]['width'],112.+1e-5)
        self.assertTrue(any(n.get('scale',1.)<1. for n in notes))

    def test_transparent_hero_margin_is_not_evidence_of_intentional_overlap(self):
        from PIL import Image,ImageDraw
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from engine.qwen_refine import _repair_hero_clearance
        with TemporaryDirectory() as directory:
            asset=Path(directory)/'hero.png'
            image=Image.new('RGBA',(100,100),(0,0,0,0))
            paint=ImageDraw.Draw(image)
            paint.rectangle((50,0,99,99),fill='red');paint.rectangle((0,0,49,3),fill='red')
            image.save(asset)
            hero=dict(id='hero',type='image',asset_id=str(asset),fit='contain',
                      x=150.,y=150.,width=200.,height=350.)
            callout=text('callout',x=100.,y=270.,width=80.,height=30.)
            proposal=deepcopy([hero,callout]);proposal[1]['x']=140.
            result,notes=_repair_hero_clearance(proposal,units([hero,callout]),500,500,[])
            self.assertLess(result[1]['x'],proposal[1]['x'])
            self.assertTrue(any(n['kind']=='hero-column-clearance' for n in notes))

    def test_centered_emblem_keeps_separated_icons_and_copy_in_one_lockup(self):
        from engine.qwen_refine import lockup_groups,source_emblem_error
        source=[text('price',x=180.,y=80.,width=140.,height=60.,font_size=48.,content='$49'),
                text('caption',x=180.,y=210.,width=140.,height=30.,font_size=18.),
                dict(id='left',type='shape',x=160.,y=155.,width=25.,height=45.),
                dict(id='right',type='shape',x=315.,y=155.,width=25.,height=45.),
                dict(id='rule',type='shape',x=190.,y=260.,width=120.,height=2.)]
        # Local proximity links connect the complete emblem through the caption.
        source[1]['y']=200.;source[1]['height']=40.
        groups=lockup_groups(units(source))
        ids={e['id'] for e in source}
        self.assertIn(ids,groups)
        proposed=deepcopy(source)
        proposed[0]['x']+=55.;proposed[1]['x']-=40.;proposed[3]['y']+=20.
        result,notes=repair_relationships(proposed,units(source),500,500)
        by_id={e['id']:e for e in result};scale=by_id['left']['width']/25.
        self.assertAlmostEqual(by_id['right']['x']-by_id['left']['x'],155.*scale)
        self.assertAlmostEqual(by_id['rule']['y']-by_id['left']['y'],105.*scale)
        self.assertTrue(any(set(n.get('element_ids',[]))==ids for n in notes))
        self.assertLess(source_emblem_error(result,units(source)),.025)
        broken=deepcopy(result)
        next(e for e in broken if e['id']=='rule')['y']+=100.
        self.assertGreater(source_emblem_error(broken,units(source)),.10)

    def test_ranker_prefers_safe_complete_emblem_over_detached_ornament(self):
        entries=[dict(candidate=dict(name='qwen-original'),cost=0.,low_contrast=[],
                      source_lockup_error=.25,has_layout_defects=False),
                 dict(candidate=dict(name='qwen-proposal'),cost=0.,low_contrast=[],
                      source_lockup_error=.01,has_layout_defects=False)]
        result=_select_visual_candidate(entries,[0.,0.],[.99,.4],True,True)
        self.assertEqual(result['selected_index'],1)
        self.assertEqual(result['source_lockup_rejected_names'],['qwen-original'])

    def test_bottom_hero_preserves_height_share_and_bottom_anchor(self):
        from engine.qwen_refine import _repair_edge_heroes
        hero=dict(id='hero',type='image',x=150.,y=100.,width=200.,height=400.,fit='contain')
        proposal=[dict(hero,x=220.,y=220.,width=100.,height=200.)]
        result,notes=_repair_edge_heroes(proposal,units([hero]),600,600)
        repaired=result[0]
        self.assertAlmostEqual(repaired['y']+repaired['height'],600.)
        self.assertAlmostEqual(repaired['x']+repaired['width']/2,300.)
        self.assertGreaterEqual(repaired['height'],.85*400./500.*600.)
        self.assertAlmostEqual(repaired['width']/repaired['height'],.5)
        self.assertTrue(notes)

    def test_clean_base_proposal_is_not_replaced_by_contrast_only_sibling(self):
        entries=[dict(candidate=dict(name='qwen-proposal'),cost=.5,low_contrast=[],layout_defect_cost=.5,
                      has_layout_defects=False),
                 dict(candidate=dict(name='qwen-original'),cost=4.,low_contrast=['label'],layout_defect_cost=0.),
                 dict(candidate=dict(name='qwen-original-contrast-repaired-01'),cost=.6,
                      low_contrast=[],layout_defect_cost=0.)]
        result=_select_visual_candidate(entries,[.5,4.,.6],[.4,.2,.99],True,True)
        self.assertEqual(result['selected_index'],0)
        self.assertIn('qwen-original-contrast-repaired-01',result['contrast_repair_suppressed_names'])
        # A tiny threshold miss is still reported, but must not trigger a
        # destructive relocation of an otherwise readable authored badge.
        entries[0]['candidate']['elements']=[dict(id='label',font_size=12.)]
        entries[0]['low_contrast']=['label']
        entries[0]['contrast_by_text']={'label':dict(minimum=4.1,lower_quintile=4.36)}
        result=_select_visual_candidate(entries,[.5,4.,.6],[.4,.2,.99],True,True)
        self.assertEqual(result['selected_index'],0)

    def test_baked_photo_border_repair_moves_copy_and_button_together(self):
        from PIL import Image, ImageDraw
        from engine.qwen_refine import _background_column_variants,lockup_groups
        headline=text('headline',x=40.,y=80.,width=140.,height=60.,font_size=24.)
        pill=dict(id='pill',type='shape',x=40.,y=180.,width=130.,height=30.,z_index=0)
        button=text('button',x=48.,y=185.,width=110.,height=20.,font_size=12.,z_index=1)
        source=[headline,pill,button];proposal=deepcopy(source)
        for e in proposal:e['x']+=60.
        source_image=Image.new('RGB',(500,500),'white');target_image=source_image.copy()
        ImageDraw.Draw(source_image).rectangle((0,0,250,499),fill='#df497f')
        ImageDraw.Draw(target_image).rectangle((0,0,150,499),fill='#df497f')
        with patch('engine.resize._feasible',return_value=True), \
             patch('engine.resize._unintended_text_overlaps',return_value=[]):
            candidates=_background_column_variants(proposal,units(source),500,500,
                ['headline'],target_image,source_image,lockup_groups(units(source)))
        self.assertTrue(candidates)
        fixed,changes=candidates[0];by_id={e['id']:e for e in fixed}
        self.assertEqual(set(changes[0]['element_ids']),{'headline','pill','button'})
        self.assertLessEqual(max(e['x']+e['width'] for e in fixed),146.)
        self.assertAlmostEqual(by_id['button']['x']-by_id['pill']['x'],8.)
        self.assertEqual(proposal[0]['x'],100.)

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

    def test_strong_qwen_is_not_penalized_for_heuristic_display_font_shrink(self):
        def entry(name,cost,size,busy):
            return dict(candidate=dict(name=name,elements=[
                dict(id='title',type='text',x=10.,y=10.,width=80.,height=30.,font_size=size),
                dict(id='body',type='text',x=10.,y=50.,width=80.,height=12.,font_size=10.),
            ]),cost=cost,low_contrast=[],invisible_text=[],busy_background=busy,
                layout_defect_cost=0.,contrast_by_text={
                    'title':dict(lower_quintile=5.1),'body':dict(lower_quintile=5.4)})
        entries=[entry('visual-first',1.,20.,[]),
                 entry('qwen-proposal',4.,48.,['title'])]
        report=_select_visual_candidate(entries,[1.,4.],[.4,.9],
            {'qwen-proposal':True},False,False,display_text_ids={'title'})
        self.assertEqual(report['selected_index'],1)
        self.assertEqual(report['busy_background_tolerated_names'],['qwen-proposal'])
        self.assertIn(1,report['tie_band'])
        self.assertTrue(report['qwen_safety_checks']['busy_background_tolerated'])

    def test_siglip_rerank_uses_display_font_tolerance(self):
        def entry(name,cost,size,busy):
            return dict(candidate=dict(name=name,elements=[
                dict(id='title',type='text',content='Title',x=10.,y=10.,
                     width=80.,height=30.,font_size=size),
            ]),cost=cost,layout_defect_cost=0.,low_contrast=[],invisible_text=[],
                busy_background=busy,contrast_by_text={'title':dict(lower_quintile=5.2)},
                image=None)
        entries=[entry('visual-first',1.,20.,[]),
                 entry('qwen-proposal',4.,48.,['title'])]
        with patch('engine.vision._load_ranker',return_value=None), \
             patch('engine.vision.score_layouts',return_value=([.4,.9],'test scores')):
            report=_siglip_rerank(entries,lambda row:row['cost'],
                qwen_feasible={'qwen-proposal':True},display_text_ids={'title'})
        self.assertEqual(report['selected_index'],1)
        self.assertTrue(report['qwen_safety_checks']['busy_background_tolerated'])

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

    def test_broken_source_art_stack_gets_preserved_sibling_geometry(self):
        source=[
            dict(id='back',type='shape',shape_type='rectangle',x=0,y=0,width=100,height=100,
                 z_index=0,fill_color='#204060'),
            dict(id='front',type='shape',shape_type='ellipse',x=50,y=40,width=100,height=100,
                 z_index=1,fill_color='#f06040'),
        ]
        proposed=[deepcopy(source[0]),deepcopy(source[1])]
        proposed[0].update(x=200,y=100,width=120,height=120)
        proposed[1].update(x=400,y=100,width=120,height=120)
        unit=dict(items=source,bounds=(0,0,150,140),preserve_layer_stack=True,
                  source_canvas=(500,500))
        before=deepcopy(proposed)

        broken=_broken_source_stack_units(proposed,[unit])
        self.assertEqual(broken,[unit])
        repaired=_preserve_layered_proposals(proposed,broken)
        self.assertEqual(proposed,before)
        self.assertEqual(_broken_source_stack_units(repaired,[unit]),[])
        repaired_by_id={e['id']:e for e in repaired}
        self.assertAlmostEqual(repaired_by_id['front']['x']-repaired_by_id['back']['x'],
                               (source[1]['x']-source[0]['x'])*
                               repaired_by_id['back']['width']/source[0]['width'])

    def test_layer_stack_uses_uniform_scale_and_refits_tracked_text(self):
        photo=dict(id='photo',type='image',x=0.,y=0.,width=100.,height=100.,
                   fit='contain',z_index=0)
        label=text('label',content='AB',x=20.,width=60.,height=24.,font_size=24.,
                   letter_spacing=4.,z_index=1)
        source=[photo,label]
        proposal=deepcopy(source)
        proposal[0].update(x=200.,y=100.,width=200.,height=50.)
        proposal[1].update(x=220.,y=110.,width=120.,height=12.,font_size=24.)
        unit=dict(items=source,bounds=(0.,0.,100.,100.),preserve_layer_stack=True,
                  source_canvas=(500,500))
        result=_preserve_layered_proposals(proposal,[unit])
        by_id={e['id']:e for e in result}
        self.assertEqual(by_id['photo']['fit'],'contain')
        self.assertAlmostEqual(by_id['photo']['width']/by_id['photo']['height'],1.)
        self.assertAlmostEqual(by_id['label']['letter_spacing']/label['letter_spacing'],
                               by_id['label']['font_size']/label['font_size'])
        self.assertLessEqual(_text(by_id['label'])[0],by_id['label']['height']+1e-5)

    def test_superscript_frames_have_clearance_after_anchoring(self):
        anchor=text('price',content='$49',x=10.,width=110.,height=60.,font_size=48.)
        satellite=text('cents',content='99',x=100.,width=45.,height=24.,font_size=16.)
        originals=[anchor,satellite]
        proposal=deepcopy(originals)
        proposal[0].update(x=120.,width=320.,height=80.)
        proposal[1].update(x=130.,width=220.,height=30.)
        before=deepcopy(originals)
        result,changed=anchor_superscripts(proposal,units(originals))
        by_id={e['id']:e for e in result}
        self.assertEqual(originals,before)
        self.assertIn('price',changed);self.assertIn('cents',changed)
        self.assertGreaterEqual(by_id['cents']['x'],
                                by_id['price']['x']+by_id['price']['width']+2.-1e-5)

    def test_cover_strip_repair_expands_to_back_all_lockup_text(self):
        from PIL import Image, ImageDraw
        strip=dict(id='panel',type='image',asset_id='panel.png',fit='cover',
                   x=10.,y=30.,width=56.,height=20.,z_index=0)
        label=text('headline',content='TITLE',x=8.,y=30.,width=60.,height=20.,
                   font_size=12.,color='#ffffff',z_index=1)
        source=[strip,label]
        unit=dict(items=source,source_canvas=(100,100),bounds=(8.,30.,68.,50.),
                  role='copy',semantic='headline')
        under=Image.new('RGB',(100,100),(0,0,0));full=under.copy()
        ImageDraw.Draw(full).line((12,36,34,36),fill=(255,255,255),width=2)
        with patch('engine.resize._feasible',return_value=True), \
             patch('engine.resize._unintended_text_overlaps',return_value=[]):
            candidates,_=contrast_variant(source,[unit],100,100,['headline'],under,full,
                                          return_alternatives=True)
        expanded=[(candidate,changes) for candidate,changes in candidates
                  if changes and changes[0].get('repair_type')=='backing-strip-expanded']
        self.assertTrue(expanded)
        candidate,changes=expanded[0];by_id={e['id']:e for e in candidate}
        backed=by_id['panel'];padding=max(4.,1.)
        self.assertLessEqual(backed['x'],label['x']-padding+1e-5)
        self.assertGreaterEqual(backed['x']+backed['width'],
                                label['x']+label['width']+padding-1e-5)
        self.assertEqual(backed['fit'],'cover')

    def test_intact_source_art_stack_does_not_trigger_preservation_candidate(self):
        source=[
            dict(id='back',type='shape',shape_type='rectangle',x=0,y=0,width=100,height=100,
                 z_index=0,fill_color='#204060'),
            dict(id='front',type='shape',shape_type='ellipse',x=50,y=40,width=100,height=100,
                 z_index=1,fill_color='#f06040'),
        ]
        proposal=[deepcopy(source[0]),deepcopy(source[1])]
        for element in proposal:
            element['x']+=120;element['y']+=80
        unit=dict(items=source,bounds=(0,0,150,140),preserve_layer_stack=True,
                  source_canvas=(500,500))
        self.assertEqual(_broken_source_stack_units(proposal,[unit]),[])

    def test_runtime_adds_layer_candidate_only_after_qwen_breaks_stack(self):
        source=dict(canvas=dict(width=500,height=500),elements=[
            dict(id='back',type='shape',shape_type='rectangle',x=50,y=50,width=100,height=100,
                 z_index=0,fill_color='#204060'),
            dict(id='front',type='shape',shape_type='ellipse',x=100,y=100,width=100,height=100,
                 z_index=1,fill_color='#f06040'),
        ])
        proposal=deepcopy(source['elements'])
        proposal[0].update(x=20,y=20)
        proposal[1].update(x=300,y=20)
        before=deepcopy(source)
        def inspect_candidates(candidates,*args,**kwargs):
            names={candidate['name'] for candidate in candidates}
            self.assertIn('qwen-layer-preserved',names)
            return candidates[0],dict(used=False,selected_index=0)
        with patch.dict(os.environ,{'MAGICANVAS_QWEN_ENABLED':'1'}), \
             patch('engine.qwen_seed.generate_seed',return_value=(proposal,'mock proposal')), \
             patch('engine.resize._visual_rerank',side_effect=inspect_candidates):
            _,report=_adapt_with_diagnostics(source,dict(width=500,height=300))
        self.assertEqual(source,before)
        diagnostics=report['qwen_diagnostics']
        self.assertEqual(diagnostics['layer_stack_repaired_ids'],['back','front'])
        _,analysis_units=_analyze(source)
        self.assertEqual(_broken_source_stack_units(
            diagnostics['layer_preserved_elements'],analysis_units),[])

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
