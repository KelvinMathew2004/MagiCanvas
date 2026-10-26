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
    def test_full_canvas_layers_keep_authored_fit_modes(self):
        from engine.resize import _assemble
        source=dict(canvas=dict(width=500,height=500),elements=[
            dict(id='base',type='image',x=0,y=0,width=500,height=500,z_index=0,fit='cover'),
            dict(id='frame',type='image',x=0,y=0,width=500,height=500,z_index=1,fit='contain'),
            text('label',x=100,y=100,width=300,height=40,z_index=2),
        ])
        backgrounds,analysis_units=_analyze(source)
        self.assertEqual({e['id'] for e in backgrounds},{'base','frame'})
        # A full-canvas asset is a backdrop, not evidence that separate copy
        # was flattened into it. Preserve that label as a foreground unit.
        self.assertEqual([e['id'] for u in analysis_units for e in u['items']],['label'])
        assembled=_assemble(source['elements'],backgrounds,source,600,400)
        by_id={e['id']:e for e in assembled['elements']}
        self.assertEqual(by_id['frame']['fit'],'contain')
        self.assertEqual(by_id['base']['fit'],'cover')

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

    def test_font_calibration_keeps_an_already_readable_size_without_mutating_original(self):
        original=text('a');proposal=deepcopy(original);proposal['font_size']=10
        before=deepcopy(proposal)
        fitted,changed=font_variant([proposal],units([original]),500,500)
        self.assertEqual(proposal,before)
        self.assertEqual(changed,[]);self.assertEqual(fitted[0]['font_size'],10)
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

    def test_raw_font_fitted_candidate_is_not_constraint_projected(self):
        source=dict(canvas=dict(width=500,height=500),elements=[text('first',y=50.),text('second',y=130.)])
        proposal=deepcopy(source['elements']);proposal[1]['y']=proposal[0]['y']
        def inspect(candidates,*args,**kwargs):
            raw=next(c for c in candidates if c['name']=='qwen-pre-container-font-fitted')
            by_id={e['id']:e for e in raw['elements']}
            self.assertAlmostEqual(by_id['first']['y'],by_id['second']['y'])
            self.assertTrue(any(c['name']=='qwen-raw-constraint-repaired' for c in candidates))
            return next(c for c in candidates if not c['name'].startswith('qwen-')),dict(used=False)
        with patch.dict(os.environ,{'MAGICANVAS_QWEN_ENABLED':'1'}), \
             patch('engine.qwen_seed.generate_seed',return_value=(proposal,'mock')), \
             patch('engine.resize._visual_rerank',side_effect=inspect):
            _adapt_with_diagnostics(source,dict(width=500,height=500))

    def test_runtime_projects_broken_art_stack_before_render_ranking(self):
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
            _,analysis_units=_analyze(source)
            modeled=[c for c in candidates if c['name'].startswith('qwen-')]
            self.assertTrue(modeled)
            projected=[c for c in modeled if c.get('structure',{}).get('containers')]
            self.assertTrue(projected)
            for candidate in projected:
                self.assertEqual(_broken_source_stack_units(candidate['elements'],analysis_units),[])
            # A structurally valid raw layout must remain available beside the
            # container projection; the renderer decides whether to preserve it.
            self.assertTrue(any(c['name']=='qwen-pre-container-font-fitted' for c in modeled))
            return projected[0],dict(used=False,selected_index=0)
        with patch.dict(os.environ,{'MAGICANVAS_QWEN_ENABLED':'1'}), \
             patch('engine.qwen_seed.generate_seed',return_value=(proposal,'mock proposal')), \
             patch('engine.resize._visual_rerank',side_effect=inspect_candidates):
            _,report=_adapt_with_diagnostics(source,dict(width=500,height=300))
        self.assertEqual(source,before)
        diagnostics=report['qwen_diagnostics']
        self.assertEqual(diagnostics['structural_pass_version'],1)
        self.assertEqual(set(diagnostics['container_tree']['children'][0]['element_ids']),{'back','front'})
        self.assertTrue(diagnostics['selected_structure']['containers'])

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

    def test_training_filters_pairs_that_change_unpredicted_rotation(self):
        from scripts.prepare_qwen_training import pair_filter_reason
        a=dict(canvas=dict(width=100,height=100),elements=[
            dict(id='a',type='image',x=0,y=0,width=100,height=100,z_index=0,rotation=-90)])
        b=dict(canvas=dict(width=100,height=100),elements=[
            dict(id='b',type='image',x=0,y=0,width=100,height=100,z_index=0,rotation=0)])
        pair={'matches':[{'a_index':0,'b_index':0}]}
        self.assertEqual(pair_filter_reason(pair,a,b),'element_rotation_changed')
        b['elements'][0]['rotation']=270
        self.assertIsNone(pair_filter_reason(pair,a,b))

    def test_centered_single_line_source_text_can_recover_from_proposal_wrap(self):
        from engine.qwen_refine import font_variant
        original=text('headline',content='WIDE HEADLINE',x=150.,y=100.,width=200.,height=25.,
                      font_size=20.,text_align='center')
        proposal=deepcopy(original);proposal.update(x=230.,width=40.,height=50.)
        result,changed=font_variant([proposal],units([original]),500.,500.)
        self.assertIn('headline',changed)
        self.assertEqual(_text(result[0])[2],1)
        self.assertGreater(result[0]['width'],proposal['width'])

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

    def test_rotated_element_uniform_scaling_and_bounds_clamping(self):
        import torch
        from engine.qwen_seed import generate_seed, _rotated_bounds
        source = dict(canvas=dict(width=500, height=500), elements=[
            dict(id='badge', type='image', asset_id='badge.png', x=100., y=100.,
                 width=100., height=100., rotation=45.0)
        ])
        # Propose non-uniform scaling (width=200, height=150)
        payload = json.dumps({'elements': [
            {'id': 'e000', 'box': [0, 0, 400, 300]}  # 200w x 150h on 500x500 canvas
        ]})
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, *args, **kwargs): return 'prompt'
            def __call__(self, s, **kwargs):
                return {'input_ids': torch.tensor([[1]])} if kwargs.get('return_tensors') else {'input_ids': list(s)}
            def decode(self, *args, **kwargs): return payload
        class Model:
            config = SimpleNamespace(max_position_embeddings=8192)
            def generate(self, **kwargs): return torch.tensor([[1, 2, 3]])

        with patch('engine.qwen_seed._load', return_value=(Model(), Tokenizer(), torch, 'cpu', 2)):
            result, _ = generate_seed(source, {'width': 500, 'height': 500}, source['elements'])

        badge = result[0]
        # Must maintain rotation
        self.assertAlmostEqual(badge['rotation'], 45.0)
        # Center-pivot uniform scaling: width and height must scale uniformly
        self.assertAlmostEqual(badge['width'], badge['height'])
        # Rotated bounds must stay inside canvas [0, 500] x [0, 500]
        rb = _rotated_bounds(badge)
        self.assertGreaterEqual(rb[0], -1e-4)
        self.assertGreaterEqual(rb[1], -1e-4)
        self.assertLessEqual(rb[2], 500.0 + 1e-4)
        self.assertLessEqual(rb[3], 500.0 + 1e-4)

    def test_rotated_bleed_backdrop_keeps_source_side_anchor(self):
        import torch
        from engine.qwen_seed import generate_seed
        source=dict(canvas=dict(width=1200,height=628),elements=[
            dict(id='bleed',type='image',asset_id='backdrop.png',fit='contain',
                 x=-447.5,y=-95.9,width=1267.5,height=846.2,z_index=0,rotation=-90.)
        ])
        payload=json.dumps({'elements':[{'id':'e000','box':[0,0,1000,1000]}]})
        class Tokenizer:
            eos_token_id=0
            def apply_chat_template(self,*args,**kwargs):return 'prompt'
            def __call__(self,s,**kwargs):
                return {'input_ids':torch.tensor([[1]])} if kwargs.get('return_tensors') else {'input_ids':list(s)}
            def decode(self,*args,**kwargs):return payload
        class Model:
            config=SimpleNamespace(max_position_embeddings=8192)
            def generate(self,**kwargs):return torch.tensor([[1,2,3]])
        with patch('engine.qwen_seed._load',return_value=(Model(),Tokenizer(),torch,'cpu',2)):
            result,_=generate_seed(source,{'width':940,'height':788},source['elements'])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0]['x'],0.,places=4)
        self.assertEqual(result[0]['rotation'],-90.)

    def test_rotated_lockup_companions_grouping_and_repair(self):
        from engine.qwen_refine import lockup_groups, repair_relationships
        # Tilted frame and text companion with similar rotation
        frame = dict(id='frame', type='shape', x=100., y=100., width=120., height=80., rotation=15.0)
        label = text('label', x=120., y=125., width=80., height=30., rotation=15.0)
        source = [frame, label]
        unit = dict(items=source, bounds=(100., 100., 220., 180.), source_canvas=(500, 500))

        # Check they are grouped as rotated companions
        groups = lockup_groups([unit])
        self.assertIn({'frame', 'label'}, groups)

        # Distort the proposal (e.g. sheared/separated)
        proposal = deepcopy(source)
        proposal[0].update(x=200., y=150., width=150., height=100.)
        proposal[1].update(x=350., y=300., width=100., height=40.)  # sheared away
        repaired, notes = repair_relationships(proposal, [unit], 500, 500)
        by_id = {e['id']: e for e in repaired}

        # Verify uniform group scaling and preserved relative center distance
        scale = by_id['frame']['width'] / frame['width']
        self.assertAlmostEqual(by_id['frame']['height'] / frame['height'], scale, places=2)
        # Frame and label must not shear apart
        src_dx = label['x'] - frame['x']
        src_dy = label['y'] - frame['y']
        tgt_dx = by_id['label']['x'] - by_id['frame']['x']
        tgt_dy = by_id['label']['y'] - by_id['frame']['y']
        self.assertAlmostEqual(tgt_dx, src_dx * scale, places=1)
        self.assertAlmostEqual(tgt_dy, src_dy * scale, places=1)
        self.assertTrue(any(n.get('kind') == 'rigid-source-lockup' for n in notes))

    def test_quiet_zone_2d_placement_moves_off_busy_horizontal_subject(self):
        from PIL import Image, ImageDraw
        # Create an under image: top half is calm white sky, bottom half is busy dark couch with dense edges
        under = Image.new('RGB', (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(under)
        # Busy texture at bottom [200, 400]
        for y in range(200, 400, 4):
            for x in range(0, 400, 4):
                val = 40 if (x + y) % 8 == 0 else 180
                draw.rectangle((x, y, x + 3, y + 3), fill=(val, val, val))

        # Full image has black text rendered on top
        full = under.copy()
        draw_full = ImageDraw.Draw(full)
        # Black text on the busy bottom area
        draw_full.text((100, 280), "BUSY TEXT", fill=(0, 0, 0))

        copy_elem = text('busy_title', x=100., y=280., width=200., height=40.,
                         color='rgb(0,0,0)', font_size=24.)
        unit = dict(items=[copy_elem], bounds=(100., 280., 300., 320.), source_canvas=(400, 400))
        elements = [copy_elem]

        repairs, _ = contrast_variant(elements, [unit], 400, 400, ['busy_title'], under, full,
                                      return_alternatives=True)
        # Should contain a 2D repositioning candidate moving to quiet upper zone
        quiet_repairs = [c for c, changes in repairs
                         if any(ch.get('repair_type') == 'lockup-repositioned-2d' for ch in changes)]
        self.assertTrue(len(quiet_repairs) > 0)
        best_cand, best_changes = next((c, changes) for c, changes in repairs
                                       if any(ch.get('repair_type') == 'lockup-repositioned-2d' for ch in changes))
        # dy should move text up into calm sky (dy < 0)
        change = best_changes[0]
        self.assertLess(change['dy'], 0)
        self.assertEqual(change['placement'], 'quiet-zone-2d')

    def test_layer_stack_anchors_to_largest_visual_element_when_anchor_ids_empty(self):
        # Case 01 pattern: unit with a backing image (96x96) and two text elements
        bg_image = dict(id='img_004', type='image', asset_id='img.png', x=156., y=472.,
                        width=96., height=96., fit='contain', z_index=0)
        title = text('txt_011', x=80., y=580., width=248., height=20., font_size=18., z_index=1)
        subtitle = text('txt_012', x=80., y=605., width=248., height=16., font_size=14., z_index=2)
        source_items = [bg_image, title, subtitle]
        unit = dict(items=source_items, bounds=(80., 472., 328., 621.), preserve_layer_stack=True,
                    source_canvas=(1024, 1024))

        proposal = deepcopy(source_items)
        # Model proposes repositioning the visual anchor at x=400, y=200 with 1.5x scale
        proposal[0].update(x=400., y=200., width=144., height=144.)
        # Model proposes distorted/misaligned text
        proposal[1].update(x=100., y=100., width=500., height=50.)
        proposal[2].update(x=120., y=160., width=500., height=40.)

        preserved, anchor_ids = _preserve_layered_proposals(proposal, [unit], return_anchor_ids=True)
        # Largest visual element must be the anchor ID
        self.assertEqual(anchor_ids, ['img_004'])
        by_id = {e['id']: e for e in preserved}
        # Scaled by 1.5x around anchor
        self.assertAlmostEqual(by_id['img_004']['width'], 144.)
        self.assertAlmostEqual(by_id['img_004']['height'], 144.)
        # Text must be refitted relative to the visual anchor
        factor = 144. / 96.
        src_offset_y = title['y'] - bg_image['y']
        tgt_offset_y = by_id['txt_011']['y'] - by_id['img_004']['y']
        self.assertAlmostEqual(tgt_offset_y, src_offset_y * factor, places=1)

    def test_large_overlapping_hero_does_not_bridge_distant_copy_clusters(self):
        from engine.qwen_refine import build_container_tree
        source=dict(canvas=dict(width=1000,height=1000),elements=[
            dict(id='hero',type='image',x=200.,y=100.,width=600.,height=800.,z_index=0),
            text('left',x=100.,y=300.,width=200.,height=50.,z_index=1),
            text('right',x=700.,y=300.,width=200.,height=50.,z_index=2)])
        tree=build_container_tree(source)
        self.assertFalse(any({'left','right'}<=set(c['element_ids']) for c in tree['children']))

    def test_translucent_decoration_cannot_bridge_separate_foreground_groups(self):
        from engine.qwen_refine import build_container_tree
        source=dict(canvas=dict(width=1000,height=1000),elements=[
            dict(id='decoration',type='shape',x=150.,y=250.,width=700.,height=100.,z_index=0,opacity=.15),
            text('left',x=100.,y=300.,width=200.,height=50.,z_index=1),
            text('right',x=700.,y=300.,width=200.,height=50.,z_index=2)])
        tree=build_container_tree(source)
        self.assertFalse(any({'left','right'}<=set(c['element_ids']) for c in tree['children']))

    def test_small_contained_card_uses_banner_height_and_stays_centered(self):
        from engine.qwen_refine import build_container_tree,project_container_tree
        source=dict(canvas=dict(width=1000,height=1400),elements=[
            dict(id='card',type='shape',x=100.,y=100.,width=600.,height=600.,z_index=0),
            text('label',x=150.,y=300.,width=400.,height=200.,font_size=50.,z_index=1)])
        proposed=deepcopy(source['elements'])
        for e in proposed:
            for k in ('x','y','width','height'):e[k]*=.1
            if e['type']=='text':e['font_size']=10.
        result,_=project_container_tree(proposed,build_container_tree(source),900,300)
        card=next(e for e in result if e['id']=='card')
        self.assertAlmostEqual(card['height'],300*.76)
        self.assertAlmostEqual(card['y']+card['height']/2,150.)
        self.assertTrue(all(0<=e['x'] and e['x']+e['width']<=900 and 0<=e['y'] and e['y']+e['height']<=300 for e in result))

    def test_qwen_early_stopping_criteria_stops_repetition_and_closes_json(self):
        import torch
        from engine.qwen_seed import QwenEarlyStoppingCriteria, generate_seed

        expected = {'e000', 'e001', 'e002'}
        class MockTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return self.current_text

        tok = MockTokenizer()
        criteria = QwenEarlyStoppingCriteria(tok, prompt_len=0, expected_ids=expected)

        # Incomplete text: should not stop
        tok.current_text = '{"elements":[{"id":"e000","box":[0,0,10,10]},{"id":"e001","box":[10,10,20,20]}'
        criteria._counter = 15
        self.assertFalse(criteria(torch.tensor([[1, 2, 3]])))

        # All IDs present and closed with ]}: should stop
        tok.current_text = '{"elements":[{"id":"e000","box":[0,0,10,10]},{"id":"e001","box":[10,10,20,20]},{"id":"e002","box":[20,20,30,30]}]}'
        criteria._counter = 15
        self.assertTrue(criteria(torch.tensor([[1, 2, 3]])))
        self.assertTrue(criteria.stopped_early)

        # Repetition loop: duplicate element begins: should stop
        criteria2 = QwenEarlyStoppingCriteria(tok, prompt_len=0, expected_ids=expected)
        tok.current_text = '{"elements":[{"id":"e000","box":[0,0,10,10]},{"id":"e001","box":[10,10,20,20]},{"id":"e002","box":[20,20,30,30]},{"id":"e001"'
        criteria2._counter = 15
        self.assertTrue(criteria2(torch.tensor([[1, 2, 3]])))
        self.assertTrue(criteria2.stopped_early)
        criteria3=QwenEarlyStoppingCriteria(tok,prompt_len=0,expected_ids=expected)
        tok.current_text='{"elements":[{"id":"e000"},{"id":"e001"},{"id":"e999"}]}'
        criteria3._counter=15
        self.assertFalse(criteria3(torch.tensor([[1,2,3]])))



if __name__=='__main__':unittest.main()
