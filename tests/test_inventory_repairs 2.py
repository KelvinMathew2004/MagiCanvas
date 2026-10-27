"""Regression contracts for source structure, raster safety and extreme reflow."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest.mock import patch
from engine.resize import (_adapt_with_diagnostics,_analyze,_decorations,_source_geometry_safety,
    _skyscraper_seed,_raw_bounds,_unintended_control_overlaps,_calibrate_text_boxes,_render_text_ink)
from engine.qwen_refine import build_container_tree,project_container_tree,_source_callout_groups


def text(eid,x,y,w=180,h=35,**kw):
    return dict(id=eid,type='text',x=x,y=y,width=w,height=h,content='A title',
                font_family='Solstice Sans',font_size=24.,text_align='center',z_index=3,**kw)

class InventoryRepairTests(unittest.TestCase):
    def test_vertical_brand_stack_inherits_canvas_axis_on_a_banner(self):
        from engine.resize import _simple_stack_axis_error
        source=json.loads((Path(__file__).resolve().parents[1]/'input/chameleon_sample_level1_warmup.json').read_text())['source']
        _,units=_analyze(source);items=deepcopy(source['elements'][1:])
        items[0].update(x=720.,y=0.,width=200.,height=80.)
        items[1].update(x=150.,y=120.,width=700.,height=80.)
        items[2].update(x=150.,y=260.,width=700.,height=40.)
        self.assertGreater(_simple_stack_axis_error(items,units,source,1000),0)
        _simple_stack_axis_error(items,units,source,1000,apply=True)
        for e in items:self.assertAlmostEqual(e['x']+e['width']/2,500.)

    def test_inline_brand_reflow_keeps_its_own_axes(self):
        from engine.resize import _simple_stack_axis_error
        source=json.loads((Path(__file__).resolve().parents[1]/'input/chameleon_sample_level1_warmup.json').read_text())['source']
        _,units=_analyze(source);items=deepcopy(source['elements'][1:])
        for e in items:e.update(y=100.,height=80.)
        before=deepcopy(items)
        self.assertEqual(_simple_stack_axis_error(items,units,source,1000,apply=True),0.)
        self.assertEqual(items,before)

    def test_identity_returns_independent_exact_copy_without_inference(self):
        source=dict(canvas=dict(width=500,height=500),elements=[text('a',150,80)])
        with patch('engine.qwen_seed.generate_seed',side_effect=AssertionError('must not infer')):
            result,report=_adapt_with_diagnostics(source,dict(width=500,height=500))
        self.assertEqual(result,source);self.assertEqual(report['winner'],'identity')
        result['elements'][0]['x']=9;self.assertEqual(source['elements'][0]['x'],150)

    def test_aligned_hero_logo_and_copy_do_not_become_one_rigid_poster(self):
        source=dict(canvas=dict(width=1000,height=1000),elements=[
            dict(id='logo',type='image',x=400,y=30,width=200,height=80,z_index=1),
            dict(id='hero',type='image',x=300,y=150,width=400,height=400,z_index=1),
            text('headline',250,580,500,80),text('body',300,700,400,60)])
        tree=build_container_tree(source)
        self.assertFalse(any({'logo','hero','headline','body'}<=set(n['element_ids']) for n in tree['children']))
        self.assertFalse(any({'hero','headline'}<=set(n['element_ids']) for n in tree['children']))

    def test_translucent_decorations_do_not_capture_backed_text(self):
        source=dict(canvas=dict(width=500,height=500),elements=[
            dict(id='circle',type='shape',shape_type='ellipse',x=20,y=20,width=400,height=400,opacity=.15,z_index=0),
            text('title',100,100,200,40)])
        tree=build_container_tree(source)
        self.assertFalse(any(r[0]=='backing' and r[1]=='circle' for r in tree['relationships']))
        self.assertEqual(len(tree['children']),2)

    def test_decoration_builder_does_not_mutate_occupied_or_duplicate_ids(self):
        circle=dict(id='circle',type='shape',shape_type='ellipse',x=0,y=0,width=100,height=100,opacity=.15,z_index=0)
        source=dict(canvas=dict(width=500,height=500),elements=[circle,text('title',100,100)])
        _,units=_analyze(source);occupied=[deepcopy(source['elements'][1])];before=deepcopy(occupied)
        added=_decorations([u for u in units if u['role']=='decoration'],500,500,500,500,occupied)
        self.assertEqual(occupied,before);self.assertEqual([e['id'] for e in added],['circle'])

    def test_circle_aspect_is_preserved_after_extreme_layout(self):
        circle=dict(id='circle',type='shape',shape_type='ellipse',x=30,y=30,width=100,height=100,opacity=.15,z_index=0)
        source=dict(canvas=dict(width=500,height=500),elements=[circle])
        candidate=[dict(circle,width=1200,height=90)]
        out=_source_geometry_safety(candidate,source,_analyze(source)[1],1600,400)
        self.assertEqual(out[0]['width'],out[0]['height'])
        self.assertLessEqual(out[0]['x']+out[0]['width'],1600)

    def test_source_disjoint_controls_detect_backplate_collisions(self):
        a=dict(id='a',type='shape',shape_type='rectangle',x=20,y=20,width=100,height=30,group_id='first')
        b=dict(a,id='b',x=200,group_id='second')
        source=dict(canvas=dict(width=500,height=200),elements=[a,b])
        collided=[a,dict(b,x=90)]
        self.assertEqual(len(_unintended_control_overlaps(collided,source)),1)
        self.assertEqual(_unintended_control_overlaps([a,b],source),[])

    def test_rigid_projection_scales_font_and_tracking_with_frame(self):
        panel=dict(id='panel',type='image',x=100,y=100,width=300,height=150,z_index=0)
        label=text('label',150,150,200,40,letter_spacing=3.)
        source=dict(canvas=dict(width=500,height=500),elements=[panel,label])
        proposal=[dict(panel,width=600,height=300),dict(label,width=400,height=80,font_size=5.)]
        out,notes=project_container_tree(proposal,build_container_tree(source),1000,1000)
        size=next(e['font_size'] for e in out if e['id']=='label')
        self.assertAlmostEqual(size,24*notes[0]['scale'])
        self.assertAlmostEqual(out[1]['letter_spacing']/size,3/24)

    def test_skyscraper_stack_has_all_ids_disjoint_badges_and_raw_bounds(self):
        root=Path(__file__).resolve().parents[1]
        source=json.loads((root/'input/chameleon_sample_level3_advanced.json').read_text())['source']
        _,units=_analyze(source)
        out=_skyscraper_seed(units,160,600,(1080,1350))
        self.assertEqual(len(out),len({e['id'] for e in out}))
        expected={e['id'] for u in units for e in u['items']}
        self.assertEqual({e['id'] for e in out},expected)
        for e in out:
            if e['id'].startswith('ribbon'):continue
            l,t,r,b=_raw_bounds([e]);self.assertGreaterEqual(l,-1e-6);self.assertGreaterEqual(t,-1e-6)
            self.assertLessEqual(r,160+1e-6);self.assertLessEqual(b,600+1e-6)
        badges=sorted((e for e in out if e['id'].startswith('badge') and e['type']=='shape'),key=lambda e:e['y'])
        for a,b in zip(badges,badges[1:]):self.assertLessEqual(a['y']+a['height'],b['y'])

    def test_calibration_prevents_positive_element_surface_overflow(self):
        e=text('a',20,20,20,30);e.update(content='ECO-FRIENDSHIP',text_align='left')
        source=dict(canvas=dict(width=500,height=200),elements=[deepcopy(e)])
        out,_=_calibrate_text_boxes([e],source,500,200)
        ink=_render_text_ink(out[0]);self.assertIsNotNone(ink)
        self.assertLessEqual(ink[2],round(out[0]['width']))
        self.assertLessEqual(ink[3],round(out[0]['height']))

if __name__=='__main__':unittest.main()
