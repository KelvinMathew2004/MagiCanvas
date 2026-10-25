"""Structural invariants under noisy layout proposals; no design/pair IDs."""
from copy import deepcopy
import unittest

from engine.qwen_refine import build_container_tree,project_container_tree,snap_source_axes
from engine.resize import (_calibrate_text_boxes,_layout_font_manager,_render_text_ink,
                           _skia_advances,_set_font_size)


def text(eid,x,y,w=100,h=30,**kwargs):
    return dict(id=eid,type='text',x=float(x),y=float(y),width=float(w),height=float(h),
                content=kwargs.pop('content','A title'),font_family='Solstice Sans',font_size=20.,
                letter_spacing=2.,text_align=kwargs.pop('text_align','center'),z_index=2,**kwargs)


def design(items,w=500,h=500):
    return dict(canvas=dict(width=w,height=h),elements=items)


class ContainerTreeTests(unittest.TestCase):
    def test_full_canvas_background_does_not_join_separate_callouts(self):
        s=design([dict(id='bg',type='image',x=0,y=0,width=500,height=500),
                  text('top',20,20),text('bottom',350,430)])
        tree=build_container_tree(s)
        self.assertNotIn('bg',tree['source_elements'])
        self.assertEqual(sorted(len(n['element_ids']) for n in tree['children']),[1,1])

    def test_noisy_thin_rules_cannot_move_or_resize_a_rigid_lockup(self):
        s=design([text('title',100,150,200),text('body',100,210,200),
                  dict(id='rule',type='image',x=100.,y=195.,width=200.,height=2.,z_index=1)])
        tree=build_container_tree(s);q=deepcopy(s['elements'])
        for e in q:
            e['x']=e['x']*1.2+20;e['y']=e['y']*1.2+5;e['width']*=1.2;e['height']*=1.2
        q[2].update(x=0,y=0,width=450,height=4)
        out,notes=project_container_tree(q,tree,650,650);out={e['id']:e for e in out}
        self.assertEqual(set(notes[0]['anchor_ids']),{'title','body'})
        self.assertAlmostEqual(out['body']['y']-out['title']['y'],72.)
        self.assertAlmostEqual(out['rule']['width'],240.)
        self.assertAlmostEqual(out['rule']['x'],out['title']['x'])
        self.assertAlmostEqual(out['title']['letter_spacing']/out['title']['font_size'],.1)
        again,_=project_container_tree(list(out.values()),tree,650,650)
        for e in again:
            for k in ('x','y','width','height'):self.assertAlmostEqual(e[k],out[e['id']][k])

    def test_backing_tree_preserves_nested_container_membership(self):
        s=design([dict(id='outer',type='image',x=20.,y=20.,width=400.,height=400.,z_index=0),
                  dict(id='inner',type='image',x=60.,y=60.,width=150.,height=100.,z_index=1),
                  text('inside',75,80,100,30),text('caption',260,300,100,30)])
        tree=build_container_tree(s)
        root=next(n for n in tree['children'] if 'outer' in n['element_ids'])
        nested=[n for n in root['children'] if n['kind']=='container']
        self.assertTrue(any({'inner','inside'}<=set(n['element_ids']) for n in nested))

    def test_container_clamps_as_one_unit_and_images_keep_proportions(self):
        s=design([dict(id='frame',type='image',x=30.,y=30.,width=300.,height=300.,z_index=0),
                  text('title',80,100,200,40)])
        q=deepcopy(s['elements']);q[0].update(x=-90.,y=-40.,width=900.,height=900.)
        q[1].update(x=-30.,y=-40.,width=600.,height=120.)
        out,_=project_container_tree(q,build_container_tree(s),320,200)
        a,b=out
        self.assertGreaterEqual(a['x'],0);self.assertGreaterEqual(a['y'],0)
        self.assertLessEqual(a['x']+a['width'],320);self.assertLessEqual(a['y']+a['height'],200)
        self.assertAlmostEqual(a['width']/a['height'],1.)
        self.assertAlmostEqual((b['x']-a['x'])/a['width'],50/300)

    def test_axis_cluster_snaps_actual_median_and_ignores_reflowed_column(self):
        s=design([text('a',100,30,180),text('b',100,160,120),text('c',100,280,100)])
        q=deepcopy(s['elements']);q[0]['x']=102.;q[1]['x']=106.;q[2]['x']=300.
        out,notes=snap_source_axes(q,s,500,500)
        self.assertAlmostEqual(out[0]['x'],104.)
        self.assertAlmostEqual(out[1]['x'],104.)
        self.assertEqual(out[2]['x'],300.)
        self.assertTrue(notes)

    def test_source_near_alignment_does_not_chain_across_many_pixels(self):
        s=design([text('a',100,20,90),text('b',103,150,120),text('c',106,300,150)])
        q=deepcopy(s['elements']);q[0]['x']=101.;q[1]['x']=105.;q[2]['x']=110.
        out,_=snap_source_axes(q,s,500,500)
        self.assertEqual(out[2]['x'],110.)
        self.assertAlmostEqual(out[0]['x'],out[1]['x'])

    def test_rotated_source_and_tracking_survive_projection(self):
        s=design([text('a',100,100,180,rotation=352.),text('b',100,170,180)])
        q=deepcopy(s['elements']);q[0]['rotation']=0;q[1]['y']=250
        out,_=project_container_tree(q,build_container_tree(s),500,500)
        self.assertEqual(out[0]['rotation'],352.)
        self.assertAlmostEqual(out[1]['y']-out[0]['y'],70.)

    def test_snapping_cannot_erase_an_authored_panel_inset(self):
        panel=dict(id='panel',type='shape',x=100.,y=100.,width=200.,height=100.,z_index=0)
        label=text('label',102,130,150,30,text_align='left')
        source=design([panel,label]);before=deepcopy(source)
        out,_=snap_source_axes(deepcopy(source['elements']),source,500,500)
        self.assertGreaterEqual(out[1]['x']-out[0]['x'],2.-1e-6)
        self.assertEqual(source,before)

    def test_source_corner_crop_is_preserved_by_the_whole_container(self):
        corner=dict(id='corner',type='image',x=-20.,y=-10.,width=180.,height=120.,z_index=0)
        label=text('label',20,30,100,30,text_align='left')
        source=design([corner,label]);proposal=deepcopy(source['elements'])
        for e in proposal:
            for key in ('x','y','width','height'):e[key]*=1.5
        out,_=project_container_tree(proposal,build_container_tree(source),750,750)
        self.assertAlmostEqual(out[0]['x']/out[0]['width'],-20/180)
        self.assertAlmostEqual(out[0]['y']/out[0]['height'],-10/120)
        from engine.resize import _authored_image_bleed
        self.assertTrue(_authored_image_bleed(out[0],corner,(500,500),750,750))
        self.assertAlmostEqual(out[1]['x']-out[0]['x'],60.)
        self.assertAlmostEqual(out[1]['y']-out[0]['y'],60.)

    def test_single_source_centered_container_inherits_the_canvas_axis(self):
        s=design([text('a',150,180,200),text('b',150,240,200)])
        q=deepcopy(s['elements'])
        for e in q:e['x']+=25
        out,_=project_container_tree(q,build_container_tree(s),500,500)
        for e in out:self.assertAlmostEqual(e['x']+e['width']/2,250.)

    def test_native_shape_can_stretch_but_image_cannot(self):
        s=design([dict(id='pill',type='shape',x=100.,y=100.,width=200.,height=60.,z_index=0),
                  text('label',125,112,150,30)])
        q=deepcopy(s['elements']);q[0].update(width=300.);q[1].update(width=200.)
        out,notes=project_container_tree(q,build_container_tree(s),600,500)
        self.assertEqual(notes[0]['mode'],'semi-elastic')
        self.assertEqual(out[1]['font_size'],20.)
        self.assertGreater(out[0]['width'],200.)


class RenderCalibrationTests(unittest.TestCase):
    def test_skia_probe_agrees_with_the_converter_widths(self):
        from renderer.converter.wrap import char_widths
        manager=_layout_font_manager()
        if manager is None:self.skipTest('Local fonts unavailable')
        expected=char_widths(manager,'Solstice Sans','regular','regular',19.5,'Wide track')
        self.assertEqual(tuple(expected),_skia_advances('Wide track','Solstice Sans',False,False,19.5))

    def test_widens_a_tight_box_before_shrinking_font(self):
        e=text('title',40,40,20,45,content='WIDE TITLE',text_align='left')
        s=design([deepcopy(e)])
        out,notes=_calibrate_text_boxes([e],s,500,300)
        self.assertGreater(out[0]['width'],20)
        self.assertEqual(out[0]['font_size'],20)
        self.assertTrue(notes[0]['widened'])
        ink=_render_text_ink(out[0]);self.assertIsNotNone(ink)
        self.assertLessEqual(ink[2],out[0]['width']+1)

    def test_font_shrinks_only_when_neighbour_blocks_expansion_and_tracking_scales(self):
        e=text('label',10,30,40,30,content='A LONG TRACKED TITLE',text_align='left')
        blocker=text('other',60,30,110,30,text_align='left')
        out,_=_calibrate_text_boxes([e,blocker],design([deepcopy(e),deepcopy(blocker)]),200,100)
        self.assertLess(out[0]['font_size'],20)
        self.assertAlmostEqual(out[0]['letter_spacing']/out[0]['font_size'],.1)
        self.assertLessEqual(out[0]['x']+out[0]['width'],blocker['x'])

    def test_fractional_wrap_size_and_integer_drawing_both_fit_the_box(self):
        from engine.resize import _text_info
        e=text('cta',20,20,250,40,content='Shop the Summer Line',text_align='left')
        e.update(font_size=20.9998,font_weight='bold',letter_spacing=0.)
        required=_text_info(e,1e9)[1]
        e['width']=required+1.
        self.assertEqual(_text_info(e)[2],1)
        out,_=_calibrate_text_boxes([e],design([deepcopy(e)]),500,300)
        self.assertEqual(out[0]['font_size'],e['font_size'])
        self.assertLessEqual(_render_text_ink(out[0])[2],out[0]['width']+1.)

    def test_font_changes_scale_negative_tracking_as_well(self):
        e=text('tight',0,0);e['letter_spacing']=-2.
        _set_font_size(e,30)
        self.assertEqual(e['letter_spacing'],-3.)


if __name__=='__main__':unittest.main()
