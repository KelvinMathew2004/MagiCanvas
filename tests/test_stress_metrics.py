"""Safety gates distinguish rendered defects from template/style preferences."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from PIL import Image
from engine.resize import _analyze,_objective_layout_cost,_select_visual_candidate,_render_text_ink,_calibrate_text_boxes,_unintended_text_overlaps,_assemble
from engine.qwen_refine import whole_lockup_contrast_variants
from scripts.evaluate_qwen_adapter import summarize_engine,measure
from scripts.mine_qwen_batch_defects import _hard_failure


def title():
    return dict(id='title',type='text',x=30.,y=50.,width=140.,height=40.,font_size=24.,
                content='Readable title',font_family='Solstice Sans',text_align='left',
                letter_spacing=0.,line_height=1.2,color='#ffffff',z_index=2)


class StressMetricsTests(unittest.TestCase):
    def test_mining_keeps_inherited_contrast_separate_from_defects(self):
        self.assertFalse(_hard_failure(dict(defect_flags=[],review_flags=[],low_contrast_count=2)))
        self.assertTrue(_hard_failure(dict(defect_flags=['contrast_below_source_floor'],review_flags=[])))

    def test_mining_geometry_regression_remains_a_review_trigger(self):
        self.assertTrue(_hard_failure(dict(defect_flags=[],review_flags=['geometry_regression_vs_qwen_off'])))

    def test_intersecting_envelopes_without_painted_collision_are_safe(self):
        a=title();a.update(content='A',font_size=40.,width=80.,height=60.)
        b=deepcopy(a);b.update(id='b',x=a['x']+20,y=a['y']+15)
        units=[{'items':[a,dict(b,x=450.)]}]
        self.assertEqual(_unintended_text_overlaps([a,b],units),[])
        b.update(x=a['x'],y=a['y'])
        self.assertEqual(_unintended_text_overlaps([a,b],units),[('b','title')])

    def test_authored_font_overhang_is_not_new_clipping(self):
        text=title();source={'canvas':{'width':600,'height':400},'elements':[text]}
        def ink(e):return (2.,-e['font_size']*.2,e['width']-2.,e['height']-2.)
        with patch('engine.resize._render_text_ink',side_effect=ink):
            fitted,_=_calibrate_text_boxes([text],source,600,400)
            self.assertEqual(fitted[0]['font_size'],text['font_size'])
            self.assertEqual(_objective_layout_cost([text],_analyze(source)[1],source,600,400,1.),0.)

    def test_right_aligned_anchor_is_moved_before_font_shrinking(self):
        text=title();text.update(x=180.,y=60.,width=140.,height=40.,text_align='right')
        backing=dict(id='back',type='shape',shape_type='rectangle',x=160.,y=40.,
                     width=200.,height=100.,z_index=0)
        source={'canvas':{'width':500,'height':200},'elements':[backing,text]}
        moved=deepcopy(text);moved['x']=230.
        result,notes=_calibrate_text_boxes([backing,moved],source,500,200)
        fitted=result[1]
        self.assertEqual(fitted['font_size'],text['font_size'])
        self.assertLessEqual(fitted['x']+fitted['width'],360.)
        self.assertTrue(notes)

    def test_objective_does_not_reward_a_template_position(self):
        text=title();ink=_render_text_ink(text)
        if ink:text['height']=max(text['height'],ink[3]+2)
        source={'canvas':{'width':600,'height':400},'elements':[text]}
        units=_analyze(source)[1];moved=deepcopy(text);moved.update(x=350.,y=260.)
        self.assertAlmostEqual(_objective_layout_cost([text],units,source,600,400,1.),
                               _objective_layout_cost([moved],units,source,600,400,1.))
        clipped=deepcopy(moved);clipped['height']=1.
        self.assertGreater(_objective_layout_cost([clipped],units,source,600,400,1.),0.)

    def test_mild_contrast_and_unit_gaps_do_not_veto_physically_clean_qwen(self):
        q=dict(candidate={'name':'qwen-pre-container-font-fitted'},
               physical_eligible=True,has_layout_defects=False,layout_defect_cost=0.,
               low_contrast=['label'],mild_contrast_regressions=['label'],contrast_regressions=[])
        fallback=dict(candidate={'name':'copy-first'},has_layout_defects=False,
                      layout_defect_cost=0.,low_contrast=[],source_relationship_error=.2)
        result=_select_visual_candidate([q,fallback],[1.,0.],[0.,100.],
                                        {'qwen-pre-container-font-fitted':False},True)
        self.assertEqual(result['selected_index'],0)
        q['contrast_regressions']=['label'];q['has_layout_defects']=True
        result=_select_visual_candidate([q,fallback],[1.,0.],[100.,0.],True,True)
        self.assertEqual(result['selected_index'],1)

    def test_readable_banner_reflow_can_beat_microscopic_clean_qwen(self):
        q=dict(candidate={'name':'qwen-proposal'},physical_eligible=True,has_layout_defects=False,
               layout_defect_cost=0.,low_contrast=[],readability_penalty=20.)
        h=dict(candidate={'name':'inline-banner'},has_layout_defects=False,layout_defect_cost=0.,
               low_contrast=[],readability_penalty=3.)
        result=_select_visual_candidate([q,h],[20.,3.],[100.,0.],True,True)
        self.assertEqual(result['selected_index'],1)

    def test_larger_clean_primary_objects_can_beat_a_shrunken_banner_stack(self):
        q=dict(candidate={'name':'qwen-proposal'},physical_eligible=True,has_layout_defects=False,
               layout_defect_cost=0.,low_contrast=[],primary_scales={'title':20.,'hero':30.,'logo':10.})
        h=dict(candidate={'name':'inline-banner'},has_layout_defects=False,layout_defect_cost=0.,
               low_contrast=[],primary_scales={'title':30.,'hero':60.,'logo':20.})
        result=_select_visual_candidate([q,h],[0.,1.],[100.,0.],True,True)
        self.assertEqual(result['selected_index'],1)
        # If the learned headline is larger, neither layout dominates all primaries.
        q['primary_scales']['title']=40.
        result=_select_visual_candidate([q,h],[0.,1.],[100.,0.],True,True)
        self.assertEqual(result['selected_index'],0)

    def test_explicit_label_cannot_leave_its_backing(self):
        from engine.resize import _backing_containment_defects
        backing=dict(id='button',type='shape',shape_type='rectangle',x=20.,y=20.,width=220.,height=60.,
                     z_index=0,group_id='button-group')
        label=title();label.update(id='label',group_id='button-group',x=30.,y=30.,width=180.,height=40.)
        source=dict(canvas=dict(width=500,height=200),elements=[backing,label])
        moved=deepcopy(label);moved['y']=90.
        self.assertEqual(_backing_containment_defects([backing,label],source),[])
        self.assertTrue(_backing_containment_defects([backing,moved],source))

    def test_large_vlm_score_cannot_override_a_hard_defect(self):
        entries=[dict(candidate={'name':'qwen-proposal'},has_layout_defects=True,
                      layout_defect_cost=400.,low_contrast=[]),
                 dict(candidate={'name':'semantic-stack'},has_layout_defects=False,
                      layout_defect_cost=0.,low_contrast=[])]
        result=_select_visual_candidate(entries,[400.,0.],[100.,0.],True,True)
        self.assertEqual(result['selected_index'],1)

    def test_emblem_similarity_is_not_a_physical_defect(self):
        text=title();text['width']=400.
        source={'canvas':{'width':600,'height':400},'elements':[text]}
        with patch('engine.qwen_refine.source_emblem_error',return_value=.48):
            self.assertEqual(_objective_layout_cost([text],_analyze(source)[1],source,600,400,1.),0.)
        entries=[dict(candidate={'name':'qwen-container-locked-font-fitted'},
                      has_layout_defects=False,layout_defect_cost=0.,source_lockup_error=.05,low_contrast=[]),
                 dict(candidate={'name':'semantic-stack'},has_layout_defects=False,
                      layout_defect_cost=0.,low_contrast=[])]
        result=_select_visual_candidate(entries,[0.,0.],[0.,100.],True,True)
        self.assertEqual(result['selected_index'],0)

    def test_bleeding_rotated_backdrop_retains_scale_and_legal_frames(self):
        from engine.qwen_refine import _rotated_bounds
        photo=dict(id='photo',type='image',asset_id='missing.png',x=-447.5,y=-95.9,
                   width=1267.5,height=846.2,rotation=-90.,z_index=0,fit='contain')
        text=title();source={'canvas':{'width':1200,'height':628},'elements':[photo,text]}
        backgrounds,units=_analyze(source)
        self.assertEqual([e['id'] for e in backgrounds],['photo'])
        self.assertNotIn('photo',{e['id'] for u in units for e in u['items']})
        fitted=_assemble([text],backgrounds,source,940,788)['elements'][0]
        l,t,r,b=_rotated_bounds(fitted)
        self.assertAlmostEqual(l,0.)
        self.assertAlmostEqual(fitted['width']/fitted['height'],photo['width']/photo['height'])
        self.assertEqual((fitted['rotation'],fitted['asset_id']),(0.,'missing.png'))
        for x,y,w,h in [(fitted['x'],fitted['y'],fitted['width'],fitted['height']),(l,t,r-l,b-t)]:
            self.assertGreaterEqual(x,-1e-6);self.assertGreaterEqual(y,-1e-6)
            self.assertLessEqual(x+w,940.+1e-6);self.assertLessEqual(y+h,788.+1e-6)
        small=deepcopy(photo);small.update(x=-10.,y=100.,width=80.,height=100.)
        source['elements'][0]=small
        self.assertFalse(_analyze(source)[0])

    def test_calm_cell_moves_every_container_member_rigidly(self):
        a=title();a.update(group_id='lockup',content='Label',width=100.,height=40.)
        line=dict(id='line',type='shape',shape_type='rectangle',fill_color='#ffffff',
                  group_id='lockup',x=30.,y=95.,width=100.,height=2.,z_index=1)
        source={'canvas':{'width':400,'height':200},'elements':[a,line]}
        under=Image.new('RGB',(400,200),'white');under.paste('black',(200,0,400,200))
        with patch('engine.resize._feasible',return_value=False):
            variants=whole_lockup_contrast_variants(source['elements'],_analyze(source)[1],source,400,200,['title'],under)
        self.assertTrue(variants)
        by_id={e['id']:e for e in variants[0][0]}
        self.assertAlmostEqual(by_id['title']['x']-a['x'],by_id['line']['x']-line['x'])
        self.assertAlmostEqual(by_id['title']['y']-a['y'],by_id['line']['y']-line['y'])
        self.assertEqual(by_id['title']['font_size'],a['font_size'])
        self.assertGreater(by_id['title']['x'],200.)

    def test_oversized_inferred_art_group_can_move_its_explicit_copy_lockup(self):
        a=title();a.update(group_id='copy',width=100.,height=60.,y=50.,content='Title')
        line=dict(id='line',type='shape',shape_type='rectangle',fill_color='#ffffff',
                  group_id='copy',x=30.,y=115.,width=100.,height=2.,z_index=1)
        art=dict(id='art',type='image',asset_id='missing.png',x=0.,y=85.,
                 width=400.,height=110.,z_index=0)
        source={'canvas':{'width':400,'height':200},'elements':[art,a,line]}
        under=Image.new('RGB',(400,200),'white');under.paste('black',(200,0,400,200))
        variants=whole_lockup_contrast_variants(source['elements'],_analyze(source)[1],source,400,200,['title'],under)
        self.assertTrue(variants)
        result={e['id']:e for e in variants[0][0]}
        self.assertEqual(result['art'],art)
        self.assertGreater(result['title']['x'],200.)
        self.assertAlmostEqual(result['title']['x']-a['x'],result['line']['x']-line['x'])
        self.assertAlmostEqual(result['title']['y']-a['y'],result['line']['y']-line['y'])

    def test_engine_summary_keeps_raw_list_separate_from_count(self):
        rows=[dict(valid=True,geometry_mae=.1,font_mae=.2,raw_geometry_mae=.3,
                   raw_relative_font_mae=.4,raw_proposal_valid=True,raw_json_complete=True,
                   pair_id='a',hard_checks_passed=True,qwen_family_win=True)]
        summary=summarize_engine(rows)
        self.assertEqual(summary['raw_proposal_valid_count'],1)
        self.assertEqual(summary['raw_proposal_valid_rate'],1.)

    def test_duplicate_ids_do_not_pass_geometry_measurement(self):
        row=dict(id='a',x=0,y=0,width=10,height=10)
        self.assertFalse(measure([row,row],{'elements':[row]},dict(width=100,height=100))['exact_ids'])
