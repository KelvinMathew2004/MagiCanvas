"""Logo slots, editorial columns, hierarchy and vision preprocessing regressions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from PIL import Image

from engine.resize import (_analyze, _bounds, _logo_row, _editorial_split_seed, _banner_seed,
                           _action_band, _callout_block, _decorations, _energy_terms,
                           _badge_flow, _constrain, _move, _align_peer_rows,
                           _stack_balance, _stack_seed, _text)
from engine.vision import _letterbox_square

ROOT=Path(__file__).resolve().parents[1]


def sample(level):
    return json.loads(next((ROOT/'input').glob(f'*level{level}*.json')).read_text())['source']


class LayoutMechanicsTests(unittest.TestCase):
    def test_single_small_top_image_can_be_logo_without_a_second_image(self):
        source=dict(canvas=dict(width=800,height=800),elements=[
            dict(id='opaque',type='image',asset_id='unavailable.png',x=325,y=40,width=150,height=150)])
        unit=_analyze(source)[1][0]
        self.assertEqual((unit['role'],unit['semantic']),('header','logo'))

    def test_banner_logo_uses_tall_slot_and_respects_growth_cap(self):
        units=_analyze(sample(1))[1]
        logos=[u for u in units if u['semantic']=='logo']
        result,h=_logo_row(logos,108,60,.22,3)
        self.assertGreater(h,15)
        self.assertLessEqual(h,60)
        before=logos[0]['items'][0];after=result[0]
        self.assertLessEqual(after['width']/before['width'],.22*1.8+1e-6)
        self.assertAlmostEqual(after['width']/after['height'],before['width']/before['height'])

    def test_banner_keeps_upper_badge_with_copy_and_cta_in_inline_column(self):
        source=sample(2);units=_analyze(source)[1];before=deepcopy(units)
        elements=_editorial_split_seed(units,1600,400,source['canvas']['width'],source['canvas']['height'],.75,.42)
        by_id={e['id']:e for e in elements}
        def box(role):
            return _bounds([by_id[e['id']] for u in units if u['role']==role for e in u['items']])
        copy=box('copy');action=box('action')
        self.assertGreaterEqual(action[0],copy[2]-1e-6)
        upper=[u for u in units if u['role']=='tag' and u.get('source_upper')]
        self.assertTrue(upper)
        for u in upper:
            badge=_bounds([by_id[e['id']] for e in u['items']])
            self.assertAlmostEqual(badge[0],copy[0],places=5)
            self.assertLess(badge[3],copy[1])
        self.assertEqual(units,before)

    def test_warmup_banner_packs_real_widths_and_keeps_text_cta_single_line(self):
        source=sample(1);units=_analyze(source)[1]
        for width,height in ((1600,400),(728,90)):
            with self.subTest(size=(width,height)):
                elements=_banner_seed(units,width,height,1.1,.2,False,(1080,1080))
                by_id={e['id']:e for e in elements}
                copy=next(u for u in units if u['role']=='copy')
                action=next(u for u in units if u['role']=='action')
                cb=_bounds([by_id[e['id']] for e in copy['items']])
                ab=_bounds([by_id[e['id']] for e in action['items']])
                self.assertGreaterEqual(ab[0],cb[2]-1e-6)
                self.assertLess(ab[0]-cb[2],width*.15)
                self.assertEqual(_text(by_id[action['items'][0]['id']])[2],1)

    def test_leaderboard_cta_has_its_own_inline_column(self):
        source=sample(2);units=_analyze(source)[1]
        elements=_banner_seed(units,728,90,1.1,.2,False,(1080,1080))
        by_id={e['id']:e for e in elements}
        copy=[u for u in units if u['role']=='copy']
        action=next(u for u in units if u['role']=='action')
        cb=_bounds([by_id[e['id']] for u in copy for e in u['items']])
        ab=_bounds([by_id[e['id']] for e in action['items']])
        self.assertGreaterEqual(ab[0],cb[2]-1e-6)
        baseline=_energy_terms(elements,units,source,728,90,1)['banner_action_stack']
        stacked=deepcopy(elements)
        stacked_by_id={e['id']:e for e in stacked}
        for item in action['items']:
            stacked_by_id[item['id']]['x']=cb[0]
        self.assertGreater(_energy_terms(stacked,units,source,728,90,1)['banner_action_stack'],baseline)

    def test_wide_hero_clears_right_corner_content_and_uses_height(self):
        source=sample(3);units=_analyze(source)[1]
        elements=_banner_seed(units,1600,400,1.1,.3,False,(1080,1350))
        by_id={e['id']:e for e in elements}
        hero=next(u for u in units if u['semantic']=='hero')
        callout=next(u for u in units if u.get('corner_callout'))
        hb=_bounds([by_id[e['id']] for e in hero['items']])
        cb=_bounds([by_id[e['id']] for e in callout['items']])
        self.assertGreaterEqual((hb[3]-hb[1])/400,.64)
        self.assertLessEqual(hb[2],cb[0]+1e-6)

    def test_landscape_peer_tags_stay_separate_from_primary_action(self):
        units=_analyze(sample(3))[1]
        tags=[u for u in units if u['role']=='tag'];actions=[u for u in units if u['role']=='action']
        elements,h=_action_band(tags,actions,1200,260,1,20,False,align='left')
        by_id={e['id']:e for e in elements}
        tag_box=_bounds([by_id[e['id']] for u in tags for e in u['items']])
        action_box=_bounds([by_id[e['id']] for u in actions for e in u['items']])
        self.assertAlmostEqual(tag_box[0],action_box[0])
        self.assertLess(tag_box[3],action_box[1])
        tag_heights=[_bounds([by_id[e['id']] for e in u['items']])[3]-
                     _bounds([by_id[e['id']] for e in u['items']])[1] for u in tags]
        self.assertLess(max(tag_heights)-min(tag_heights),1e-5)
        self.assertLessEqual(h,260+1e-5)
        self.assertGreater(action_box[3]-action_box[1],
                           max(_bounds([by_id[e['id']] for e in u['items']])[3]-
                               _bounds([by_id[e['id']] for e in u['items']])[1] for u in tags))

    def test_peer_button_row_is_repaired_to_a_shared_vertical_axis(self):
        source=sample(3);units=_analyze(source)[1]
        tags=[u for u in units if u['role']=='tag'];actions=[u for u in units if u['role']=='action']
        elements=_banner_seed(units,1200,260,1.1,.3,False,(1080,1350))
        by_id={e['id']:e for e in elements}
        for e in tags[0]['items']:
            by_id[e['id']]['y']+=8
        repaired=_constrain(elements,units,1200,260)
        self.assertIsNotNone(repaired)
        for e in tags[1]['items']:
            next(item for item in repaired if item['id']==e['id'])['y']+=6
        _align_peer_rows(repaired,units)
        fixed={e['id']:e for e in repaired}
        centers=[]
        for unit in tags:
            box=_bounds([fixed[e['id']] for e in unit['items']])
            centers.append((box[1]+box[3])/2)
        self.assertLess(max(centers)-min(centers),1e-5)
        heights=[_bounds([fixed[e['id']] for e in unit['items']])[3]-
                 _bounds([fixed[e['id']] for e in unit['items']])[1] for unit in tags]
        self.assertLess(max(heights)-min(heights),1e-5)

    def test_contrasting_badge_stays_between_matching_button_colors(self):
        colors=['#eb6b35','#241725','#eb6b35']
        units=[];records=[]
        for i,color in enumerate(colors):
            x=i*40
            item=dict(id=f'tag{i}',type='shape',shape_type='rectangle',fill_color=color,
                      x=x,y=0,width=30,height=12)
            unit=dict(items=[item],bounds=(x,0,x+30,12),role='tag',semantic='badge',
                      peer_ids=('tag0','tag1','tag2'),source_upper=False)
            units.append(unit);records.append((unit,[item]))
        centered=_badge_flow(records,120,40)
        displaced=deepcopy(records)
        # Move the contrasting black pill to the edge, leaving both orange pills together.
        displaced[0][1][0]['x']=40
        displaced[1][1][0]['x']=0
        displaced[2][1][0]['x']=80
        self.assertGreater(_badge_flow(displaced,120,40),centered)

    def test_banner_cta_keeps_clearance_from_product_when_room_exists(self):
        source=sample(3);units=_analyze(source)[1]
        elements=_banner_seed(units,1600,400,1.1,.3,False,(1080,1350))
        by_id={e['id']:e for e in elements}
        action=next(u for u in units if u['semantic']=='cta')
        hero=next(u for u in units if u['semantic']=='hero')
        action_items=[by_id[e['id']] for e in action['items']]
        ab,hb=_bounds(action_items),_bounds([by_id[e['id']] for e in hero['items']])
        spaced=deepcopy(elements);spaced_by_id={e['id']:e for e in spaced}
        spaced_items=[spaced_by_id[e['id']] for e in action['items']]
        _move(spaced_items,50,ab[1])
        before=_energy_terms(spaced,units,source,1600,400,1.1)['primary_clearance']
        cramped=deepcopy(elements);cramped_by_id={e['id']:e for e in cramped}
        cramped_items=[cramped_by_id[e['id']] for e in action['items']]
        _move(cramped_items,hb[0]-(ab[2]-ab[0])*.15,(hb[1]+hb[3]-ab[3]+ab[1])/2)
        after=_energy_terms(cramped,units,source,1600,400,1.1)['primary_clearance']
        self.assertGreater(after,before)

    def test_callout_grows_above_short_side_scale_with_bounded_height(self):
        source=sample(3);u=next(u for u in _analyze(source)[1] if u.get('corner_callout'))
        before=deepcopy(u)
        block=_callout_block(u,728,90,1080,1350);box=_bounds(block)
        old=u['bounds'];scale=(box[2]-box[0])/(old[2]-old[0])
        self.assertGreater(scale,90/1350)
        self.assertLessEqual(box[3]-box[1],90*.38+1e-6)
        self.assertEqual(u,before)

    def test_tiny_hero_receives_prominence_penalty(self):
        source=sample(3);units=_analyze(source)[1]
        hero=next(u for u in units if u['semantic']=='hero');ids={e['id'] for e in hero['items']}
        small=deepcopy(source['elements'])
        for e in small:
            if e['id'] in ids:e['width']*=.1;e['height']*=.1
        baseline=_energy_terms(source['elements'],units,source,1080,1350,1)
        changed=_energy_terms(small,units,source,1080,1350,1)
        self.assertEqual(baseline['hero_prominence'],0)
        self.assertGreater(changed['hero_prominence'],.5)

    def test_square_hero_stays_within_source_relative_scale_when_slot_expands(self):
        source=sample(3);units=_analyze(source)[1]
        hero=next(u for u in units if u['semantic']=='hero')
        small=_stack_seed(units,1080,1080,1.15,.42,True,(1080,1350))
        large=_stack_seed(units,1080,1080,1.15,.72,True,(1080,1350))
        def height(elements):
            by_id={e['id']:e for e in elements}
            return _bounds([by_id[e['id']] for e in hero['items']])[3]-_bounds([by_id[e['id']] for e in hero['items']])[1]
        source_height=hero['bounds'][3]-hero['bounds'][1]
        self.assertLessEqual(height(large),source_height*1.15+1e-5)
        self.assertLessEqual(height(small),source_height*1.15+1e-5)
        self.assertLess(height(small),height(large))

    def test_background_circle_keeps_its_crop_at_the_canvas_edge(self):
        units=_analyze(sample(2))[1]
        circle=next(u for u in units if u['role']=='decoration' and u['items'][0].get('shape_type')=='ellipse')
        box=_bounds(_decorations([circle],1080,1080,1080,1080,[]))
        self.assertLess(box[0],0)
        self.assertLess(box[1],0)

    def test_smaller_supporting_copy_keeps_subheading_relationship(self):
        units=_analyze(sample(2))[1]
        headline=next(u for u in units if u['semantic']=='headline')
        supporting=next(u for u in units if u['semantic']=='subheading')
        self.assertLess(max(e['font_size'] for e in supporting['items']),
                        max(e['font_size'] for e in headline['items'])*.48)

    def test_vertical_stack_prefers_balanced_group_and_outer_gaps(self):
        units=[
            dict(semantic='logo',role='header',source_upper=False),
            dict(semantic='hero',role='visual',source_upper=False),
            dict(semantic='headline',role='copy',source_upper=False),
            dict(semantic='cta',role='action',source_upper=False),
        ]
        records=[(u,[]) for u in units]
        balanced=[(0,50,20,100),(0,200,20,300),(0,400,20,500),(0,600,20,650)]
        uneven=[(0,10,20,60),(0,90,20,190),(0,500,20,600),(0,900,20,950)]
        self.assertLess(_stack_balance(records,balanced,1000,1000),
                        _stack_balance(records,uneven,1000,1000))

    def test_decorative_shapes_can_scale_with_available_space(self):
        source=sample(2);units=_analyze(source)[1]
        circle=next(u for u in units if u['role']=='decoration' and u['items'][0].get('shape_type')=='ellipse')
        result=_decorations([circle],1080,1080,1080,1080,[])
        box=_bounds(result)
        self.assertGreater(box[2]-box[0],1080*.32)

    def test_avoidable_wrap_penalty_when_horizontal_room_exists(self):
        source=dict(canvas=dict(width=800,height=100),elements=[
            dict(id='copy',type='text',content='Chase the Light',font_family='Solstice Serif',font_size=20,
                 color='rgb(255,255,255)',x=150,y=10,width=50,height=80),
            dict(id='neighbor',type='shape',shape_type='rectangle',fill_color='rgb(10,20,30)',
                 x=650,y=10,width=100,height=80)])
        units=_analyze(source)[1]
        narrow=_energy_terms(source['elements'],units,source,800,100,1)
        expanded=deepcopy(source['elements']);expanded[0]['width']=400
        wide=_energy_terms(expanded,units,source,800,100,1)
        self.assertGreater(narrow['avoidable_wrap'],wide['avoidable_wrap'])



class LetterboxTests(unittest.TestCase):
    def test_wide_and_tall_contents_keep_geometry_and_source_pixels(self):
        for size in ((32,8),(8,32),(16,16)):
            with self.subTest(size=size):
                source=Image.new('RGB',size,(10,20,30));source.putpixel((size[0]//2,size[1]//2),(240,90,40))
                before=source.tobytes();result=_letterbox_square(source);side=max(size)
                self.assertEqual(result.size,(side,side))
                x=(side-size[0])//2;y=(side-size[1])//2
                self.assertEqual(result.crop((x,y,x+size[0],y+size[1])).tobytes(),before)
                self.assertEqual(source.tobytes(),before)
                self.assertEqual(result.getpixel((0,0)),(10,20,30))
                self.assertIsNot(result,source)

    def test_scoring_batches_letterboxed_candidates_in_one_forward_pass(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import MagicMock,patch
        from engine.vision import score_layouts
        processor=MagicMock(return_value={})
        logits=MagicMock();positive=MagicMock();negative=MagicMock()
        logits.__getitem__.side_effect=[positive,negative]
        positive.mean.return_value.__sub__.return_value.tolist.return_value=[.1,.2]
        model=MagicMock(return_value=SimpleNamespace(logits_per_image=logits))
        torch=SimpleNamespace(inference_mode=lambda:nullcontext())
        with patch('engine.vision._load',return_value=(model,processor,torch)):
            scores,_=score_layouts([Image.new('RGB',(32,8)),Image.new('RGB',(8,32))])
        self.assertEqual(scores,[.1,.2])
        self.assertEqual(model.call_count,1)
        self.assertEqual([image.size for image in processor.call_args.kwargs['images']],[(32,32),(32,32)])



if __name__=='__main__':unittest.main()
