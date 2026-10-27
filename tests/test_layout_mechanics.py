"""Logo slots, editorial columns, hierarchy and vision preprocessing regressions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from PIL import Image

from engine.resize import (_analyze, _bounds, _logo_row, _editorial_split_seed, _banner_seed,
                           _action_band, _badge_row, _callout_block, _decorations, _energy_terms,
                           _badge_flow, _constrain, _move, _align_peer_rows,
                           _stack_balance, _stack_seed, _text, _copy,
                           _repair_banner_logo_column_balance)
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

    def test_dense_leaderboard_logo_balances_visible_gutters(self):
        source=sample(3);units=_analyze(source)[1]
        elements=_banner_seed(units,728,90,1.1,.2,False,(1080,1350))
        by_id={e['id']:e for e in elements}
        logo=next(u for u in units if u['semantic']=='logo')
        copy=[u for u in units if u['role']=='copy']
        _repair_banner_logo_column_balance(elements,units,728,90)
        by_id={e['id']:e for e in elements}
        logo_box=_bounds([by_id[e['id']] for e in logo['items']])
        copy_box=_bounds([by_id[e['id']] for u in copy for e in u['items']])
        inset=90*.05;clearance=90*.025
        self.assertAlmostEqual(logo_box[0]-inset,
                               copy_box[0]-clearance-logo_box[2],places=4)

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
        for unit in units:unit['raw_frame_limit']=(1200,260)
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

    def test_spacing_sibling_preserves_peer_internals_and_compacts_gaps(self):
        from engine.resize import _even_source_peer_spacing
        source=sample(3);units=_analyze(source)[1]
        elements=_banner_seed(units,1200,260,1.1,.3,False,(1080,1350))
        tags=sorted((u for u in units if u.get('peer_ids')),key=lambda u:u['bounds'][0])
        by_id={e['id']:e for e in elements}
        for e in tags[1]['items']:by_id[e['id']]['x']+=5
        old=[_bounds([by_id[e['id']] for e in u['items']]) for u in tags]
        before=deepcopy(elements)
        result=_even_source_peer_spacing(elements,units,1200,260)
        self.assertIsNotNone(result)
        fixed={e['id']:e for e in result}
        boxes=[_bounds([fixed[e['id']] for e in u['items']]) for u in tags]
        self.assertAlmostEqual(boxes[1][0]-boxes[0][2],boxes[2][0]-boxes[1][2])
        self.assertLessEqual(boxes[1][0]-boxes[0][2],.46*(boxes[0][3]-boxes[0][1]))
        for a,b in zip(old,boxes):self.assertAlmostEqual(a[2]-a[0],b[2]-b[0])
        self.assertEqual(elements,before)

    def _unequal_peer_column(self):
        units=[];elements=[];source_x=236.;target_x=110.
        ids=tuple(f'control{i}' for i in range(5))
        for i,(sw,tw) in enumerate(zip((80,100,120,100,80),(90,140,160,110,120))):
            panel=dict(id=ids[i],type='shape',x=source_x,y=300,width=sw,height=40)
            label=dict(id=f'label{i}',type='text',x=source_x+8,y=310,width=sw-16,height=20,font_size=15,content=str(i))
            units.append(dict(role='tag',peer_ids=ids,bounds=(source_x,300,source_x+sw,340),items=[panel,label]))
            elements.extend([dict(panel,x=target_x,y=400,width=tw,height=30),
                             dict(label,x=target_x+8,y=405,width=tw-16,height=20)])
            source_x+=sw+12;target_x+=tw+12
        for i,(sx,sw,tx,tw) in enumerate(((300,400,400,400),(350,300,450,300))):
            item=dict(id=f'copy{i}',type='text',x=sx,y=100+i*70,width=sw,height=40,font_size=20,content='Supporting column',text_align='center')
            units.append(dict(role='copy',bounds=(sx,item['y'],sx+sw,item['y']+40),items=[item]))
            elements.append(dict(item,x=tx,y=100+i*70,width=tw))
        return units,elements

    def test_unequal_five_peer_row_inherits_corroborated_copy_axis(self):
        from engine.resize import _even_source_peer_spacing
        from statistics import median
        units,elements=self._unequal_peer_column();original=deepcopy(elements)
        result=_even_source_peer_spacing(elements,units,1200,600)
        self.assertIsNotNone(result);by={e['id']:e for e in result}
        boxes=[_bounds([by[e['id']] for e in u['items']]) for u in units[:5]]
        self.assertAlmostEqual(median((b[0]+b[2])/2 for b in boxes),600.)
        gaps=[b[0]-a[2] for a,b in zip(boxes,boxes[1:])]
        self.assertLess(max(gaps)-min(gaps),1e-8)
        for i in range(5):
            self.assertAlmostEqual(by[f'label{i}']['x']-by[f'control{i}']['x'],8.)
            self.assertEqual(by[f'label{i}']['font_size'],15)
        self.assertEqual(elements,original)

    def test_peer_axis_does_not_override_reflowed_copy_columns(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column();elements[-1]['x']+=250
        self.assertIsNone(_even_source_peer_spacing(elements,units,1200,600))

    def test_peer_axis_does_not_infer_center_from_left_aligned_text_frames(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        for u in units[-2:]:u['items'][0]['text_align']='left'
        self.assertIsNone(_even_source_peer_spacing(elements,units,1200,600))

    def test_peer_row_follows_a_reflowed_left_copy_column(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        for e in elements[-2:]:e.update(x=400,text_align='left')
        result=_even_source_peer_spacing(elements,units,1200,600)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(next(e for e in result if e['id']=='control0')['x'],400.)

    def test_peer_row_follows_a_reflowed_right_copy_column(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        for e in elements[-2:]:e.update(x=800-e['width'],text_align='right')
        result=_even_source_peer_spacing(elements,units,1200,600)
        self.assertIsNotNone(result)
        last=next(e for e in result if e['id']=='control4')
        self.assertAlmostEqual(last['x']+last['width'],800.)

    def test_peer_row_does_not_use_incidental_edges_of_centered_text(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        for e in elements[-2:]:e['x']=400
        self.assertIsNone(_even_source_peer_spacing(elements,units,1200,600))

    def test_peer_axis_does_not_invent_centering_for_asymmetric_source(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        for u in units[-2:]:
            a,b,c,d=u['bounds'];u['bounds']=(a+60,b,c+60,d)
            u['items'][0]['x']+=60
        self.assertIsNone(_even_source_peer_spacing(elements,units,1200,600))

    def test_peer_axis_preserves_explicit_source_left_alignment(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        a,b,c,d=units[-2]['bounds'];units[-2]['bounds']=(236,b,764,d)
        units[-2]['items'][0].update(x=236,width=528)
        result=_even_source_peer_spacing(elements,units,1200,600)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(next(e for e in result if e['id']=='control0')['x'],400.)

    def test_peer_axis_leaves_wrapped_controls_unchanged(self):
        from engine.resize import _even_source_peer_spacing
        units,elements=self._unequal_peer_column()
        elements[4]['y']+=80;elements[5]['y']+=80
        self.assertIsNone(_even_source_peer_spacing(elements,units,1200,600))

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

    def test_background_circle_stays_inside_raw_canvas_frame(self):
        units=_analyze(sample(2))[1]
        circle=next(u for u in units if u['role']=='decoration' and u['items'][0].get('shape_type')=='ellipse')
        box=_bounds(_decorations([circle],1080,1080,1080,1080,[]))
        self.assertGreaterEqual(box[0],0)
        self.assertGreaterEqual(box[1],0)
        self.assertLessEqual(box[2],1080)
        self.assertLessEqual(box[3],1080)

    def test_shared_badges_wrap_in_a_narrow_skyscraper_slot(self):
        units=_analyze(sample(3))[1]
        tags=[u for u in units if u['role']=='tag']
        for unit in tags:
            unit['target_aspect_ratio']=160/600
            unit['target_short_side']=160
            unit['font_median']=20
        # The badge band itself can be wider than tall even on a skyscraper;
        # the target canvas aspect still needs to trigger the readable wrap.
        elements,height=_badge_row(tags,144,100,1.,8.)
        by_id={e['id']:e for e in elements}
        boxes=sorted((_bounds([by_id[e['id']] for e in unit['items']]) for unit in tags),key=lambda box:box[1])
        self.assertGreater(height,boxes[0][3]-boxes[0][1])
        self.assertTrue(all(boxes[i+1][1]>=boxes[i][3] for i in range(len(boxes)-1)))
        self.assertTrue(all(box[2]-box[0]>=120 for box in boxes))
        self.assertTrue(all(max(e['font_size'] for e in by_id.values() if e['type']=='text'
                                and e['id'] in {item['id'] for item in unit['items']})>=8.8
                            for unit in tags))
        wide_elements,_=_badge_row(tags,400,100,.30,10.)
        wide_by_id={e['id']:e for e in wide_elements}
        wide_boxes=[_bounds([wide_by_id[e['id']] for e in unit['items']]) for unit in tags]
        wide_centers=[(box[1]+box[3])/2 for box in wide_boxes]
        self.assertLess(max(wide_centers)-min(wide_centers),1.)

    def test_narrow_reflow_preserves_source_corner_rotation_and_footer(self):
        source=sample(3);units=_analyze(source)[1]
        callout=next(u for u in units if u.get('corner_callout'))
        ids={e['id'] for e in callout['items']}
        elements=_stack_seed(units,160,600,.30,.42,True,(1080,1350),move_corner_callouts=True)
        from engine.resize import _source_geometry_safety
        elements=_source_geometry_safety(elements,source,units,160,600)
        moved=[e for e in elements if e['id'] in ids]
        box=_bounds(moved)
        self.assertTrue(moved)
        self.assertTrue(all(e.get('rotation',0)==callout['items'][0].get('rotation',0) for e in moved))
        self.assertLess(box[1],600*.20)
        self.assertGreater((box[0]+box[2])/2,80)
        footer_asset=[e for e in elements if e['id']=='holiday_watermark']
        asset_box=_bounds(footer_asset)
        self.assertGreater(asset_box[1],300)
        self.assertLessEqual(asset_box[3],600)
        self.assertLessEqual(asset_box[2]-asset_box[0],160*.14+1e-5)

    def test_smaller_supporting_copy_keeps_subheading_relationship(self):
        units=_analyze(sample(2))[1]
        headline=next(u for u in units if u['semantic']=='headline')
        supporting=next(u for u in units if u['semantic']=='subheading')
        self.assertLess(max(e['font_size'] for e in supporting['items']),
                        max(e['font_size'] for e in headline['items'])*.48)

    def test_copy_spacing_responds_to_canvas_padding(self):
        units=sorted((u for u in _analyze(sample(3))[1] if u['role']=='copy'),
                     key=lambda u:u['bounds'][1])
        tight=_copy(units,900,1300,.8,24)
        padded=_copy(units,900,1300,.8,54)
        tight_by_id={e['id']:e for e in tight};padded_by_id={e['id']:e for e in padded}
        head=next(u for u in units if u['semantic']=='headline')
        sub=next(u for u in units if u['semantic']=='subheading')
        def gap(by_id,a,b):
            ab=_bounds([by_id[e['id']] for e in a['items']])
            bb=_bounds([by_id[e['id']] for e in b['items']])
            return bb[1]-ab[3]
        self.assertGreater(gap(padded_by_id,head,sub),gap(tight_by_id,head,sub))

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

    def test_vertical_stack_compacts_disconnected_core_content(self):
        units=[dict(semantic=role,role=unit_role,source_upper=False,items=[])
               for role,unit_role in (('logo','header'),('headline','copy'),('cta','action'))]
        records=[(unit,[]) for unit in units]
        compact=[(0,210,100,230),(0,260,100,280),(0,310,100,330)]
        spread=[(0,20,100,40),(0,280,100,300),(0,550,100,570)]
        self.assertLess(_stack_balance(records,compact,160,600),
                        _stack_balance(records,spread,160,600))

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
