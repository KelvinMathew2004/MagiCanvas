"""Candidate axes survive refinement; no template IDs or fixed format rules."""
from copy import deepcopy
import unittest
from engine.source_structure import copy_axis_contracts, apply_copy_axis_contracts


def column(align='left'):
    def text(eid,x,y,w):
        return dict(id=eid,type='text',content='Words',font_family='Arial',font_size=10,
                    x=x,y=y,width=w,height=16,text_align=align)
    xs=([50,50] if align=='left' else [90,50] if align=='right' else [70,50])
    es=[text('a',xs[0],45,60),text('b',xs[1],75,100)]
    es += [dict(id='panel',type='shape',shape_type='rectangle',x=50,y=10,width=40,height=20),
           text('label',55,12,30)]
    units=[dict(role='copy',semantic=sem,items=[deepcopy(e)],bounds=(e['x'],e['y'],e['x']+e['width'],e['y']+e['height']))
           for e,sem in zip(es[:2],('headline','subheading'))]
    units += [dict(role='tag',semantic='badge',items=deepcopy(es[2:]),bounds=(50,10,90,30))]
    return es,units


class CopyAxisContracts(unittest.TestCase):
    def test_left_column_and_entire_kicker_survive_independent_moves(self):
        es,units=column();contracts=copy_axis_contracts(es,units,400,200)
        self.assertEqual(len(contracts),1);self.assertEqual(len(contracts[0]['members']),3)
        es[0]['x']+=13;es[1]['x']-=4;es[0]['width']+=11
        es[2]['x']+=5;es[3]['x']+=5
        before=deepcopy(es)
        apply_copy_axis_contracts(es,contracts,units,400,200)
        self.assertEqual([e['x'] for e in es[:3]],[50,50,50])
        self.assertEqual(es[3]['x']-es[2]['x'],5)
        for a,b in zip(es,before):
            for key in ('y','width','height','font_size'):self.assertEqual(a.get(key),b.get(key))

    def test_center_and_right_use_their_explicit_axis(self):
        for align in ('center','right'):
            es,units=column(align);contracts=copy_axis_contracts(es,units,400,200)
            self.assertEqual(len(contracts),1)
            es[0]['x']+=9;es[1]['x']-=7
            apply_copy_axis_contracts(es,contracts,units,400,200)
            f=.5 if align=='center' else 1
            self.assertAlmostEqual(es[0]['x']+f*es[0]['width'],es[1]['x']+f*es[1]['width'])

    def test_offset_copy_split_columns_and_incidental_edges_are_not_contracts(self):
        for mode in ('offset','mixed','centers','row','far','rotated'):
            es,units=column()
            if mode=='rotated':es[0]['rotation']=8
            if mode=='offset':es[1]['x']+=3
            if mode=='mixed':es[1]['text_align']='right'
            if mode=='centers':
                for e in es[:2]:e['text_align']='center'
            if mode=='row':es[1]['y']=45
            if mode=='far':es[1]['y']=150
            self.assertEqual(copy_axis_contracts(es,units,400,200),[],mode)

    def test_unaligned_corner_badge_is_not_acquired(self):
        es,units=column();es[2]['x']=250;es[3]['x']=255
        self.assertEqual(len(copy_axis_contracts(es,units,400,200)[0]['members']),2)

    def test_collision_rejects_the_whole_contract(self):
        es,units=column();contracts=copy_axis_contracts(es,units,400,200)
        es[0]['x']+=100;es[1]['x']+=100
        obstacle=dict(id='art',type='shape',shape_type='rectangle',x=48,y=45,width=20,height=16)
        es.append(obstacle);units.append(dict(role='visual',items=[deepcopy(obstacle)]))
        before=deepcopy(es)
        self.assertEqual(apply_copy_axis_contracts(es,contracts,units,400,200),[])
        self.assertEqual(es,before)
