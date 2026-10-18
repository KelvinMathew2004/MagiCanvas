"""Contract and relationship regressions; run with python -m unittest discover -s tests."""
from copy import deepcopy
import json
from pathlib import Path
import random
import socket
import unittest
from unittest.mock import patch

from engine.resize import resize, _bounds, _text
from engine.main import self_check

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [(1080,1080),(1080,1920),(1920,1080),(1600,400),(728,90),(320,568)]


class ResizeTests(unittest.TestCase):
    def assert_contract(self, source, target, output):
        self.assertEqual(output['canvas'], target)
        self.assertEqual([e['id'] for e in output['elements']], [e['id'] for e in source['elements']])
        self.assertEqual(self_check(source, output), [])
        mutable = {'x','y','width','height','font_size','letter_spacing','corner_radius','stroke_width','fit','text_align'}
        for old,new in zip(source['elements'], output['elements']):
            self.assertEqual(set(old)-mutable, set(new)-mutable)
            for key,value in old.items():
                if key not in mutable:
                    self.assertEqual(new[key], value)
            l,t,r,b = _bounds([new])
            edge_bleed = new['type']=='shape' and new.get('shape_type')=='ellipse' and new.get('opacity',1)<.4
            if not edge_bleed:
                self.assertGreaterEqual(l, -.00001)
                self.assertGreaterEqual(t, -.00001)
                self.assertLessEqual(r, target['width']+.00001)
                self.assertLessEqual(b, target['height']+.00001)
            if new['type']=='image' and old.get('fit','contain')!='cover' and old['width']*old['height']:
                self.assertAlmostEqual(new['width']/new['height'], old['width']/old['height'], places=6)
            if new['type']=='text':
                self.assertGreaterEqual(new['font_size'], 1)
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            return
        Draft202012Validator(json.loads((ROOT/'chameleon_design_schema_v0.9.json').read_text())).validate(output)

    def test_samples_across_formats_without_network_or_input_mutation(self):
        for path in sorted((ROOT/'input').glob('*.json')):
            source=json.loads(path.read_text())['source']
            before=deepcopy(source)
            for width,height in TARGETS:
                with self.subTest(sample=path.name,target=(width,height)):
                    target=dict(width=width,height=height)
                    with patch.object(socket,'create_connection',side_effect=AssertionError('network access')):
                        out=resize(source,target)
                    self.assert_contract(source,target,out)
                    self.assertEqual(source,before)

    def test_artwork_group_uses_one_uniform_transform(self):
        source=dict(canvas=dict(width=800,height=800),elements=[
            dict(id='panel',type='shape',shape_type='rectangle',fill_color='#a08060',x=400,y=120,width=300,height=500,group_id='art'),
            dict(id='stem',type='shape',shape_type='rectangle',fill_color='#204020',x=530,y=220,width=10,height=180,rotation=-12,group_id='art'),
            dict(id='leaf',type='shape',shape_type='ellipse',fill_color='#406040',x=480,y=210,width=70,height=30,rotation=25,group_id='art'),
            dict(id='words',type='text',content='An intentional layout',font_family='Solstice Serif',font_size=40,color='#202020',x=30,y=200,width=300,height=200)])
        for w,h in TARGETS:
            target=dict(width=w,height=h);out=resize(source,target)
            self.assert_contract(source,target,out)
            before=source['elements'][:3];after=out['elements'][:3]
            scale=after[0]['width']/before[0]['width']
            for a,b in zip(before,after):
                self.assertAlmostEqual(b['width'],a['width']*scale)
                self.assertAlmostEqual(b['height'],a['height']*scale)
                self.assertEqual(a.get('rotation',0),b.get('rotation',0))
                self.assertAlmostEqual(b['x']-after[0]['x'],(a['x']-before[0]['x'])*scale)
                self.assertAlmostEqual(b['y']-after[0]['y'],(a['y']-before[0]['y'])*scale)

    def test_button_text_fits_and_background_covers_target(self):
        task=json.loads((ROOT/'input/chameleon_sample_level2_core.json').read_text())
        output=resize(task['source'],dict(width=1080,height=1920))
        for e in output['elements']:
            if e['type']=='text':
                h,w,_=_text(e)
                self.assertLessEqual(h,e['height']+.01)
                self.assertLessEqual(w,e['width']+.01)
        background=output['elements'][0]
        self.assertEqual((background['x'],background['y'],background['width'],background['height']),(0,0,1080,1920))
        label=next(e for e in output['elements'] if e['id']=='cta_label')
        self.assertEqual(_text(label)[2],1)

    def test_unknown_ids_properties_long_words_and_zero_size(self):
        rng=random.Random(93)
        for n in (1,7,21):
            elements=[]
            for i in range(n):
                elements.append(dict(id=f'opaque-{rng.randrange(10000000)}',type='text',content=('Supercalifragilisticexpialidocious ' if i%2 else 'Wide copy\nwith a manual break ')*2,
                    font_family='Solstice Sans',font_size=20+i%5,color='#222',x=rng.randrange(600),y=rng.randrange(600),width=180,height=80,
                    custom_metadata={'nested':[i]},letter_spacing=.5,max_lines=5))
            elements.append(dict(id='zero',type='shape',shape_type='line',fill_color='#000',x=10,y=10,width=0,height=0))
            source=dict(canvas=dict(width=800,height=800),elements=elements)
            target=dict(width=300,height=900)
            self.assert_contract(source,target,resize(source,target))

    def test_invalid_targets_and_duplicate_ids(self):
        source=dict(canvas=dict(width=100,height=100),elements=[])
        for value in (0,-1,float('nan'),float('inf'),True,10.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                resize(source,dict(width=value,height=100))
        source['elements']=[dict(id='same'),dict(id='same')]
        with self.assertRaisesRegex(ValueError,'unique'):
            resize(source,dict(width=100,height=100))

    def test_deterministic(self):
        task=json.loads((ROOT/'input/chameleon_sample_level2_core.json').read_text())
        self.assertEqual(resize(task['source'],task['target_canvas']),resize(task['source'],task['target_canvas']))


class SiglipRerankTests(unittest.TestCase):
    """The learned SigLIP re-rank only picks among already-valid candidates inside
    a near-tie band; it degrades cleanly when the model is absent and never runs
    with fewer than two candidates or overrides a materially worse layout."""

    def _entry(self, name, energy, cost=0.):
        from PIL import Image
        return dict(candidate=dict(name=name, energy=energy), cost=cost,
                    image=Image.new('RGB', (8, 8)))

    def _cost(self, e):
        return e['candidate']['energy']+e['cost']

    def test_fallback_when_model_absent_returns_no_selection(self):
        from engine.resize import _siglip_rerank
        entries=[self._entry('A',100.),self._entry('B',110.)]
        with patch('engine.vision.score_layouts',return_value=(None,'model unavailable')):
            report=_siglip_rerank(entries,self._cost)
        self.assertFalse(report['used'])
        self.assertIsNone(report['selected_index'])

    def test_single_candidate_skips_tie_break(self):
        from engine.resize import _siglip_rerank
        entries=[self._entry('A',1.)]
        report=_siglip_rerank(entries,self._cost)
        self.assertFalse(report['used'])
        self.assertIsNone(report['selected_index'])

    def test_learned_score_breaks_a_near_tie(self):
        from unittest.mock import patch
        from engine.resize import _siglip_rerank
        # A and B are effectively tied; SigLIP prefers B.
        entries=[self._entry('A',100.),self._entry('B',100.1)]
        with patch('engine.vision.score_layouts',return_value=([0.,5.],'mock')):
            report=_siglip_rerank(entries,self._cost)
        self.assertTrue(report['used'])
        self.assertEqual(report['selected_index'],1)
        self.assertTrue(report['overrode_heuristic'])

    def test_does_not_rescue_a_materially_worse_candidate(self):
        from unittest.mock import patch
        from engine.resize import _siglip_rerank
        # B is far worse on the heuristics (outside the band); even though SigLIP
        # loves it, the better-heuristic candidate A must win.
        entries=[self._entry('A',100.),self._entry('B',500.)]
        with patch('engine.vision.score_layouts',return_value=([0.,9.],'mock')):
            report=_siglip_rerank(entries,self._cost)
        self.assertTrue(report['used'])
        self.assertEqual(report['selected_index'],0)
        self.assertFalse(report['overrode_heuristic'])

    def test_square_visual_does_not_override_readability_cost(self):
        from unittest.mock import patch
        from engine.resize import _siglip_rerank
        # A visually preferred but substantially more expensive large-hero
        # candidate must not beat the smaller, more readable square layout.
        entries=[self._entry('readable',8.6552),self._entry('large-hero',12.217)]
        with patch('engine.vision.score_layouts',return_value=([0.,9.],'mock')):
            report=_siglip_rerank(entries,self._cost)
        self.assertEqual(report['selected_index'],0)
        self.assertFalse(report['overrode_heuristic'])


if __name__=='__main__':
    unittest.main()
