"""Generic painted-control and source-reading-order regression checks."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from PIL import Image, ImageDraw
from engine.layout_contracts import controls, repair, control_body, _image_regions, _primary_axis, _brand_order
from engine.resize import _analyze, _text_visible_bounds, _text_visual_coverage


def text(eid,x,y,w=150,h=30,size=24):
    return dict(id=eid,type='text',x=x,y=y,width=w,height=h,content='SALE',
                font_family='Solstice Sans',font_size=size,text_align='center',z_index=2)


class VisualContractTests(unittest.TestCase):
    def test_image_control_owns_only_its_label_and_clears_other_copy(self):
        panel=dict(id='pill',type='shape',shape_type='rectangle',x=100,y=300,width=200,height=50,z_index=1)
        source=dict(canvas=dict(width=500,height=500),elements=[panel,text('label',120,310),text('headline',100,100)])
        self.assertEqual(controls(source),(('pill','label'),))
        candidate=deepcopy(source['elements']);candidate[2]['y']=295
        result=repair(candidate,source,_analyze(source)[1],500,500);by={e['id']:e for e in result}
        self.assertGreater(control_body(by['pill'])[1],_text_visible_bounds(by['headline'])[3]+3)

    def test_small_control_restores_readable_font_and_side_padding(self):
        panel=dict(id='pill',type='shape',shape_type='rectangle',x=100,y=300,width=240,height=60,z_index=1)
        source=dict(canvas=dict(width=500,height=500),elements=[panel,text('label',145,315,150,30)])
        candidate=deepcopy(source['elements']);candidate[0].update(width=60,height=15);candidate[1].update(width=60,font_size=6)
        result=repair(candidate,source,_analyze(source)[1],500,500);by={e['id']:e for e in result}
        ink=_text_visible_bounds(by['label']);body=control_body(by['pill'])
        self.assertGreaterEqual(by['label']['font_size'],24)
        self.assertGreater(ink[0]-body[0],3);self.assertGreater(body[2]-ink[2],3)

    def test_ribbon_core_excludes_high_tails(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'ribbon.png';im=Image.new('RGBA',(300,100));d=ImageDraw.Draw(im)
            d.rectangle((40,25,259,99),fill='white');d.rectangle((0,0,30,99),fill='white');d.rectangle((270,0,299,99),fill='white');im.save(path)
            core,_=_image_regions(str(path));self.assertAlmostEqual(core[1],.25,places=2)

    def test_transparent_padding_is_not_text_occlusion(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'flowers.png';im=Image.new('RGBA',(300,100));ImageDraw.Draw(im).rectangle((0,0,15,99),fill='red');im.save(path)
            visual=dict(id='flower',type='image',asset_id=str(path),x=0,y=0,width=300,height=100,z_index=5)
            label=text('label',80,20)
            self.assertEqual(_text_visual_coverage(label,visual),0)
            ImageDraw.Draw(im).rectangle((0,0,299,99),fill='red');other=Path(folder)/'solid.png';im.save(other);visual['asset_id']=str(other)
            self.assertGreater(_text_visual_coverage(label,visual),.99)

    def test_primary_source_axis_centers_independent_units(self):
        source=dict(canvas=dict(width=500,height=500),elements=[text('head',175,100),text('body',175,200)])
        items=deepcopy(source['elements']);items[0]['x']+=30;items[1]['x']-=20
        units=[dict(semantic=role,bounds=(175,y,325,y+30),items=[e]) for role,y,e in zip(('headline','body'),(100,200),source['elements'])]
        _primary_axis(items,source,units,500,700,apply=True)
        for e in items:
            b=_text_visible_bounds(e);self.assertAlmostEqual((b[0]+b[2])/2,250)

    def test_banner_preserves_logo_before_hero(self):
        logo=dict(id='logo',type='image',x=200,y=20,width=100,height=40)
        hero=dict(id='hero',type='image',x=200,y=150,width=100,height=150)
        source=dict(canvas=dict(width=500,height=500),elements=[logo,hero])
        units=[dict(semantic=role,bounds=(e['x'],e['y'],e['x']+e['width'],e['y']+e['height']),items=[e]) for role,e in [('logo',logo),('hero',hero)]]
        items=[dict(logo,x=80,y=20,width=60,height=25),dict(hero,x=0,y=10,width=60,height=70)]
        _brand_order(items,source,units,728,90,apply=True)
        self.assertLess(items[0]['x']+items[0]['width'],items[1]['x'])

    def test_axis_projection_cannot_merge_source_separated_copy(self):
        from engine.layout_contracts import _copy_clearance
        source=dict(canvas=dict(width=500,height=500),elements=[text('head',150,100),text('body',150,200)])
        items=deepcopy(source['elements']);items[1]['y']=items[0]['y']
        _copy_clearance(items,source,500,700)
        self.assertGreater(_text_visible_bounds(items[1])[1],_text_visible_bounds(items[0])[3])

    def test_readable_control_growth_keeps_peer_panels_separate(self):
        from engine.layout_contracts import _control_peer_clearance
        a=dict(id='a',type='shape',shape_type='rectangle',x=50,y=30,width=150,height=35,z_index=1)
        b=dict(a,id='b',y=100)
        source=dict(canvas=dict(width=300,height=300),elements=[a,text('at',60,35,130,25,18),b,text('bt',60,105,130,25,18)])
        items=deepcopy(source['elements']);items[0]['height']=80
        _control_peer_clearance(items,source,300,300)
        self.assertGreater(items[2]['y'],items[0]['y']+items[0]['height'])
        self.assertEqual(items[3]['y']-105,items[2]['y']-100)

    def test_rotated_ribbon_centers_painted_label_in_rotated_space(self):
        from engine.layout_contracts import center_control_labels
        p=dict(id='ribbon',type='shape',shape_type='rectangle',x=100,y=50,width=250,height=50,z_index=1,rotation=-12,group_id='r')
        t=text('label',100,63,250,30,20);t.update(group_id='r',rotation=-12)
        source=dict(canvas=dict(width=500,height=500),elements=[p,t]);items=deepcopy(source['elements'])
        items[1]['y']+=15;center_control_labels(items,source,500,500)
        b=_text_visible_bounds(items[1]);self.assertAlmostEqual((b[0]+b[2])/2,225)
        self.assertAlmostEqual((b[1]+b[3])/2,75)

    def test_source_bottom_attachment_retained_only_in_comparable_regime(self):
        from engine.layout_contracts import _source_edge_attachments
        with TemporaryDirectory() as folder:
            path=Path(folder)/'hero.png';Image.new('RGBA',(100,300),'red').save(path)
            hero=dict(id='hero',type='image',asset_id=str(path),x=200,y=200,width=100,height=300,z_index=0)
            source=dict(canvas=dict(width=500,height=500),elements=[hero])
            items=[dict(hero,y=50)];_source_edge_attachments(items,source,500,500)
            self.assertEqual(items[0]['y']+items[0]['height'],500)
            items=[dict(hero,y=50)];_source_edge_attachments(items,source,1500,400)
            self.assertEqual(items[0]['y'],50)

    def test_inset_card_preserves_source_offsets_before_gutter_scaling(self):
        from engine.layout_contracts import _card_gutters
        panel=dict(id='panel',type='shape',shape_type='rectangle',x=50,y=50,width=400,height=400,z_index=0,group_id='card')
        label=text('title',100,220);label['group_id']='card'
        icon=dict(id='icon',type='image',x=100,y=100,width=60,height=60,z_index=4,group_id='card')
        source=dict(canvas=dict(width=500,height=500),elements=[panel,label,icon])
        items=deepcopy(source['elements']);items[2]['y']=220
        _card_gutters(items,source,_analyze(source)[1],500,500)
        self.assertAlmostEqual((items[1]['y']-items[2]['y'])/items[0]['height'],120/400)
