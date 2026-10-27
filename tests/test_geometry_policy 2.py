from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from PIL import Image

from engine.geometry_policy import preserve_aspects, aspect_issues, elastic_shape_ids
from engine.layout_contracts import art_occlusions, _separate_art
from engine.resize import _text_visible_bounds, _text_visual_coverage_stats, _background_fit


class GeometryPolicyTests(unittest.TestCase):
    def test_artwork_rigid_but_label_panels_and_rules_elastic(self):
        def shape(eid, kind, x, y, w, h, **extra):
            return dict(id=eid,type='shape',shape_type=kind,x=x,y=y,width=w,height=h,z_index=0,**extra)
        items=[shape('circle','ellipse',0,0,100,100,opacity=.15),
               shape('ornament','triangle',100,0,100,80),
               shape('panel','rectangle',100,200,200,60),
               shape('rule','rectangle',100,280,200,2),
               dict(id='label',type='text',x=120,y=215,width=160,height=25,z_index=1)]
        source=dict(canvas=dict(width=500,height=500),elements=items)
        self.assertEqual(elastic_shape_ids(source),frozenset(('panel','rule')))
        output=deepcopy(items)
        for e in output:
            if e['type']=='shape':e['width']*=2
        self.assertEqual(len(aspect_issues(output,source)),2)
        fixed=preserve_aspects(output,source,1000,500)
        self.assertEqual(aspect_issues(fixed,source),[])
        self.assertEqual(fixed[0]['width'],fixed[0]['height'])
        self.assertEqual(fixed[2]['width'],400)
        self.assertEqual(preserve_aspects(fixed,source,1000,500),fixed)
        self.assertEqual(source['elements'],items)

    def test_foreground_image_uniform_and_background_cover_never_fill(self):
        image=dict(id='image',type='image',x=20,y=20,width=80,height=120,fit='contain')
        source=dict(canvas=dict(width=300,height=300),elements=[image])
        proposed=[dict(image,width=200,height=100)]
        fixed=preserve_aspects(proposed,source,600,200)[0]
        self.assertAlmostEqual(fixed['width']/fixed['height'],2/3)
        crop=[dict(proposed[0],fit='cover')]
        self.assertEqual(preserve_aspects(crop,source,600,200),crop)
        self.assertEqual(preserve_aspects([dict(image,fit='fill')],source,600,200)[0]['fit'],'contain')
        self.assertIn(_background_fit(dict(image,fit='fill'),source,600,200),('cover','contain'))

    def test_assembly_cannot_bypass_circle_constraint(self):
        from engine.resize import _assemble
        circle=dict(id='circle',type='shape',shape_type='ellipse',x=100,y=100,width=100,height=100,opacity=.15)
        source=dict(canvas=dict(width=500,height=500),elements=[circle])
        output=_assemble([dict(circle,width=120,height=220)],[],source,300,600)
        self.assertAlmostEqual(output['elements'][0]['width']/output['elements'][0]['height'],1.)

    def test_graphic_background_preserves_asset_and_does_not_create_matte(self):
        from engine.resize import _assemble
        from unittest.mock import patch
        with TemporaryDirectory() as folder:
            root=Path(folder);im=Image.new('RGB',(500,500),'#204080')
            from PIL import ImageDraw
            ImageDraw.Draw(im).ellipse((200,200,300,300),fill='#ff8800')
            im.save(root/'poster.png')
            image=dict(id='background',type='image',asset_id='poster.png',fit='contain',x=0,y=0,width=500,height=500)
            source=dict(canvas=dict(width=500,height=500),elements=[image])
            with patch('engine.resize._asset_root',return_value=root):
                result=_assemble([],source['elements'],source,600,200)['elements'][0]
            self.assertEqual(result['fit'],'cover')
            self.assertEqual(result['asset_id'],'poster.png')
            self.assertEqual([p.name for p in root.iterdir()],['poster.png'])
            with Image.open(root/'poster.png') as original:
                self.assertEqual(original.size,(500,500))

    def test_first_line_collision_is_not_diluted_by_long_copy(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'solid.png';Image.new('RGBA',(10,25),'red').save(path)
            text=dict(id='copy',type='text',content='MMMMMMMMMM\nMMMMMMMMMM\nMMMMMMMMMM\nMMMMMMMMMM',
                      font_family='Solstice Sans',font_size=24,text_align='left',x=40,y=40,width=300,height=150,z_index=2)
            ink=_text_visible_bounds(text)
            art=dict(id='art',type='image',asset_id=str(path),fit='contain',x=450,y=40,width=5,height=25,z_index=1)
            source=dict(canvas=dict(width=500,height=300),elements=[text,art])
            candidate=[deepcopy(text),dict(art,x=ink[2]-4,y=ink[1])]
            average,peak,pixels=_text_visual_coverage_stats(candidate[0],candidate[1])
            self.assertLess(average,.02)
            self.assertGreater(peak,.01)
            self.assertGreater(pixels,8)
            self.assertEqual(len(art_occlusions(candidate,source)),1)
            _separate_art(candidate,source,500,300)
            self.assertEqual(art_occlusions(candidate,source),[])
            old_contact=dict(art,x=ink[2]-6,y=ink[1])
            reflowed_contact=dict(old_contact,x=old_contact['x']-1)
            supported=dict(source,elements=[deepcopy(text),old_contact])
            self.assertEqual(art_occlusions([deepcopy(text),reflowed_contact],supported),[])
            # The same source-authored overlap remains legal.
            authored=dict(source,elements=[deepcopy(text),dict(art,x=ink[0],y=ink[1],width=100,height=150)])
            self.assertEqual(art_occlusions(authored['elements'],authored),[])


if __name__=='__main__':unittest.main()
