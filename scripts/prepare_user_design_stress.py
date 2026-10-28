#!/usr/bin/env python3
"""Make disclosed, asset-backed test copies; never rewrite the supplied originals.

Fixture authoring only: synthetic artwork, font substitution and duplicate-ID
repair here are NOT inference-time engine behavior or training examples.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from collections import Counter
import json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
WORK=ROOT/'data/user_design_stress'
DEST=ROOT/'tests/fixtures/user_level3_sources'


def prepare():
    from PIL import Image,ImageDraw,ImageFont
    from viewer.server import validate_task
    assets=WORK/'assets';assets.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    font=ROOT/'brand_kit/fonts/SolsticeSans-Bold.ttf'
    photos=[ROOT/'data/crello_train'/p for p in (
        'assets/5d9ca336abc8ea6d1c09c4e6/element_000.png',
        'assets/59b7aef31350e8329300f57a/element_000.png',
        'assets/592e79d595a7a863ddcda680/element_000.png',
        'assets/590afa7495a7a863ddcd6df1/element_000.png',
        'assets/5f982f25a637ee11e3dc9176/element_000.png')]
    themes=[('NORTHWIND','#12291f','#ffd36e'),('MERIDIAN','#141329','#c7bbff'),('FENNEL & FIG','#f5efdf','#6c793c'),('LUMEN','#160f28','#ff80c7')]
    audits=[]
    for index,path in enumerate(sorted((WORK/'originals').glob('*.json'))):
        original=json.loads(path.read_text());task=deepcopy(original);source=task['source'];brand,bg,ink=themes[index]
        changes=[]
        try:validate_task(original);original_error=None
        except ValueError as exc:original_error=str(exc)
        duplicates=Counter(e['id'] for e in source['elements']);seen=Counter()
        for e in source['elements']:
            old_id=e['id'];seen[old_id]+=1
            if seen[old_id]>1:
                e['id']=old_id+'__authored_'+str(seen[old_id]);changes.append(dict(kind='duplicate_id_in_test_copy',before=old_id,after=e['id']))
            if e['type']=='text':
                old=e['font_family'];e['font_family']='Solstice Serif' if any(k in old.lower() for k in ('display','serif')) else 'Solstice Sans'
                changes.append(dict(kind='font_substitution',id=e['id'],before=old,after=e['font_family']))
            if e['type']!='image':continue
            filename=f'u{index+2:02d}_{e["id"]}.png';dest=assets/filename;old=e['asset_id']
            # All generated art is fixture-owned, never a modified input asset.
            if e['id']=='bg':
                Image.new('RGB',(600,600),bg).save(dest)
            elif e['id']=='logo' or e['id'] in ('wine_icon','seal'):
                im=Image.new('RGBA',(600,220),(0,0,0,0));d=ImageDraw.Draw(im)
                label=brand if e['id']=='logo' else 'WINE' if e['id']=='wine_icon' else 'SEASONAL'
                f=ImageFont.truetype(str(font),52 if len(label)<13 else 42)
                d.text((300,110),label,font=f,fill=ink,anchor='mm');im.save(dest)
            else:
                photo=photos[0 if index==0 else 2 if index==1 else 4 if index==2 else 3]
                if index==0 and e['id']!='hero':photo=photos[1]
                if not photo.is_file():raise FileNotFoundError(photo)
                shutil.copy2(photo,dest)
            e['asset_id']='../../data/user_design_stress/assets/'+filename
            changes.append(dict(kind='authorized_asset_substitution',id=e['id'],before=old,after=e['asset_id'],sha256=sha256(dest.read_bytes()).hexdigest()))
        validate_task(task)
        source['name']=brand+' — supplied complex test (asset-backed copy)'
        key=f'u{index+2:02d}'
        targets=[dict(name='requested',**task['target_canvas']),dict(name='skyscraper',width=160,height=600)]
        fixture=dict(title=source['name'],source=source,targets=targets,original_task=original,
                     provenance='User-supplied authored stress fixture, not an official sample or hidden test. Local artwork/font substitutions authorized. All original geometry/content retained; duplicate ID renamed only in this test copy.',
                     authoring_changes=changes,original_validation_error=original_error)
        (DEST/(key+'.json')).write_text(json.dumps(fixture,indent=2,ensure_ascii=False)+'\n')
        audits.append(dict(key=key,original_file=path.name,original_sha256=sha256(path.read_bytes()).hexdigest(),elements=len(source['elements']),original_validation_error=original_error,duplicate_ids=[k for k,v in duplicates.items() if v>1],changes=changes))
    if len(audits)!=4:raise ValueError('Expected exactly four supplied originals')
    (WORK/'source_integrity.json').write_text(json.dumps(audits,indent=2,ensure_ascii=False)+'\n')
    return audits


if __name__=='__main__':
    print(json.dumps([dict(key=r['key'],elements=r['elements'],original_validation_error=r['original_validation_error']) for r in prepare()]))
