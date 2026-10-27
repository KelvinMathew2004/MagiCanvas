"""Source-evidenced negative-space rectangles mapped through actual image fits."""
from copy import deepcopy
from math import ceil
import numpy as np
from PIL import Image,ImageDraw


def image_map(element,asset_size):
    iw,ih=asset_size;w,h=element['width'],element['height'];fit=element.get('fit','contain')
    if fit=='fill':return element['x'],element['y'],w/iw,h/ih
    scale=(max if fit=='cover' else min)(w/iw,h/ih)
    return element['x']+(w-iw*scale)/2,element['y']+(h-ih*scale)/2,scale,scale


def project_rectangle(box,source_image,target_image,asset_size):
    """Source canvas → intrinsic pixels → target canvas (including cover crop)."""
    a,b,sx,sy=image_map(source_image,asset_size);c,d,tx,ty=image_map(target_image,asset_size)
    return (c+(box[0]-a)/sx*tx,d+(box[1]-b)/sy*ty,c+(box[2]-a)/sx*tx,d+(box[3]-b)/sy*ty)


def source_zones(source,background_raster,contrast,units):
    from engine.resize import _analyze,_text_visible_bounds
    sw,sh=source['canvas']['width'],source['canvas']['height'];backgrounds=_analyze(source)[0]
    photos=[e for e in backgrounds if e['type']=='image' and abs(float(e.get('rotation',0)))<.5]
    if len(photos)!=1:return []
    image=background_raster.convert('RGB').copy();image.thumbnail((256,256));pixels=np.asarray(image).astype(float);ih,iw=pixels.shape[:2];result=[]
    copy_ids={e['id'] for u in units if u['role']=='copy' for e in u['items'] if e['type']=='text'}
    for text in source['elements']:
        if text['id'] not in copy_ids or contrast.get(text['id'],{}).get('p05',0)<3.2:continue
        b=_text_visible_bounds(text);l=max(0,int(b[0]/sw*iw));r=min(iw,max(l+1,ceil(b[2]/sw*iw)));t=max(0,int(b[1]/sh*ih));v=min(ih,max(t+1,ceil(b[3]/sh*ih)))
        patch=pixels[t:v,l:r]
        if patch.size==0 or np.std(patch@np.array([.2126,.7152,.0722]))>=22:continue
        color=np.median(patch.reshape(-1,3),axis=0);mask=np.linalg.norm(pixels-color,axis=2)<=35
        yy,xx=np.nonzero(mask[t:v,l:r])
        if len(xx)<.9*patch.shape[0]*patch.shape[1]:continue
        seed=(int(l+xx[len(xx)//2]),int(t+yy[len(yy)//2]));im=Image.fromarray(mask.astype('uint8')*255).copy();ImageDraw.floodfill(im,seed,128,thresh=0);component=np.asarray(im)==128
        # Intersect runs across the lockup's rows, rather than enclosing a
        # concave region whose rectangle could include a pedestal/subject.
        left,right=0,iw
        for y in range(t,v):
            x=seed[0]
            if not component[y,x]:left,right=0,0;break
            lo=hi=x
            while lo>0 and component[y,lo-1]:lo-=1
            while hi+1<iw and component[y,hi+1]:hi+=1
            left=max(left,lo);right=min(right,hi+1)
        if right-left<max(4,r-l):continue
        top,bottom=t,v
        while top>0 and component[top-1,left:right].all():top-=1
        while bottom<ih and component[bottom,left:right].all():bottom+=1
        result.append(dict(text_id=text['id'],background_id=photos[0]['id'],box=(left/iw*sw,top/ih*sh,right/iw*sw,bottom/ih*sh),color=color.tolist()))
    return result


def fitted_variant(elements,source,units,backgrounds,zones,metrics,width,height,assets):
    from engine.qwen_refine import _contrast_text_lockup,build_container_tree
    from engine.resize import _text_visible_bounds,_raw_bounds,_set_font_size
    by={e['id']:e for e in elements};src={e['id']:e for e in source['elements']};target={e['id']:e for e in backgrounds};tree=None
    for zone in zones:
        tid=zone['text_id'];text=by.get(tid);bg=target.get(zone['background_id'])
        if text is None or bg is None:continue
        try:
            with Image.open(assets/src[bg['id']]['asset_id']) as im:size=im.size
        except OSError:continue
        box=project_rectangle(zone['box'],src[bg['id']],bg,size);box=(max(0,box[0]),max(0,box[1]),min(width,box[2]),min(height,box[3]))
        ink=_text_visible_bounds(text);m=metrics.get(tid,{})
        crosses=ink[0]<box[0] or ink[2]>box[2] or ink[1]<box[1] or ink[3]>box[3]
        if not crosses or not (m.get('p05',99)<2.0 or m.get('background_rgb_jump',0)>45):continue
        if tree is None:tree=build_container_tree(source,units)
        ids=_contrast_text_lockup({tid},tree,source)&by.keys()
        # Copy that shared the same clean source column travels together.
        for other in zones:
            if other['background_id']==zone['background_id'] and abs(other['box'][0]-zone['box'][0])<source['canvas']['width']*.08:
                ids|=_contrast_text_lockup({other['text_id']},tree,source)&by.keys()
        pad=.025*min(width,height);availw=box[2]-box[0]-2*pad;availh=box[3]-box[1]-2*pad
        if min(availw,availh)<max(8,min(width,height)*.08):continue
        members=[by[i] for i in ids];l,t,r,b=_raw_bounds(members);scale=min(1.,availw/max(1,r-l),availh/max(1,b-t))
        if scale<.35:continue
        x=box[0]+pad;y=min(max(box[1]+pad,t),box[3]-pad-(b-t)*scale)
        result=deepcopy(elements)
        for e in result:
            if e['id'] not in ids:continue
            e.update(x=x+(e['x']-l)*scale,y=y+(e['y']-t)*scale,width=e['width']*scale,height=e['height']*scale)
            if e['type']=='text':_set_font_size(e,e['font_size']*scale)
        return result,dict(repair_type='source-clean-zone',element_ids=sorted(ids),projected_zone=box,scale=scale)
    return None
