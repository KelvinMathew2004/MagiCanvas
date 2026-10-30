"""Visible artwork bounds for layout analysis; supplied rendering stays unchanged."""
from functools import lru_cache
from math import cos, radians, sin
from PIL import Image
from renderer.converter.assets import asset_path


@lru_cache(maxsize=512)
def _alpha_bbox(path: str):
    """Normalized visible alpha bounds for an asset, independent of its file canvas."""
    try:
        image=Image.open(path).convert("RGBA")
        alpha=image.getchannel("A")
        # Ignore almost transparent resampling noise at the edges.
        alpha=alpha.point(lambda value:255 if value>=8 else 0)
        bbox=alpha.getbbox()
        if not bbox:
            return (0.,0.,0.,0.)
        left,top,right,bottom=bbox
        return (left/image.width,top/image.height,right/image.width,bottom/image.height)
    except (OSError,ValueError):
        return None



@lru_cache(maxsize=512)
def _asset_size(path: str):
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError,ValueError):
        return None



def visible_content_bounds(element, assets_dir):
    """Return rendered content bounds, excluding transparent asset margins."""
    x,y,w,h=(float(element.get(key,0)) for key in ("x","y","width","height"))
    if float(element.get("opacity",1))<=0:
        return (x,y,x,y)
    left,top,right,bottom=0.,0.,1.,1.
    if element.get("type")=="image":
        path=asset_path(str(assets_dir) if assets_dir else None,element.get("asset_id"))
        bbox=_alpha_bbox(str(path)) if path else None
        if bbox:
            il,it,ir,ib=bbox
            fit=element.get("fit","contain")
            if fit=="fill":
                left,top,right,bottom=il,it,ir,ib
            elif fit=="cover":
                size=_asset_size(str(path))
                if not size:
                    return (x,y,x+w,y+h)
                iw,ih=size
                scale=max(w/iw,h/ih)
                rendered_w,rendered_h=iw*scale,ih*scale
                crop_x=(rendered_w-w)/2;crop_y=(rendered_h-h)/2
                left=max(0.,(il*rendered_w-crop_x)/w)
                top=max(0.,(it*rendered_h-crop_y)/h)
                right=min(1.,(ir*rendered_w-crop_x)/w)
                bottom=min(1.,(ib*rendered_h-crop_y)/h)
                if right<=left or bottom<=top:
                    return (x,y,x,y)
            else:
                size=_asset_size(str(path))
                if not size:
                    return (x,y,x+w,y+h)
                iw,ih=size
                scale=min(w/iw,h/ih)
                rendered_w,rendered_h=iw*scale,ih*scale
                offset_x=(w-rendered_w)/2;offset_y=(h-rendered_h)/2
                left=(offset_x+il*rendered_w)/w
                top=(offset_y+it*rendered_h)/h
                right=(offset_x+ir*rendered_w)/w
                bottom=(offset_y+ib*rendered_h)/h
    vx,vy,vr,vb=x+left*w,y+top*h,x+right*w,y+bottom*h
    angle=radians(float(element.get("rotation",0)))
    if angle:
        cx,cy=x+w/2,y+h/2
        rw,rh=vr-vx,vb-vy
        ccx,ccy=(vx+vr)/2,(vy+vb)/2
        rx,ry=ccx-cx,ccy-cy
        rcx=cx+rx*cos(angle)-ry*sin(angle)
        rcy=cy+rx*sin(angle)+ry*cos(angle)
        extent_w=abs(rw*cos(angle))+abs(rh*sin(angle))
        extent_h=abs(rw*sin(angle))+abs(rh*cos(angle))
        return (rcx-extent_w/2,rcy-extent_h/2,rcx+extent_w/2,rcy+extent_h/2)
    return (vx,vy,vr,vb)

