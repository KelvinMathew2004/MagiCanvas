from pathlib import Path
from functools import lru_cache
from math import cos, radians, sin
from typing import Optional

from PIL import Image, ImageDraw


def asset_path(assets_dir: Optional[str], asset_id: Optional[str]) -> Optional[Path]:
    if not assets_dir or not asset_id:
        return None
    candidate = Path(assets_dir) / asset_id
    return candidate if candidate.is_file() else None


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


def load_asset_image(
    assets_dir: Optional[str],
    asset_id: Optional[str],
    width: float,
    height: float,
    fit: str = "contain",
) -> Image.Image:
    """Load an image element's asset, or a placeholder if it can't be found.

    `fit` (`contain`/`cover`/`fill`, per the v0.9 schema) is emulated here
    because cr-renderer itself just stretches whatever image it's given
    into the element's box - it has no notion of aspect-preserving fit.
    """
    path = asset_path(assets_dir, asset_id)
    if path is None:
        return _placeholder(asset_id, width, height)
    image = Image.open(path).convert("RGBA")
    return _fit_image(image, width, height, fit)


def _placeholder(asset_id: Optional[str], width: float, height: float) -> Image.Image:
    w, h = max(1, round(width)), max(1, round(height))
    image = Image.new("RGBA", (w, h), (214, 214, 214, 255))
    draw = ImageDraw.Draw(image)
    border = max(2, min(w, h) // 100)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(150, 40, 40, 255), width=border)
    label = f"missing asset:\n{asset_id or '(none)'}"
    try:
        draw.multiline_text((w / 2, h / 2), label, fill=(120, 20, 20, 255), anchor="mm", align="center")
    except (TypeError, ValueError):
        # Older Pillow without multiline `anchor` support - fall back silently.
        draw.text((4, 4), label, fill=(120, 20, 20, 255))
    return image


def _fit_image(image: Image.Image, width: float, height: float, fit: str) -> Image.Image:
    w, h = max(1, round(width)), max(1, round(height))
    if fit == "contain":
        scale = min(w / image.width, h / image.height)
        nw, nh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        resized = image.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2), resized)
        return canvas
    if fit == "cover":
        scale = max(w / image.width, h / image.height)
        nw, nh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        resized = image.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
    # fit == "fill" (or anything unrecognized): stretch to exactly fit, same
    # as cr-renderer's own default `drawImageRect` behavior.
    return image.resize((w, h), Image.LANCZOS)
