"""Vectorized equivalents of the selector's original glyph-mask statistics."""
import numpy as np


def luminance(rgb):
    # Preserve the historical selector gamma (2.0), including source floors.
    # A standards-based gamma migration needs its own calibrated ablation.
    values=np.asarray(rgb,dtype=np.float64)/255.
    return np.sum(np.where(values<=.04045,values/12.92,((values+.055)/1.055)**2)*np.array([.2126,.7152,.0722]),axis=-1)


def glyph_statistics(background,mask,color,opacity=1.):
    pixels=np.asarray(background,dtype=np.uint8).reshape(-1,3)
    active=pixels[np.asarray(mask).reshape(-1)>12]
    if not len(active):return None
    alpha=color[3]*opacity
    fg=np.array(color[:3],dtype=np.float64)*alpha+active.astype(np.float64)*(1-alpha)
    a,b=luminance(fg),luminance(active)
    contrasts=np.sort((np.maximum(a,b)+.05)/(np.minimum(a,b)+.05))
    # Match old sorted-list order statistics, not interpolated percentiles.
    indices=np.arange(1,len(b));rough=float(np.abs(np.diff(b))[indices%8!=0].sum()/56.)
    return dict(minimum=float(contrasts[0]),lower_quintile=float(contrasts[len(contrasts)//5]),
                p05=float(contrasts[len(contrasts)//20]),ink_pixels=len(active),roughness=rough,
                luma_std=float(b.std()),background_rgb_jump=float(np.ptp(pixels,axis=0).max()))


def glyph_overflow(source,elements,width,height):
    """Exact shared glyph/frame/canvas overflow, retaining source bearings."""
    from engine.resize import _render_text_ink,_source_ink_overhang
    originals={e['id']:e for e in source['elements']};overflow={}
    for e in elements:
        if e.get('type')!='text':continue
        ink=_render_text_ink(e)
        if ink is None:continue
        bx,by=_source_ink_overhang(e,originals.get(e['id']))
        amount=max(0.,-ink[0]-bx,-ink[1]-by,ink[2]-round(e['width']),ink[3]-round(e['height']),
                   -e['x']-ink[0],-e['y']-ink[1],e['x']+ink[2]-width,e['y']+ink[3]-height)
        if amount>.01:overflow[e['id']]=round(amount,3)
    return overflow
