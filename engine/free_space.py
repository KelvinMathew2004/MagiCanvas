"""Conservative grid free-space proposals; exact rendered gates still apply.

Histogram/monotone-stack maximal rectangles on at most 32x32 cells. Cells
intersecting occupied geometry are fully blocked, so quantization loses small
free pockets but cannot invent empty space. No fallback into occupied cells.
"""
from math import ceil,floor,hypot


def free_rectangles(width,height,occupied,*,cells=32,margin=0.,padding=0.):
    if width<=0 or height<=0 or not 1<=cells<=64:return []
    dx,dy=width/cells,height/cells;grid=[[False]*cells for _ in range(cells)]
    for x,y,r,b in occupied:
        if r<=x or b<=y:continue
        for row in range(max(0,floor((y-padding)/dy)),min(cells,ceil((b+padding)/dy))):
            for col in range(max(0,floor((x-padding)/dx)),min(cells,ceil((r+padding)/dx))):grid[row][col]=True
    heights=[0]*cells;found=set()
    for row in range(cells):
        heights=[0 if grid[row][col] else heights[col]+1 for col in range(cells)];stack=[]
        for col in range(cells+1):
            current=heights[col] if col<cells else 0;start=col
            while stack and stack[-1][1]>=current:
                left,h=stack.pop();start=left
                if not h:continue
                top,bottom=row-h+1,row+1
                # Discard rectangles extendible in any direction. This avoids
                # quadratic containment pruning while retaining all maximal ones.
                if bottom<cells and not any(grid[bottom][left:col]):continue
                if top and not any(grid[top-1][left:col]):continue
                if left and not any(grid[r][left-1] for r in range(top,bottom)):continue
                if col<cells and not any(grid[r][col] for r in range(top,bottom)):continue
                found.add((left,top,col,bottom))
            stack.append((start,current))
    result=[]
    for left,top,right,bottom in found:
        b=(max(margin,left*dx),max(margin,top*dy),min(width-margin,right*dx),min(height-margin,bottom*dy))
        if b[2]>b[0] and b[3]>b[1]:result.append(b)
    return sorted(set(result),key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),b))


def place(box,rectangles,anchor,canvas,*,scale_cap=1.):
    """Maximize readable scale minus normalized anchor drift; stable ties."""
    bw,bh=box;w,h=canvas;options=[]
    if min(bw,bh,w,h)<=0:return None
    for rect in rectangles:
        s=min(scale_cap,(rect[2]-rect[0])/bw,(rect[3]-rect[1])/bh)
        if s<=0:continue
        x=max(rect[0],min(rect[2]-bw*s,anchor[0]-bw*s/2));y=max(rect[1],min(rect[3]-bh*s,anchor[1]-bh*s/2))
        drift=hypot(x+bw*s/2-anchor[0],y+bh*s/2-anchor[1])/hypot(w,h)
        options.append((-(s-.25*drift),-s,drift,x,y,s))
    if not options:return None
    _,_,drift,x,y,s=min(options)
    return dict(x=x,y=y,scale=s,anchor_drift=drift)
