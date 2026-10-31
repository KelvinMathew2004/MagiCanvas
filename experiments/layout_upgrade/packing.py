"""Optional CP-SAT macro packing. Never imported by production by default."""
from copy import deepcopy
from math import ceil, sqrt


def candidates(source, width, height, proposal=None, *, seconds=.1, tight=True):
    from ortools.sat.python import cp_model
    from engine.macro_layout import composites, bounds, _prepared, _ambient_layers, clamp_edge_strips
    from engine.container_flow import measured_block
    from engine.source_structure import _transform
    prepared = _prepared(source)
    spec = composites(prepared, tight=tight)
    blocks = spec['blocks']
    if not 2 <= len(blocks) <= 12:
        return []
    margin = max(2, ceil(.045 * min(width, height)))
    gap = max(2, ceil(.025 * min(width, height)))
    available_w, available_h = width - 2*margin, height - 2*margin
    if min(available_w, available_h) <= 0:
        return []
    source_w, source_h = source['canvas']['width'], source['canvas']['height']
    desired = sqrt(width*height/(source_w*source_h))
    variants = []
    for block in blocks:
        options = []; seen = set()
        for fraction in (1., .66, .5, .33, .25):
            for font_fraction in (1., .8, .6):
                elements = measured_block(block, available_w*fraction, available_h, source,
                                          desired*font_fraction, (width, height))
                if not elements: continue
                b = bounds(elements); w, h = ceil(b[2]-b[0]), ceil(b[3]-b[1])
                if min(w,h)<=0 or w>available_w or h>available_h: continue
                key = (w,h,round(font_fraction,2))
                if key in seen: continue
                seen.add(key)
                local = _transform(elements, 1., 0., 0., b)
                # Prefer preserved type hierarchy and actual hero area, not a
                # bare union box which can reward empty cards.
                originals = {e['id']:e for e in block}
                type_ratios = [e['font_size']/max(1.,originals[e['id']]['font_size']*desired)
                               for e in local if e['type']=='text']
                media = sum(e['width']*e['height'] for e in local if e['type']=='image')/(width*height)
                score = int(10000*min(type_ratios,default=font_fraction) + 4000*media)
                options.append(dict(width=w,height=h,elements=local,score=score))
        if not options: return []
        variants.append(options)
    model = cp_model.CpModel(); xs=[]; ys=[]; choices=[]; intervals_x=[]; intervals_y=[]; rewards=[]; drifts=[]
    predicted = {e['id']:e for e in proposal or []}
    for i,(block,options) in enumerate(zip(blocks,variants)):
        x = model.new_int_var(margin, width-margin, f'x{i}')
        y = model.new_int_var(margin, height-margin, f'y{i}')
        xs.append(x);ys.append(y);active=[]
        for j,v in enumerate(options):
            use = model.new_bool_var(f'mode{i}_{j}');active.append(use)
            model.add(x+v['width']<=width-margin).only_enforce_if(use)
            model.add(y+v['height']<=height-margin).only_enforce_if(use)
            # Inflate only the free right/bottom by the same clearance.
            intervals_x.append(model.new_optional_fixed_size_interval_var(x,v['width']+gap,use,f'ix{i}_{j}'))
            intervals_y.append(model.new_optional_fixed_size_interval_var(y,v['height']+gap,use,f'iy{i}_{j}'))
            rewards.append(v['score']*use)
        model.add_exactly_one(active);choices.append(active)
        anchors=[predicted[e['id']] for e in block if e['id'] in predicted]
        b=bounds(anchors or block)
        cx=(b[0]+b[2])/2 if anchors else (b[0]+b[2])/2/source_w*width
        cy=(b[1]+b[3])/2 if anchors else (b[1]+b[3])/2/source_h*height
        selected_w=sum(v['width']*use for v,use in zip(options,active))
        selected_h=sum(v['height']*use for v,use in zip(options,active))
        dx=model.new_int_var(0,4*width,f'dx{i}');dy=model.new_int_var(0,4*height,f'dy{i}')
        model.add_abs_equality(dx,2*x+selected_w-int(2*cx))
        model.add_abs_equality(dy,2*y+selected_h-int(2*cy))
        drifts.append(dx*max(1,1000//width)+dy*max(1,1000//height))
    model.add_no_overlap_2d(intervals_x,intervals_y)
    model.maximize(sum(rewards)-sum(drifts))
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=1;solver.parameters.random_seed=0
    status=solver.solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return []
    output=[]
    for i,options in enumerate(variants):
        v=next(v for v,use in zip(options,choices[i]) if solver.value(use))
        for e in deepcopy(v['elements']):
            e['x']+=solver.value(xs[i]);e['y']+=solver.value(ys[i]);output.append(e)
    output += _ambient_layers(source,width,height,spec.get('ambient_ids',[]))
    output += [deepcopy(e) for e in source['elements'] if e['id'] in spec['edge_ids']]
    clamp_edge_strips(output,source,width,height)
    # Reuse the existing painted-label invariant after rigid group placement;
    # a solved group box alone does not center its actual glyph ink.
    from engine.layout_contracts import center_control_labels
    center_control_labels(output,source,width,height)
    if {e['id'] for e in output} != {e['id'] for block in blocks for e in block} | set(spec.get('ambient_ids',[])) | set(spec['edge_ids']):
        return []
    return [dict(name=f'experimental-cpsat-{tight}-{seconds}',elements=output,energy=0.,
                 macro_locked=True,source_attached=True,macro_links=spec['links'],
                 solver=dict(status=solver.status_name(status),seconds=solver.wall_time,
                             objective=solver.objective_value,blocks=len(blocks)))]
