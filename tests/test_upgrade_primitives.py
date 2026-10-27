from copy import deepcopy
import json
import numpy as np
import pytest
from engine.raster_metrics import glyph_statistics
from engine.resize import _luminance
from engine.proposal_completion import free_rectangles,complete_missing
from engine.render_cache import DiagnosticRenderCache
from engine.container_flow import arrange


def test_vector_statistics_match_scalar_pixels_and_order_statistics():
    rng=np.random.default_rng(7);rgb=rng.integers(0,256,(23,31,3),dtype=np.uint8);mask=rng.integers(0,255,(23,31),dtype=np.uint8);color=(212,71,143,.72);opacity=.83
    got=glyph_statistics(rgb,mask,color,opacity);active=rgb[mask>12];ratios=[];lumas=[]
    for p in active:
        alpha=color[3]*opacity;fg=tuple(color[k]*alpha+float(p[k])*(1-alpha) for k in range(3));a,b=_luminance(fg),_luminance(p)
        ratios.append((max(a,b)+.05)/(min(a,b)+.05));lumas.append(b)
    ratios.sort()
    assert got['lower_quintile']==pytest.approx(ratios[len(ratios)//5],abs=1e-12)
    assert got['p05']==pytest.approx(ratios[len(ratios)//20],abs=1e-12)
    assert got['roughness']==pytest.approx(sum(abs(lumas[j]-lumas[j-1]) for j in range(1,len(lumas)) if j%8)/56,abs=1e-12)
    assert glyph_statistics(rgb,np.zeros((23,31)),color) is None


def test_exact_render_cache_retains_style_layer_and_fractional_differences(tmp_path):
    calls=[]
    def render(doc,assets,path,max_size):calls.append(deepcopy(doc));path.write_text(json.dumps(doc));return ['warning']
    doc=dict(canvas={'width':100,'height':100},elements=[{'id':'a','x':1.,'font_size':12,'color':'red'},{'id':'b'}])
    with DiagnosticRenderCache(render,tmp_path) as cache:
        cache.render(doc,tmp_path/'one');cache.render(deepcopy(doc),tmp_path/'two')
        for key,value in [('x',1.00001),('font_size',12.1),('color','blue')]:
            other=deepcopy(doc);other['elements'][0][key]=value;cache.render(other,tmp_path/key)
        reverse=deepcopy(doc);reverse['elements'].reverse();cache.render(reverse,tmp_path/'reverse')
        assert cache.stats['cache_hits']==1 and len(calls)==5


def test_free_rectangle_completion_has_no_occupied_overlap():
    boxes=[(0,0,60,100),(60,0,100,40)];rects=free_rectangles(100,100,boxes)
    assert rects and all(not(max(q[0],b[0])<min(q[2],b[2]) and max(q[1],b[1])<min(q[3],b[3])) for q in rects for b in boxes)


def test_completion_preserves_raw_input_and_has_bounded_coverage():
    es=[dict(id=str(i),type='shape',shape_type='rectangle',x=i*20.,y=20.,width=10.,height=10.,z_index=i) for i in range(20)]
    src=dict(canvas=dict(width=400,height=200),elements=es);raw={e['id']:{k:e[k] for k in ('x','y','width','height')} for e in es[:-2]};old=deepcopy(raw)
    completed,notes=complete_missing(src,raw,es,src['canvas'])
    assert raw==old and set(completed)=={e['id'] for e in es}
    assert len(notes)==2
    assert complete_missing(src,{k:v for k,v in raw.items() if int(k)<16},es,src['canvas'])[0] is None


def test_grid_maximizes_legible_uniform_scale_without_severing_cards():
    blocks=[]
    for i in range(4):
        blocks.append([dict(id=f'p{i}',type='shape',x=0,y=i*70,width=100,height=50),dict(id=f't{i}',type='text',x=10,y=i*70+10,width=80,height=20,font_size=14,letter_spacing=2)])
    old=deepcopy(blocks);items,report=arrange(blocks,(0,0,500,130),dict(canvas=dict(width=150,height=300),elements=sum(blocks,[])))
    assert blocks==old and report['columns']>=2
    by={e['id']:e for e in items}
    for i in range(4):
        p,t=by[f'p{i}'],by[f't{i}'];assert (t['x']-p['x'])/p['width']==pytest.approx(.1)
        assert p['width']/p['height']==pytest.approx(2)
        assert t['letter_spacing']/t['font_size']==pytest.approx(2/14)


def test_runtime_capabilities_and_disable_override(monkeypatch):
    from types import SimpleNamespace as NS
    from engine.runtime import select_runtime
    fake=NS(backends=NS(mps=NS(is_available=lambda:True)),cuda=NS(is_available=lambda:False,is_bf16_supported=lambda:False),float16='f16',float32='f32',bfloat16='bf16',get_num_threads=lambda:2)
    monkeypatch.delenv('MAGICANVAS_QWEN_DEVICE',raising=False);monkeypatch.delenv('MAGICANVAS_CPU_THREADS',raising=False)
    assert select_runtime(fake).device=='mps'
    monkeypatch.setenv('MAGICANVAS_DISABLE_MPS','1')
    assert select_runtime(fake).dtype=='f32'
    assert select_runtime(fake,'vision').device=='cpu'
    monkeypatch.setenv('MAGICANVAS_QWEN_DEVICE','cuda')
    with pytest.raises(ValueError):select_runtime(fake)


def test_adapter_verification_rejects_mismatched_hidden_width(tmp_path):
    import torch
    from safetensors.torch import save_file
    from scripts.verify_qwen_adapter import verify
    adapter=tmp_path/'a';model=tmp_path/'m';adapter.mkdir();model.mkdir()
    (adapter/'adapter_config.json').write_text(json.dumps(dict(peft_type='LORA',r=2,base_model_name_or_path='Qwen/Qwen3-test')))
    (adapter/'magicanvas_prompt_version.txt').write_text('2')
    save_file({'base.q_proj.lora_A.weight':torch.zeros((2,4))},adapter/'adapter_model.safetensors')
    (model/'config.json').write_text(json.dumps(dict(model_type='qwen3',hidden_size=4)))
    assert verify(adapter,model)['checked_q_projections']==1
    (model/'config.json').write_text(json.dumps(dict(model_type='qwen3',hidden_size=8)))
    with pytest.raises(ValueError,match='hidden width'):verify(adapter,model)


def test_repeated_flow_does_not_treat_headline_body_footer_as_peer_cards():
    from engine.container_flow import variants
    source=dict(canvas=dict(width=400,height=400),elements=[dict(id=str(i),type='text',content='Words',font_family='Arial',font_size=size,text_align='left',x=10,y=i*100+10,width=100,height=30) for i,size in enumerate((32,18,10))])
    assert variants(deepcopy(source['elements']),source)==[]
