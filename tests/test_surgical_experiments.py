"""Mathematical properties and counterexamples for opt-in research only."""
from copy import deepcopy
import random
from experiments.layout_upgrade.surgical import dual_envelope,envelope_violation,gated_critic,topology


def test_dual_envelope_random_rotations_preserve_aspect_and_source():
    rng=random.Random(173);elements=[]
    for i in range(500):
        elements.append(dict(id=str(i),type='image',x=rng.uniform(-400,800),y=rng.uniform(-300,500),
                             width=rng.uniform(1,1400),height=rng.uniform(1,800),rotation=rng.uniform(-360,360)))
    old=deepcopy(elements);out=dual_envelope(elements,728,90)
    assert elements==old
    for a,b in zip(elements,out):
        assert envelope_violation(b,728,90)<1e-10
        assert abs(a['width']/a['height']-b['width']/b['height'])<1e-10


def test_zero_size_is_not_silently_made_visible():
    a=dict(id='zero',type='image',x=0.,y=0.,width=0.,height=50.)
    assert dual_envelope([a],100,100)==[a]


def test_global_margin_and_aspect_gate_have_exact_boundaries():
    assert gated_critic([1.,.9],[.1,2.],(100,100))==([1.,.9],[])
    assert gated_critic([1.,.8],[.1,2.],(400,100))==([1.,.8],[])
    assert gated_critic([1.,.9],[.1,2.],(400,100))==([.1,2.],[0,1])


def test_top2_mixed_scale_can_promote_unrescored_third_candidate():
    scores,ids=gated_critic([1.,.9,.8],[.1,.2,9.],(400,100))
    assert ids==[0,1] and max(range(3),key=scores.__getitem__)==2


def test_overlap_does_not_prove_ownership():
    source=dict(canvas=dict(width=100,height=100),elements=[
        dict(id='photo',type='image',x=0.,y=0.,width=100.,height=100.,z_index=0),
        dict(id='caption',type='text',content='Hello',font_size=8.,x=10.,y=10.,width=30.,height=10.,z_index=1)])
    target=deepcopy(source);target['elements'][0].update(width=40.,height=40.)
    target['elements'][1].update(x=60.)
    assert not topology(source,target)['passes']
    assert topology(source,target,True)['passes']


def test_raw_font_recovery_uses_bounded_empty_width_and_retains_anchor():
    from engine.resize import _text_info,_set_font_size
    from experiments.layout_upgrade.surgical import font_slack
    e=dict(id='t',type='text',content='Continue learning today',font_family='Solstice Sans',font_size=30.,
        x=100.,y=100.,width=500.,height=60.,text_align='center',letter_spacing=.6,z_index=1)
    e['width']=_text_info(e,1e9)[1]+2
    source=dict(canvas=dict(width=900,height=400),elements=[e]);p=deepcopy(e)
    p['width']*=.37;p['height']=45.;_set_font_size(p,10.)
    before=deepcopy(p);anchor=p['x']+p['width']/2
    out,notes=font_slack([p],source,900,400,{'t':13.})
    assert p==before and notes
    assert out[0]['font_size']==12.5 and out[0]['width']<=before['width']*1.18
    assert abs(out[0]['x']+out[0]['width']/2-anchor)<1e-8
    assert _text_info(out[0])[2]==_text_info(e)[2]
    assert abs(out[0]['letter_spacing']/out[0]['font_size']-.6/30.)<1e-8
