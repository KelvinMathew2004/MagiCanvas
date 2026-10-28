import numpy as np
import pytest
from engine.structural_ranker import fit,score,FEATURES


def test_fixed_bt_fit_generalizes_known_signal_and_keeps_tied_features_finite():
    rng=np.random.default_rng(13);pairs=[]
    for i in range(80):
        a,b=rng.normal(size=(2,10));a[2]=b[2]=1.
        if a[0]+a[1]<b[0]+b[1]:a,b=b,a
        pairs.append([a,b])
    model=fit(pairs[:60]);correct=sum(score(a,model)>score(b,model) for a,b in pairs[60:])
    assert correct>=17
    assert model['scale'][2]>=1e-3 and model['weights'][2]==0
    assert model['center']==pytest.approx(np.asarray(pairs[:60]).mean(axis=(0,1)))
    assert abs(model['weights'][0])>abs(model['weights'][9])
    with pytest.raises(ValueError):score([0]*10,dict(model,features=['wrong']))


def test_ranker_does_not_ship_default_weights_or_enable_itself():
    from pathlib import Path
    assert not Path('engine/data/structural_ranker.json').exists()
    assert len(FEATURES)==10
