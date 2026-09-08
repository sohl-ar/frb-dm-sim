import numpy as np
import pytest
import torch
from frbsbi.statistics import catalog_statistics,STAT_NAMES,StatisticsBypass
from frbsbi.inference import load_checkpoint
from frbsbi.data import ROOT


def fixture_features():
    x = torch.zeros(1,4,7)
    x[0,:,0] = torch.log10(torch.tensor([100.,200.,300.,400.]))
    x[0,:,4] = torch.log10(torch.tensor([6.,11.,16.,21.]))
    x[0,:,5] = torch.tensor([.1,.2,.3,0.])
    x[0,3,6] = 1
    return x,torch.zeros(1,4,dtype=torch.bool)


def test_statistics_observable_values_masks_and_permutations():
    x,mask = fixture_features()
    s = catalog_statistics(x,mask)
    assert float(s[0,0])==pytest.approx(250.,rel=1e-6)
    assert float(s[0,13])==3
    assert float(s[0,19])==pytest.approx(1000.,rel=1e-5)
    assert float(s[0,21])<1e-4
    order = torch.tensor([3,1,0,2])
    assert torch.allclose(s,catalog_statistics(x[:,order],mask[:,order]),atol=1e-9,rtol=1e-10)
    padded = torch.cat((x,torch.full_like(x,float('nan'))),1)
    pmask = torch.cat((mask,torch.ones_like(mask)),1)
    assert torch.equal(s,catalog_statistics(padded,pmask))
    x[0,3,5] = 12345.  # Deliberately poison a missing-z slot.
    assert torch.equal(s,catalog_statistics(x,mask))


def test_empty_localized_subset_and_singleton():
    x,mask = fixture_features()
    x[:,:,6] = 1
    s = catalog_statistics(x[:,:1],mask[:,:1])
    assert torch.isfinite(s).all() and s[0,13]==0 and s[0,17]==0
    assert s[0,1]==0
    with pytest.raises(ValueError):
        catalog_statistics(x,torch.ones_like(mask))
    with pytest.raises(RuntimeError,match="normalization"):
        StatisticsBypass()(x,mask)


def test_original_checkpoint_still_loads():
    model,_ = load_checkpoint(ROOT/"results/checkpoints/phase2a-best.pt")
    assert model.statistics is None
    assert model.velocity[0].in_features==71
