"""Synthetic controls of the statistics code, not evidence about the network.

Exact grids make these deterministic unit controls. The actual SBC gate
uses independent simulator catalogs and independent random posterior draws.
"""
import numpy as np
import pytest
from frbsbi.evaluate import sbc_statistics


def test_sbc_control_and_parameter_specific_miscalibration():
    grid = (np.arange(1000)+.5)/1000
    posterior = np.broadcast_to(grid[None,:,None],(1000,1000,4)).copy()
    truth = np.broadcast_to(grid[:,None],(1000,4)).copy()
    good,_ = sbc_statistics(posterior,truth,jitter_seed=74000)
    assert all(p["status"]=="pass" for p in good)
    # Collapse only the last parameter: this must fail that parameter,
    # leaving the other coordinates unaffected by any axis mix-up.
    posterior[:,:,3] = .5
    bad,_ = sbc_statistics(posterior,truth,jitter_seed=74000)
    assert all(p["status"]=="pass" for p in bad[:3])
    assert bad[3]["status"]=="fail"
    assert bad[3]["coverage68"]==0 and bad[3]["coverage95"]==0


def test_sbc_rejects_nonfinite_inputs():
    with pytest.raises(ValueError,match="finite"):
        sbc_statistics(np.full((3,4,4),np.nan),np.zeros((3,4)),jitter_seed=74001)
