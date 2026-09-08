"""Deferred normalization regression checks; no dataset fitting or training."""
import numpy as np
import pytest
import torch
from frbsbi.statistics import (STAT_NAMES, BINARY_STAT_NAMES, BINARY_STAT_INDICES,
    NORMALIZATION_POLICY, normalization_parameters, StatisticsBypass)


def report(values):
    center,scale,_ = normalization_parameters(values)
    return {"schema_version":2,"names":STAT_NAMES,"fit_split":"training",
            "encoding_policy":NORMALIZATION_POLICY,"binary_dimensions":BINARY_STAT_NAMES,
            "center":center.tolist(),"scale":scale.tolist()}


def test_rare_and_constant_flags_are_identity_encoded():
    values = np.tile(np.arange(len(STAT_NAMES),dtype=float),(1000,1))
    values[:,list(BINARY_STAT_INDICES)] = 1.
    values[0,list(BINARY_STAT_INDICES)] = 0.
    r = report(values)
    normalized = (values-np.array(r["center"]))/np.array(r["scale"])
    assert np.array_equal(normalized[:,BINARY_STAT_INDICES],values[:,BINARY_STAT_INDICES])
    values[:,list(BINARY_STAT_INDICES)] = 1.
    r = report(values)
    assert all(r["center"][i]==0 and r["scale"][i]==1 for i in BINARY_STAT_INDICES)
    StatisticsBypass().set_normalization(r)
    r["scale"][BINARY_STAT_INDICES[0]] = .1
    with pytest.raises(ValueError,match="identity"):
        StatisticsBypass().set_normalization(r)


def test_legacy_buffers_are_not_reinterpreted_on_load():
    old = StatisticsBypass()
    old.set_normalization({"schema_version":1,"names":STAT_NAMES,"fit_split":"training",
                           "center":[.999]*len(STAT_NAMES),"scale":[.026]*len(STAT_NAMES)})
    restored = StatisticsBypass()
    restored.load_state_dict(old.state_dict(),strict=True)
    assert torch.equal(old.center,restored.center)
    assert torch.equal(old.scale,restored.scale)


def test_continuous_values_keep_zscores():
    values = np.zeros((3,len(STAT_NAMES)))
    values[:,0] = [1.,2.,3.]
    center,scale,_ = normalization_parameters(values)
    assert center[0] == values[:,0].mean()
    assert scale[0] == values[:,0].std()
