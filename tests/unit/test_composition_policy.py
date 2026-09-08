"""Deferred regression checks; no physical catalogs or learned model required."""
import numpy as np
import pytest
from frbsbi.composition import composition_plan, expanded_mask, GROUP_NAMES, quota_counts


@pytest.mark.parametrize("start,total", [(0,50000),(50000,5000),(55000,5000)])
def test_exact_split_quota_and_immutable_plan(start,total):
    plan = composition_plan(start,start+total)
    assert {name:int(np.count_nonzero(plan==i)) for i,name in enumerate(GROUP_NAMES)} == quota_counts(total)
    assert not plan.flags.writeable


def test_realized_mask_contract_across_all_sizes():
    plan = composition_plan(0,50000)
    for group in ("near_localized","all_localized"):
        seed = int(np.flatnonzero(plan==GROUP_NAMES.index(group))[0])
        for n in range(1,513):
            uniforms = np.random.default_rng(n).random(n)
            mask,assigned = expanded_mask(seed,0,50000,uniforms,.25)
            assert assigned == group
            assert 10*int(mask.sum()) >= 9*n
            if group == "all_localized":
                assert mask.all()
            assert np.array_equal(mask,expanded_mask(seed,0,50000,uniforms,.25)[0])


def test_beta_group_preserves_original_mask():
    seed = int(np.flatnonzero(composition_plan(0,50000)==0)[0])
    u = np.array([.01,.24,.25,.99])
    mask,group = expanded_mask(seed,0,50000,u,.25)
    assert group == "beta"
    assert np.array_equal(mask,u<.25)
