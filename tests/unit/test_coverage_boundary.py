"""Regression checks for the unchanged inclusive +/-2pp SBC criterion."""
import json
import numpy as np
import pytest
from frbsbi.data import ROOT
from frbsbi.evaluate import coverage_in_band,sbc_statistics


@pytest.mark.parametrize("nominal,lo,hi",[(.68,660,700),(.95,930,970)])
def test_exact_endpoints_and_immediate_outside(nominal,lo,hi):
    assert coverage_in_band(lo,1000,nominal)
    assert coverage_in_band(hi,1000,nominal)
    assert not coverage_in_band(lo-1,1000,nominal)
    assert not coverage_in_band(hi+1,1000,nominal)


def test_no_false_classification_for_every_attainable_count():
    # Integer arithmetic oracle for the prescribed 1,000-trial experiment.
    for covered in range(1001):
        assert coverage_in_band(covered,1000,.68)==(660<=covered<=700)
        assert coverage_in_band(covered,1000,.95)==(930<=covered<=970)
    assert coverage_in_band(2,3,.68)
    with pytest.raises(ValueError):
        coverage_in_band(0,0,.68)


def test_saved_sbc_only_fd_status_changes():
    root = ROOT/"results/conditioning-v1"
    old = json.loads((root/"authorized-followup/results.json").read_text())["G-P4"]
    with np.load(root/"posterior-artifacts/sbc.npz",allow_pickle=False) as d:
        new,ranks = sbc_statistics(d["posterior"],d["truth"],jitter_seed=old["rank_jitter_seed"])
    assert ranks.tolist()==old["raw_ranks"]
    changed = []
    for a,b in zip(old["parameters"],new):
        assert {k:v for k,v in a.items() if k!="status"}=={k:v for k,v in b.items() if k!="status"}
        if a["status"]!=b["status"]:
            changed.append(b["parameter"])
    assert changed==["f_d"]
    assert new[0]["status"]=="pass"
