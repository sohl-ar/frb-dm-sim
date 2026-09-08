"""Tiny wrapper and ledger controls; never load/run the trained gate suite."""
import json
import types
import numpy as np
import pytest
from frbsbi import acceptance
from frbsbi.acceptance import Acceptance,empty_ledger
from frbsbi.gate_metrics import prior_quantiles
from frbsbi.train import write_json


def bare_runner(tmp_path,monkeypatch):
    monkeypatch.setattr(acceptance,"ROOT",tmp_path)
    (tmp_path/"results").mkdir()
    obj = Acceptance.__new__(Acceptance)
    obj.run_dir = tmp_path
    obj.target = tmp_path/"ledger.json"
    obj.report = empty_ledger()
    obj.artifact_dir = tmp_path
    return obj


def samples():
    return np.repeat(prior_quantiles((np.arange(64)+.5)/64)[None,:,:],2,axis=0)


def test_new_gate_wrappers_and_ledger(tmp_path,monkeypatch):
    obj = bare_runner(tmp_path,monkeypatch)
    p = samples(); center = p.mean(1,keepdims=True)
    def generated(self,key,n,**kwargs):
        scale = 1. if kwargs.get("localized_fraction")==0. else .3
        return center+scale*(p-center),center[:,0],{"tiny_control":True}
    obj.generated = types.MethodType(generated,obj)
    assert obj.run("G-P6")["status"]=="pass"
    assert obj.run("G-P8")["status"]=="pass"
    # Exercise the actual G-P5 artifact reader and actual sbi adapter on a tiny
    # control. Either statistical status is allowed here; it is not acceptance.
    path = tmp_path/"sbc.npz"
    np.savez(path,posterior=p,truth=center[:,0])
    obj.report["artifacts"]["sbc"] = {"path":"sbc.npz","sha256":acceptance.sha(path)}
    assert obj.gate_G_P5()["status"] in ("pass","fail")
    obj.sample = lambda *args,**kwargs:p[:1]
    assert obj.run("smoke")["status"]=="reported"
    saved = json.loads((tmp_path/"results/phase2a_gates.json").read_text())
    assert saved["gates"]["G-P6"]["hard"]
    assert saved["smoke_test"]["status"]=="reported"
    assert not obj.finalize(preflight_passed=True)  # Other gates are unrun.


def test_failed_gate_stops_following_work(tmp_path,monkeypatch):
    obj = bare_runner(tmp_path,monkeypatch)
    obj.gate_G_P6 = lambda:{"status":"fail","ratios":[1.,1.,1.,1.],"tolerance":.7}
    with pytest.raises(RuntimeError,match="STOP: G-P6"):
        obj.run("G-P6")
    with pytest.raises(RuntimeError,match="earlier failure"):
        obj.run("G-P8")
    assert obj.report["gates"]["G-P8"]["status"]=="not_run"
    assert not obj.finalize(preflight_passed=True)


def test_complete_ledger_refreshes_current_preflight_provenance(tmp_path,monkeypatch):
    obj = bare_runner(tmp_path,monkeypatch)
    fixtures = {
        "selection-audit.json":{"numeric_checks_pass":True,"prints_within_factor_two":True,
                                "z02_verification":{"status":"pass"}},
        "generator-audit.json":{"status":"pass"},
        "prior-predictive.json":{"status":"pass","events":[{"frb":"fixture","quadrature_CDF":.5}]},
        "model-audit.json":{"status":"pass"},
    }
    for name,payload in fixtures.items():
        write_json(tmp_path/"results"/name,payload)
    obj.report["pretraining"] = {"T1":{"status":"pass"}}
    for row in obj.report["gates"].values():
        row["status"] = "pass" if row["hard"] else "reported"
    assert obj.finalize(preflight_passed=True)
    assert obj.report["exit_status"]==0
    assert obj.report["pretraining"]["T3"]["sha256"]==acceptance.sha(tmp_path/"results/prior-predictive.json")
    write_json(tmp_path/"results/model-audit.json",{"status":"fail"})
    assert not obj.finalize(preflight_passed=True)
