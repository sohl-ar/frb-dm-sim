"""Check handoff launch wiring with training replaced by a spy; no updates."""
from dataclasses import asdict
import hashlib
import json
import pytest
from frbsbi import train_conditioned as entry
from frbsbi.model import ModelConfig
from frbsbi.train import TrainConfig,write_json


def test_single_attempt_launch_and_baseline_archive(tmp_path,monkeypatch):
    root = tmp_path
    results = root/"results"
    run = results/"conditioning-v1"
    run.mkdir(parents=True)
    monkeypatch.setattr(entry,"ROOT",root)
    monkeypatch.setattr(entry,"RUN_DIR",run)
    normalizer = run/"normalization.json"
    write_json(normalizer,{"fixture":"launch spy only"})
    write_json(run/"launch.json",{"training":asdict(TrainConfig()),
        "model":asdict(ModelConfig(statistics_bypass=True)),
        "normalization_sha256":hashlib.sha256(normalizer.read_bytes()).hexdigest()})
    for name in ("phase2a_gates.json","generator-audit.json","model-audit.json","prior-predictive.json",
                 "selection-audit.json","training.json","posterior-gates.json"):
        write_json(results/name,{"baseline":name})
    calls = []
    def spy(*args,**kwargs):
        calls.append((args,kwargs))
        return {"status":"spy_only_no_training"}
    monkeypatch.setattr(entry,"train",spy)
    assert entry.launch()["status"]=="spy_only_no_training"
    assert len(calls)==1 and calls[0][0][0]==TrainConfig()
    assert calls[0][0][1].statistics_bypass
    archive = run/"baseline-evidence"
    for name,digest in json.loads((archive/"manifest.json").read_text()).items():
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==digest
        assert (archive/name).read_bytes()==(results/name).read_bytes()
    with pytest.raises(FileExistsError):
        entry.launch()
    assert len(calls)==1  # A second attempt cannot reach training.
