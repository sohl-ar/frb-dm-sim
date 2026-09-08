"""Read-only numerical verification, followed by a generated provenance record."""
import json
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from frbsbi.data import ROOT
from frbsbi.evaluate import sbc_statistics
from frbsbi.gate_metrics import contraction
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from complete_phase2a_diagnosis import OUT,sha,load_npz,coverage_detail

r = json.loads((OUT/"diagnosis.json").read_text())
before = json.loads((OUT/"ledger-before.json").read_text())
ledger = json.loads((RUN_DIR/"phase2a_gates.json").read_text())
assert ledger==json.loads((ROOT/"results/phase2a_gates.json").read_text())
assert ledger["gates"]["G-P6"]==before["gates"]["G-P6"]
assert not ledger["acceptance_complete"]
for names in (("low_localized","medium_localized","high_localized"),("small_N","large_N")):
    assert sum(r["groups"][g]["n_trials"] for g in names)==r["coverage_exact"][0]['0.68']['trials']
    for j,row in enumerate(r["coverage_exact"]):
        for level in ('0.68','0.95'):
            assert sum(r["groups"][g]['coverage'][j][level]['covered'] for g in names)==row[level]['covered']
for name in ("all_localized","all_unlocalized"):
    artifact = r["host_probe"]["artifacts"][name]
    path = ROOT/artifact["path"]
    assert sha(path)==artifact["sha256"]
    d = load_npz(path)
    assert coverage_detail(d["posterior"],d["truth"])==r["host_probe"][name]["coverage"]
    assert json.loads(json.dumps(contraction(d["posterior"],.5)))==r["host_probe"][name]["contraction_diagnostic"]
ablation = json.loads((OUT/"flag-ablation.json").read_text())
path = ROOT/ablation["artifact"]["path"]
assert sha(path)==ablation["artifact"]["sha256"]
d = load_npz(path)
assert coverage_detail(d["posterior"],d["truth"])==ablation["result"]["coverage"]
assert json.loads(json.dumps(contraction(d["posterior"],.5)))==ablation["result"]["contraction_diagnostic"]
original = load_npz(ROOT/before["artifacts"]["sbc"]["path"])
corrected,ranks = sbc_statistics(original["posterior"],original["truth"],jitter_seed=before["gates"]["G-P4"]["rank_jitter_seed"])
assert corrected==ledger["gates"]["G-P4"]["parameters"]==r["corrected_SBC"]
assert ranks.tolist()==before["gates"]["G-P4"]["raw_ranks"]
source_changes = subprocess.check_output(["git","diff","d83c98c","--name-only","--","src"],cwd=ROOT,text=True).splitlines()
assert source_changes==["src/frbsbi/evaluate.py"]
checkpoint = ROOT/ledger["trained_checkpoint"]
assert sha(checkpoint)==before["identity"]["checkpoint_sha256"]
suite = ET.parse(OUT/"comparison-tests.xml").getroot().find("testsuite")
tests = {k:int(suite.attrib[k]) for k in ("tests","failures","errors","skipped")}
assert tests["failures"]==tests["errors"]==0
record = {"status":"pass","source_changes":source_changes,"checkpoint_unchanged":True,
    "all_original_ranks_unchanged":True,"contraction_gate_unchanged":True,
    "group_counts_partition_original_trials":True,"group_coverage_counts_sum_to_original_counts":True,
    "host_probe_metrics_recomputed_from_NPZ":True,"counterfactual_metrics_recomputed_from_NPZ":True,
    "tests":tests,"evidence":{p.name:sha(p) for p in (OUT/"diagnosis.json",OUT/"training-support.json",OUT/"flag-ablation.json")}}
write_json(OUT/"verification.json",record)
ledger["completed_diagnosis"] = {"status":"complete","gate_recommendation":"remain_incomplete; no criterion change",
    "report":"CONDITIONING_FOLLOWUP.md","verification":(OUT/"verification.json").relative_to(ROOT).as_posix(),
    "verification_sha256":sha(OUT/"verification.json"),"evidence_hashes":record["evidence"]}
write_json(RUN_DIR/"phase2a_gates.json",ledger)
write_json(ROOT/"results/phase2a_gates.json",ledger)
print(json.dumps(record,indent=2))
