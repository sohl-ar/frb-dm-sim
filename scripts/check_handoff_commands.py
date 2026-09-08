"""Validate CLI wiring without launching training or statistical acceptance."""
import json
import subprocess
import sys
from frbsbi.data import ROOT
from frbsbi.acceptance import empty_ledger
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from frbsim.catalog import git_provenance

commands = [
    [sys.executable,"-m","frbsbi.train_conditioned","--help"],
    [sys.executable,"-m","frbsbi.acceptance","--help"],
    [sys.executable,"scripts/report_conditioning_execution.py","--section","all"],
    [sys.executable,"-m","pytest","-m","phase2a","--collect-only","-q"],
]
records = []
for command in commands:
    result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    records.append({"command":command,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
    if result.returncode:
        write_json(RUN_DIR/"command-checks.json",{"status":"fail","commands":records})
        raise RuntimeError("STOP: command wiring check failed")
assert (ROOT/"work/HANDOFF.md").read_bytes()==(ROOT/"HANDOFF.md").read_bytes()
assert not (RUN_DIR/"TRAINING_ATTEMPT.lock").exists()
assert not (RUN_DIR/"training.json").exists()
plan = empty_ledger()
plan.update(scope="prepared implementation ledger; no new training or acceptance run",
            execution_handoff="HANDOFF.md",quick_evidence="results/conditioning-v1/implementation-checks.json")
write_json(RUN_DIR/"prepared-gate-ledger.json",plan)
write_json(RUN_DIR/"command-checks.json",{**git_provenance(),"status":"pass","commands":records,
    "handoff_copies_identical":True,"training_launched":False,"full_gates_executed":False})
print(json.dumps({"status":"pass","command_checks":len(records),"training_launched":False}))
