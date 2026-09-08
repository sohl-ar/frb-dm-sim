"""Every gate run writes an atomic JSON report, including unrun gates."""
import json
import os
from pathlib import Path
import subprocess
import time
import pytest
from frbsim.tolerances import GATE_NAMES

ROOT = Path(__file__).resolve().parents[1]

PENDING = {
    "G3": "BLOCKED: L2 power and density validation not implemented; dependency status is environment-specific",
    "G4": "PARTIAL: L0 checks only; calibrated L2 G4(ii) and full noisy pipeline G4(iii) remain",
    "G5": "BLOCKED: pygedm native build; pulsar and pole validations not executed",
    "G7": "PARTIAL: L0 fixed-ISM engineering fixture only; full L2 + pygedm catalog remains",
}

def pytest_addoption(parser):
    parser.addoption("--level", choices=("l0", "l2"), default="l2",
                     help="l0 runs implemented checks; l2 (default) requires full acceptance")

def provenance():
    def git(*args):
        proc = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}

def pytest_sessionstart(session):
    session.config.gate_report = {"schema_version": 1, "started_unix": time.time(),
        **provenance(), "gates": {n: {"name": n, "status": "not_run", "checks": []} for n in GATE_NAMES}}


def pytest_collection_modifyitems(items):
    # Existing preflight assertions run before costly posterior tests.
    items.sort(key=lambda item:item.path.name=="test_posterior_acceptance.py")

@pytest.fixture
def gate(request):
    def check(name, label, measured, tolerance, passed, hard=True):
        report = request.config.gate_report
        item = report["gates"][name]
        item["checks"].append({"name": label, "status": "pass" if passed else "fail",
                               "measured": measured, "tolerance": tolerance, "hard": hard,
                               "git_sha": report["git_sha"]})
        item["status"] = "fail" if any(x["status"] == "fail" for x in item["checks"]) else "pass"
        if hard and not passed:
            pytest.fail(f"STOP: {name}/{label}: measured={measured}; tolerance={tolerance}")
    return check

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        item.config.gate_report.setdefault("test_failures", []).append(
            {"nodeid": item.nodeid, "phase": report.when, "detail": str(report.longrepr)})

def pytest_sessionfinish(session, exitstatus):
    if session.config.option.collectonly:
        return
    phase2_items = [item for item in session.items if item.get_closest_marker("phase2a")]
    if phase2_items:
        failures = session.config.gate_report.get("test_failures", [])
        from frbsbi.acceptance import empty_ledger
        from frbsbi.train import write_json
        runtime = getattr(session.config,"phase2a_runtime",None)
        # A posterior-only subset cannot claim that preflight was executed.
        required = {"test_t1.py","test_t3.py","test_selection.py","test_model.py","test_generator.py"}
        selected = {item.path.name for item in phase2_items}
        complete = False
        if runtime is not None:
            runtime.report["pretraining"]["T1"] = getattr(session.config,"phase2a_t1",{"status":"not_run"})
            complete = runtime.finalize(preflight_passed=not bool(exitstatus) and required<=selected,failures=failures)
        else:
            target = ROOT/"results/conditioning-v1/phase2a_gates.json"
            report = json.loads(target.read_text()) if target.exists() else empty_ledger()
            report.update(test_failures=failures,reason="trained conditioned gates not reached; acceptance incomplete")
            report.update(acceptance_complete=False,exit_status=1)
            if failures:
                report.setdefault("stopped_at",failures[0]["nodeid"])
            # Preserve the original baseline before promoting any new ledger.
            import shutil
            baseline = ROOT/"results/conditioning-v1/baseline-phase2a-ledger.json"
            if not baseline.exists():
                shutil.copyfile(ROOT/"results/phase2a_gates.json",baseline)
            write_json(target,report)
            write_json(ROOT/"results/phase2a_gates.json",report)
        if not complete and not exitstatus:
            session.exitstatus = exitstatus = 1
        if not complete:
            session.config.get_terminal_writer().line(
                "STOP: Phase 2a acceptance incomplete; see results/phase2a_gates.json",red=True)
    # A Phase 2a-only run must never overwrite saved Phase 1 evidence.
    if not any(item.get_closest_marker("gates") for item in session.items):
        return
    report = session.config.gate_report
    report["requested_level"] = session.config.getoption("--level")
    report["gate_tests_selected"] = any(item.get_closest_marker("gates") for item in session.items)
    for name, reason in PENDING.items():
        item = report["gates"][name]
        item["pending"] = reason
        if item["status"] == "pass":
            item["status"] = "partial"
        elif item["status"] == "not_run":
            item["status"] = "blocked"
    incomplete = any(item["status"] != "pass" for item in report["gates"].values())
    if report["gate_tests_selected"] and report["requested_level"] == "l2" and incomplete and not exitstatus:
        exitstatus = 1
        session.exitstatus = 1
        session.config.get_terminal_writer().line("STOP: full L2 acceptance incomplete; see results/gates.json", red=True)
    report["exit_status"] = int(exitstatus)
    report["elapsed_seconds"] = time.time() - report["started_unix"]
    report["acceptance_complete"] = int(exitstatus) == 0 and all(
        x["status"] == "pass" for x in report["gates"].values())
    target = Path(os.environ.get("FRBSIM_GATE_REPORT", str(ROOT / "results" / "gates.json")))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(target)
