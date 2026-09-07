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
    "G3": "BLOCKED: GLASS/healpy native dependency build; L2 power and density validation not implemented",
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
    phase2_items = [item for item in session.items if item.get_closest_marker("phase2a")]
    if phase2_items:
        from frbsbi.preflight import write_phase2a_report
        failures = session.config.gate_report.get("test_failures", [])
        write_phase2a_report(ROOT, getattr(session.config, "phase2a_t1", None),
                            not bool(exitstatus), failures)
        if not exitstatus:
            session.exitstatus = exitstatus = 1
        session.config.get_terminal_writer().line(
            "STOP: Phase 2a acceptance incomplete; see results/phase2a_gates.json", red=True)
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
