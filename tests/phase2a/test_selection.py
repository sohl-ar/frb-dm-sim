"""D4/D5 validation, including the unchanged human-comparison STOP rule."""
import importlib.util
from pathlib import Path
import pytest

pytestmark = pytest.mark.phase2a


def test_selection_authorized_prints():
    path = Path(__file__).resolve().parents[2]/"scripts"/"selection_audit.py"
    spec = importlib.util.spec_from_file_location("selection_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    report = audit.run_audit()
    assert report["numeric_checks_pass"], "STOP: Schechter sampler validation failed"
    assert report["prints_within_factor_two"], (
        "STOP: authorized LF disagrees with selection prints; see results/selection-audit.json")
