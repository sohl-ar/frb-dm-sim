"""Selected generator engineering gate; no network or training."""
import importlib.util
from pathlib import Path
import pytest

pytestmark = pytest.mark.phase2a


def test_generator_statistics_schema_seeds_and_rate():
    path = Path(__file__).resolve().parents[2]/"scripts"/"generator_audit.py"
    spec = importlib.util.spec_from_file_location("generator_audit",path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    report = audit.run_audit()
    assert report["numeric_checks_pass"], "STOP: generator distribution, schema or determinism failure"
    assert report["status"]=="pass", "STOP: generator throughput below immutable speed gate; see profile"
