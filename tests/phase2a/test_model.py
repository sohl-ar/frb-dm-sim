"""Native architecture and short-run determinism checks before full training."""
import importlib.util
from pathlib import Path
import pytest

pytestmark = pytest.mark.phase2a


def test_model_engineering():
    path = Path(__file__).resolve().parents[2]/"scripts/model_audit.py"
    spec = importlib.util.spec_from_file_location("model_audit",path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    report = audit.run_audit()
    assert report["status"]=="pass", "STOP: native model engineering check failed"
