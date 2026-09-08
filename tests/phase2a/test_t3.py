"""Prior predictive must cover each event at its own corrected redshift."""
import pytest
from frbsbi.prior_predictive import run_t3

pytestmark = pytest.mark.phase2a


def test_t3_prior_predictive():
    report = run_t3()
    assert report["status"]=="pass", "STOP: event outside central 99% prior predictive"
