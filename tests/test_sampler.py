"""Distribution-domain checks at the inference parameter-box boundaries."""
import numpy as np
import pytest
from frbsim.fields import cosmic_pdf, sample_cosmic
from frbsim.numerics import ROOT_XTOL

@pytest.mark.parametrize("F,z", [(.02, .05), (.02, 1.5), (1., .05), (1., 1.5)])
def test_pdf_parameter_corners(F, z):
    pdf = cosmic_pdf(F/np.sqrt(z))
    assert abs(pdf.mean-1) < np.sqrt(ROOT_XTOL)
    quantiles = pdf.ppf(np.linspace(0, 1, 1001))
    assert np.all(np.isfinite(quantiles)) and np.all(quantiles > 0)
    assert np.all(np.diff(quantiles) >= 0)
    assert np.all(np.diff(pdf.cdf_values) >= 0)

def test_heterogeneous_replay_and_empty_input():
    z = np.linspace(.05, 1.5, 1000)
    a = sample_cosmic(z, np.random.default_rng(123), F=1.)
    b = sample_cosmic(z, np.random.default_rng(123), F=1.)
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a)) and np.all(a > 0)
    assert sample_cosmic(np.array([]), np.random.default_rng(123)).size == 0
