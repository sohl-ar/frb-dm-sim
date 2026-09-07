import numpy as np
import pytest
from frbsim.constants import REGISTRY
from frbsim.host import from_log10, observed_host, sample_host

pytestmark = pytest.mark.gates

def test_g6(gate):
    citations = {n: REGISTRY[n].source for n in ["host_median", "host_sigma_ln"]}
    gate("G6", "host_default_citations", citations, {"require": "Macquart Eq. 3 ranges + spec choices"},
         all("Macquart Eq. 3" in v for v in citations.values()))
    mu, sigma = from_log10(2., .4)
    gate("G6", "host_base_conversion", {"median": float(np.exp(mu)), "sigma_ln": sigma},
         {"median": 100., "both_parameters_converted": True},
         np.isclose(np.exp(mu), 100.) and np.isclose(sigma, .4*np.log(10)))
    draws = sample_host(10, np.random.default_rng(6))
    gate("G6", "host_observed_scaling", float(np.max(abs(observed_host(draws, 1)*2-draws))),
         {"equals": 0.}, bool(np.array_equal(observed_host(draws, 1)*2, draws)))
