"""L0 checks only: the amended G4(ii) requires a separately calibrated L2.

Host sampling uses NATURAL LOG, with the spec's median=100, sigma_ln=.8.
This is a conditional predictive smoke test, not a fitted posterior.
"""
import numpy as np
import pytest
from frbsim.fields import cosmic_pdf, sample_cosmic
from frbsim.mean_dm import transcription
from frbsim.host import sample_host, observed_host
from frbsim import tolerances as t

pytestmark = pytest.mark.gates
TABLE1 = (("180924", 361.42, .3214), ("181112", 589.27, .4755),
          ("190102", 363.6, .2913), ("190608", 338.7, .1178),
          ("190611", 321.4, .378), ("190711", 593.1, .522))

def test_g4_mean_and_shape(gate):
    rng = np.random.default_rng(20200924)
    for z in [.1, .5, 1., 1.5]:
        pdf = cosmic_pdf(.2/np.sqrt(z))
        wider = cosmic_pdf(.2/np.sqrt(z), delta_max=2*pdf.delta_max)
        q = pdf.ppf([.05, .95])
        qwide = wider.ppf([.05, .95])
        draws = pdf.sample(500*2000, rng).reshape(500, 2000)
        relative = abs(float(draws.mean())-1)
        gate("G4", f"L0_mean_500_catalogs_z={z}",
             {"sample_relative_error": relative, "quadrature_mean": pdf.mean,
              "C0": pdf.c0, "log_A": pdf.log_A, "delta_max": pdf.delta_max,
              "truncated_std_delta": pdf.truncated_std},
             {"relative_mean": t.DM_MEAN_REL}, relative <= t.DM_MEAN_REL)
        cutoff_error = float(np.max(abs(qwide/q-1)))
        gate("G4", f"L0_cutoff_convergence_z={z}", cutoff_error,
             {"relative": t.DM_MEAN_REL}, cutoff_error <= t.DM_MEAN_REL)
        gate("G4", f"L0_asymmetry_z={z}", q.tolist(),
             {"condition": "mean-q05 < q95-mean"}, 1-q[0] < q[1]-1)

def test_g4_table1(gate):
    rng = np.random.default_rng(2020190608)
    coverage = 0
    details = []
    # No instrumental noise scale is supplied by the spec. The zero-noise
    # limit is explicit and this check is partial until a noise model is set.
    for name, observed, z in TABLE1:
        cosmic = sample_cosmic(np.full(500, z), rng, F=.2)
        host = sample_host(500, rng)
        observed_draws = 30 + 50 + cosmic + observed_host(host, z)
        correction = 30 + 50 + 50/(1+z)
        predictive = observed_draws-correction
        bounds = np.quantile(predictive, [.05, .95])
        estimated = observed-correction
        inside = bool(bounds[0] <= estimated <= bounds[1])
        coverage += inside
        details.append({"frb": name, "z": z, "dm_observed": observed,
                        "dm_cosmic_estimated": estimated, "analytic_mean": transcription(z),
                        "predictive_5_95": bounds.tolist(), "inside": inside,
                        "tentative": name == "190611"})
    gate("G4", "L0_Table1_zero_noise_limit", {"covered": coverage, "events": details,
         "noise_model": "zero-noise limit; not full noisy G4(iii) acceptance"},
         {"minimum_covered": t.PREDICTIVE_REQUIRED, "realizations_per_event": 500},
         coverage >= t.PREDICTIVE_REQUIRED)
