"""Authorized split: Euclidean geometry vs fixed-distance LF cumulative."""
import numpy as np
import pytest
from frbsim.population import LuminosityFunction
from frbsim.selection import detected, fluence_from_luminosity
from frbsim import tolerances as t
from frbsim.population import sample_redshifts
from frbsim.cosmology import luminosity_distance
from frbsim.fields import sample_cosmic

pytestmark = pytest.mark.gates

def fitted_slope(flux, thresholds):
    counts = np.array([np.count_nonzero(detected(flux, f)) for f in thresholds])
    return float(np.polyfit(np.log(thresholds), np.log(counts), 1)[0]), counts

def test_g8a(gate):
    rng = np.random.default_rng(8001)
    # Dimensionless geometry fixture, wholly separate from population z_min.
    # r_min/r_max=1e-4; sampled flux thresholds select interior radii.
    r_min, r_max = 1e-4, 1.
    r = (r_min**3 + rng.random(1_000_000)*(r_max**3-r_min**3))**(1/3)
    L = np.full(r.size, 4*np.pi)
    flux = fluence_from_luminosity(L, r)
    thresholds = np.geomspace(2., 10., 20)
    slope, counts = fitted_slope(flux, thresholds)
    gate("G8", "G8a_Euclidean_delta_LF", {"slope": slope, "counts": counts.tolist(),
         "r_min": r_min, "r_max": r_max, "thresholds": thresholds.tolist()},
         {"slope": -1.5, "absolute": t.COUNT_SLOPE_ABS}, abs(slope+1.5) <= t.COUNT_SLOPE_ABS)
    gate("G8", "G8a_monotonic", counts.tolist(), {"nonincreasing": True}, bool(np.all(np.diff(counts) <= 0)))

def test_g8b(gate):
    rng = np.random.default_rng(8002)
    for alpha in [1.8, 2., 2.2]:
        lf = LuminosityFunction("power_law", 1., 1e9, alpha, "G8b numerical fixture, not an empirical LF default")
        L = lf.sample(1_000_000, rng)
        # D_L fixed: fluence thresholds map to interior L thresholds.
        flux = fluence_from_luminosity(L, 1.)
        thresholds = np.geomspace(10., 100., 20)/(4*np.pi)
        slope, counts = fitted_slope(flux, thresholds)
        gate("G8", f"G8b_fixed_DL_alpha={alpha}", {"slope": slope, "counts": counts.tolist()},
             {"slope": -(alpha-1), "absolute": t.COUNT_SLOPE_ABS}, abs(slope+(alpha-1)) <= t.COUNT_SLOPE_ABS)
        gate("G8", f"G8b_monotonic_alpha={alpha}", counts.tolist(), {"nonincreasing": True},
             bool(np.all(np.diff(counts) <= 0)))

def test_g8_malmquist_report(gate):
    rng = np.random.default_rng(8003)
    z = sample_redshifts(10_000, rng)
    dm = sample_cosmic(z, rng)
    flux = fluence_from_luminosity(np.ones(z.size), luminosity_distance(z))
    threshold = float(np.median(flux))
    keep = detected(flux, threshold)
    gate("G8", "Malmquist_report_delta_L_fixture", {
        "scope": "numerical fixture; no empirical survey prediction",
        "threshold": threshold, "intrinsic_mean_z": float(z.mean()),
        "detected_mean_z": float(z[keep].mean()), "delta_z": float(z[keep].mean()-z.mean()),
        "intrinsic_mean_DMcosmic": float(dm.mean()), "detected_mean_DMcosmic": float(dm[keep].mean()),
        "delta_DMcosmic": float(dm[keep].mean()-dm.mean())}, {"report_only": True}, True, hard=False)
