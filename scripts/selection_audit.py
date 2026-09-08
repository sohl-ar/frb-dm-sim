"""Executed D4/D5 checks and human-comparison prints; no training."""
from dataclasses import asdict
import json
from pathlib import Path
import hashlib
import math
import numpy as np
from scipy.integrate import quad
from scipy.stats import ks_2samp
from frbsbi.selection import SchechterLF, FLUENCE_MIN, log_upper_gamma
from frbsim.cosmology import luminosity_distance, differential_volume, ASTROPY_COSMO
from frbsim.population import sfr_shape, sample_redshifts
from frbsim.numerics import POPULATION_GRID_SIZE
from frbsim.catalog import git_provenance
from frbsbi.preflight import SEED_RANGES, assert_disjoint_seed_ranges
from frbsbi.tolerances import (SELECTION_PRINT_FACTOR, TAIL_QUADRATURE_REL,
                              TAIL_REJECTION_SE, CONDITIONAL_KS_P_MIN)
from frbsim.tolerances import DISTANCE_REL

ROOT = Path(__file__).resolve().parents[1]
SEED = SEED_RANGES["audit"].start + 1


def upper_gamma_quadrature(s, x):
    shift = s*np.log(x)-x
    scaled = quad(lambda v: np.exp(s*(np.log(x)+v)-x*np.exp(v)-shift),
                  0., np.log(1+1000/x), epsabs=1e-12, epsrel=1e-11)[0]
    return float(np.exp(shift)*scaled)


def upper_gamma_series(s, x):
    # Independent small-x expansion: Gamma(s) - x^s sum_n (-x)^n/[n!(s+n)].
    # math.gamma supplies the complete gamma; no incomplete-gamma recurrence.
    terms = [(-x)**k/(math.factorial(k)*(s+k)) for k in range(64)]
    return math.gamma(s)-x**s*math.fsum(terms)


def run_audit():
    assert_disjoint_seed_ranges(SEED_RANGES)
    lf = SchechterLF()
    z = np.array([.1, .2, .5, 1., 1.4])
    threshold = 4*np.pi*luminosity_distance(z)**2*FLUENCE_MIN
    fraction = lf.tail(threshold)
    # User's rough human-comparison ranges; factor-two STOP boundary retained.
    anchors = np.array([[.02,.02],[.01,.015],[.005,.007],[.00151,.00151],[.000494,.000494]])
    checks = []
    for s in [-.16, 0., .23]:
        for x in [1e-12, 1e-4, .01, .1, 1., 10., 100.]:
            # Independent log-coordinate integral, scaled to avoid tiny absolute values.
            shift = s*np.log(x)-x
            integral = quad(lambda v: np.exp(s*(np.log(x)+v)-x*np.exp(v)-shift),
                            0., np.log(1+1000/x), epsabs=1e-12, epsrel=1e-11)[0]
            analytic = float(log_upper_gamma(s, x))
            checks.append({"s":s,"x":x,"relative_error":float(abs(np.expm1(analytic-shift-np.log(integral))))})
    rng = np.random.default_rng(SEED)
    n = 1_000_000
    intrinsic = lf.sample_above(np.full(n, lf.minimum), rng)
    counts = np.count_nonzero(intrinsic[:,None] >= threshold[None,:], axis=0)
    errors = np.abs(counts/n-fraction)/np.sqrt(fraction*(1-fraction)/n)
    accepted = intrinsic[intrinsic >= threshold[2]]
    conditional = lf.sample_above(np.full(len(accepted), threshold[2]), rng)
    ks = ks_2samp(accepted, conditional)
    overall, realized = {}, {}
    # Independent intrinsic-rejection diagnostics only; not a production
    # generator. Interpolate an independently quadrature-evaluated DL table.
    z_grid = np.linspace(.05, 1.5, POPULATION_GRID_SIZE)
    dl_grid = luminosity_distance(z_grid)
    for family in ("sfr", "constant_comoving"):
        def weight(t):
            return differential_volume(t)*(sfr_shape(t) if family == "sfr" else 1.)
        total = quad(weight, .05, 1.5, epsrel=1e-9)[0]
        selected = quad(lambda t: weight(t)*lf.tail(4*np.pi*luminosity_distance(t)**2*FLUENCE_MIN),
                        .05, 1.5, epsrel=1e-9)[0]
        overall[family] = float(selected/total)
        intrinsic_z = sample_redshifts(n, rng, family=family)
        intrinsic_dl = np.interp(intrinsic_z, z_grid, dl_grid)
        keep = intrinsic >= 4*np.pi*intrinsic_dl**2*FLUENCE_MIN
        realized[family] = {"n_intrinsic":n,"n_detected":int(np.count_nonzero(keep)),
                            "detected_fraction":float(np.mean(keep)),
                            "mean_detected_z":float(np.mean(intrinsic_z[keep])),
                            "fraction_detected_below_z05":float(np.mean(intrinsic_z[keep]<.5)),
                            "scope":"naive-rejection validation only; no training data"}
        low = quad(lambda t: weight(t)*lf.tail(4*np.pi*luminosity_distance(t)**2*FLUENCE_MIN),
                   .05, .5, epsrel=1e-9)[0]
        realized[family]["quadrature_fraction_detected_below_z05"] = float(low/selected)
    s, xmin, xt = lf.gamma+1, lf.minimum/lf.scale, float(threshold[1]/lf.scale)
    q02 = upper_gamma_quadrature(s,xt)/upper_gamma_quadrature(s,xmin)
    series02 = upper_gamma_series(s,xt)/upper_gamma_series(s,xmin)
    se02 = float(np.sqrt(q02*(1-q02)/n))
    z02_closed = bool(abs(q02/fraction[1]-1)<TAIL_QUADRATURE_REL
                     and abs(series02/q02-1)<TAIL_QUADRATURE_REL
                     and abs(counts[1]/n-q02)/se02<TAIL_REJECTION_SE)
    knee_z = np.array([1.,1.4])
    knee_dl = luminosity_distance(knee_z)
    astropy_dl = ASTROPY_COSMO.luminosity_distance(knee_z).value
    knee_f = lf.scale/(4*np.pi*knee_dl**2)
    distance_error = float(np.max(abs(knee_dl/astropy_dl-1)))
    # A threshold already ABOVE the knee must still produce valid tail draws.
    above_knee = lf.sample_above(np.full(10_000,2*lf.scale),rng)
    bright_support = bool(np.all(above_knee>=2*lf.scale) and np.ptp(above_knee)>0)
    allowed = bool(np.all(fraction >= anchors[:,0]/SELECTION_PRINT_FACTOR)
                   and np.all(fraction <= SELECTION_PRINT_FACTOR*anchors[:,1])
                   and np.all(np.diff(fraction)<0) and np.all((fraction>0)&(fraction<1)))
    report = {**git_provenance(), "seed":SEED, "lf":asdict(lf), "F_min":FLUENCE_MIN,
              "L_min":lf.minimum, "L_scale":lf.scale,
              "z":z.tolist(),"detected_fraction":fraction.tolist(),
              "population_averaged_detected_fraction":overall,
              "population_naive_rejection":realized,
              "user_print_ranges":anchors.tolist(),"prints_within_factor_two":allowed,
              "factor_below_print_lower_edge":(anchors[:,0]/fraction).tolist(),
              "print_authorization":"docs/ANCHOR_CORRECTION_V2.md; supersedes stale prints only",
              "human_corroboration":{"source":"ANCHOR CORRECTION AUTHORIZATION v2",
                  "z1_z14":"human series/quadrature agree at 10-15% level; corroboration, not numeric oracle",
                  "z02":"human quadrature agrees within approximately 1%",
                  "arbiter":"committed quadrature/rejection at 0c8c3d4 and executed multi-method checks"},
              "z02_verification":{"status":"pass" if z02_closed else "fail",
                  "DL_Mpc":float(luminosity_distance(.2)),"L_threshold":float(threshold[1]),"x_threshold":xt,
                  "unnormalized_tail_integrand_x_gamma_exp_minus_x":float(xt**lf.gamma*np.exp(-xt)),
                  "normalized_pdf_in_x_at_threshold":float(xt**lf.gamma*np.exp(-xt)/upper_gamma_quadrature(s,xmin)),
                  "quadrature_probability":q02,"series_probability":series02,
                  "recurrence_probability":float(fraction[1]),"rejection_probability":float(counts[1]/n),
                  "rejection_standard_error":se02,"rejection_error_in_se":float(abs(counts[1]/n-q02)/se02),
                  "reviewer_x_0_0022_probability":float(lf.tail(.0022*lf.scale))},
              "knee_fluences":{"z":knee_z.tolist(),"DL_Mpc":knee_dl.tolist(),"fluence_jy_ms":knee_f.tolist(),
                  "astropy_DL_Mpc":astropy_dl.tolist(),"distance_relative_error":distance_error,
                  "z14_side_of_threshold":"above" if knee_f[1]>FLUENCE_MIN else "below",
                  "interpretation":"At z=1.4 the knee is above the cut: detection starts just below the knee and includes the exponentially suppressed bright tail."},
              "above_knee_support":{"verified":bright_support,"lower_test_bound_in_knee_units":2.,
                  "sample_min_over_knee":float(above_knee.min()/lf.scale),"sample_max_over_knee":float(above_knee.max()/lf.scale),
                  "intrinsic_count_above_knee":int(np.count_nonzero(intrinsic>lf.scale)),
                  "code_inspection":"sample_above has a lower support floor only; its probability-dependent bisection bracket is not an upper luminosity cutoff"},
              "tail_quadrature":checks, "naive_rejection":{
                  "n_intrinsic":n,"detected_counts":counts.tolist(),
                  "tail_errors_in_binomial_se":errors.tolist(),
                  "conditional_ks_statistic":float(ks.statistic),"conditional_ks_pvalue":float(ks.pvalue)},
              "tolerances":{"print_factor":SELECTION_PRINT_FACTOR,"quadrature_relative":TAIL_QUADRATURE_REL,
                            "rejection_standard_errors":TAIL_REJECTION_SE,"conditional_ks_p_min":CONDITIONAL_KS_P_MIN},
              "numeric_checks_pass":bool(z02_closed and bright_support and distance_error<=DISTANCE_REL
                  and max(c["relative_error"] for c in checks)<TAIL_QUADRATURE_REL
                  and np.max(errors)<TAIL_REJECTION_SE and ks.pvalue>CONDITIONAL_KS_P_MIN)}
    report["config_hash"] = hashlib.sha256(json.dumps({"lf":asdict(lf),"seed":SEED,"F_min":FLUENCE_MIN},sort_keys=True).encode()).hexdigest()
    target = ROOT/"results"/"selection-audit.json"
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    report = run_audit()
    print(json.dumps({k:report[k] for k in ("z","detected_fraction","z02_verification","knee_fluences","population_naive_rejection","prints_within_factor_two","numeric_checks_pass")}))
    return 0 if report["prints_within_factor_two"] and report["numeric_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
