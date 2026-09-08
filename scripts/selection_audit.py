"""Executed D4/D5 checks and human-comparison prints; no training."""
from dataclasses import asdict
import json
from pathlib import Path
import hashlib
import numpy as np
from scipy.integrate import quad
from scipy.stats import ks_2samp
from frbsbi.selection import SchechterLF, FLUENCE_MIN, log_upper_gamma
from frbsim.cosmology import luminosity_distance, differential_volume
from frbsim.population import sfr_shape, sample_redshifts
from frbsim.numerics import POPULATION_GRID_SIZE
from frbsim.catalog import git_provenance
from frbsbi.preflight import SEED_RANGES, assert_disjoint_seed_ranges
from frbsbi.tolerances import (SELECTION_PRINT_FACTOR, TAIL_QUADRATURE_REL,
                              TAIL_REJECTION_SE, CONDITIONAL_KS_P_MIN)

ROOT = Path(__file__).resolve().parents[1]
SEED = SEED_RANGES["audit"].start + 1


def run_audit():
    assert_disjoint_seed_ranges(SEED_RANGES)
    lf = SchechterLF()
    z = np.array([.1, .2, .5, 1., 1.4])
    threshold = 4*np.pi*luminosity_distance(z)**2*FLUENCE_MIN
    fraction = lf.tail(threshold)
    # User's rough human-comparison ranges; factor-two STOP boundary retained.
    anchors = np.array([[.02,.02],[.01,.015],[.005,.007],[.003,.003],[.001,.002]])
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
                            "scope":"naive-rejection validation only; no training data"}
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
              "scale_fluence_at_z1":float(lf.scale/(4*np.pi*luminosity_distance(1.)**2)),
              "brightest_note":"Schechter has unbounded support; this is scale fluence, not a maximum burst fluence",
              "tail_quadrature":checks, "naive_rejection":{
                  "n_intrinsic":n,"detected_counts":counts.tolist(),
                  "tail_errors_in_binomial_se":errors.tolist(),
                  "conditional_ks_statistic":float(ks.statistic),"conditional_ks_pvalue":float(ks.pvalue)},
              "tolerances":{"print_factor":SELECTION_PRINT_FACTOR,"quadrature_relative":TAIL_QUADRATURE_REL,
                            "rejection_standard_errors":TAIL_REJECTION_SE,"conditional_ks_p_min":CONDITIONAL_KS_P_MIN},
              "numeric_checks_pass":bool(max(c["relative_error"] for c in checks)<TAIL_QUADRATURE_REL
                  and np.max(errors)<TAIL_REJECTION_SE and ks.pvalue>CONDITIONAL_KS_P_MIN)}
    report["config_hash"] = hashlib.sha256(json.dumps({"lf":asdict(lf),"seed":SEED,"F_min":FLUENCE_MIN},sort_keys=True).encode()).hexdigest()
    target = ROOT/"results"/"selection-audit.json"
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    report = run_audit()
    print(json.dumps({k:report[k] for k in ("z","detected_fraction","scale_fluence_at_z1","prints_within_factor_two","numeric_checks_pass")}))
    return 0 if report["prints_within_factor_two"] and report["numeric_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
