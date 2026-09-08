"""T3: cosmic-DM prior predictive at each Macquart event's own redshift.

f_d is uniform, F is log-uniform. Host priors use NATURAL LOG widths but
integrate out of this cosmic-only marginal; host subtraction is the fixed
paper ansatz, not a draw from the host prior. Both population families have
the same conditional cosmic-DM law at a fixed z.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from frbsim.fields import cosmic_pdf
from frbsim.mean_dm import transcription
from frbsim.catalog import git_provenance
from .generator import cosmic_from_uniforms, PRIOR_LO, PRIOR_HI
from .preflight import SEED_RANGES, assert_disjoint_seed_ranges
from .tolerances import T3_CDF_INTERVAL

ROOT = Path(__file__).resolve().parents[2]


def quadrature_cdf(dm,z,order):
    nodes,weights = leggauss(order)
    fd = PRIOR_LO[0]+(nodes+1)*(PRIOR_HI[0]-PRIOR_LO[0])/2
    logF = np.log(PRIOR_LO[1])+(nodes+1)*np.log(PRIOR_HI[1]/PRIOR_LO[1])/2
    delta = dm/(transcription(z,f_d=1.)*fd)
    total = 0.
    for F,w in zip(np.exp(logF),weights):
        pdf = cosmic_pdf(float(F/np.sqrt(z)))
        cdf = np.interp(np.log(delta),pdf.log_delta,pdf.cdf_values)
        total += w*np.dot(weights,cdf)/4
    return float(total)


def run_t3():
    assert_disjoint_seed_ranges(SEED_RANGES)
    # Read the executed post-amendment fixture, so Table 1 has one live source.
    table_report = json.loads((ROOT/"results/table1-after.json").read_text(encoding="utf-8"))
    events = next(c["measured"]["events"] for c in table_report["gates"]["G4"]["checks"]
                  if c["name"]=="L0_Table1_zero_noise_limit")
    seed,n = SEED_RANGES["audit"].start+1000,200_000
    rng = np.random.default_rng(seed)
    checks,diagnostics = [],[]
    for event in events+[{"z":z} for z in (.1,.5,1.)]:
        z = event["z"]
        fd = rng.uniform(PRIOR_LO[0],PRIOR_HI[0],n)
        F = np.exp(rng.uniform(np.log(PRIOR_LO[1]),np.log(PRIOR_HI[1]),n))
        draws = cosmic_from_uniforms(np.full(n,z),fd,F,rng.random(n))
        bounds = np.quantile(draws,T3_CDF_INTERVAL)
        item = {"z":z,"central_99_DM":bounds.tolist(),"mean_DM":float(draws.mean()),
                "median_DM":float(np.median(draws))}
        if "frb" in event:
            dm = event["dm_cosmic_estimated"]
            empirical = float(np.mean(draws<=dm))
            q64,q128 = quadrature_cdf(dm,z,64),quadrature_cdf(dm,z,128)
            inside = bool(T3_CDF_INTERVAL[0]<=q128<=T3_CDF_INTERVAL[1] and bounds[0]<=dm<=bounds[1])
            item.update({"frb":event["frb"],"dm_cosmic_estimated":dm,"empirical_CDF":empirical,
                         "quadrature_CDF":q128,"quadrature_64_CDF":q64,"order_doubling_CDF_delta":abs(q128-q64),
                         "empirical_CDF_standard_error":float(np.sqrt(q128*(1-q128)/n)),
                         "status":"pass" if inside else "fail"})
            checks.append(item)
        else:
            diagnostics.append(item)
    config = {"seed":seed,"draws_per_redshift":n,"f_d":[float(PRIOR_LO[0]),float(PRIOR_HI[0])],
              "F_loguniform":[float(PRIOR_LO[1]),float(PRIOR_HI[1])],"quadrature_orders":[64,128]}
    report = {**git_provenance(),"name":"T3","status":"pass" if all(e["status"]=="pass" for e in checks) else "fail",
              "config":config,"config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              "tolerance":{"CDF_interval":list(T3_CDF_INTERVAL)},"events":checks,"diagnostic_grid":diagnostics,
              "table_source":"results/table1-after.json",
              "table_sha256":hashlib.sha256((ROOT/"results/table1-after.json").read_bytes()).hexdigest(),
              "scope":"Cosmic DM conditional on z; no network, host or selection conditioning in this marginal"}
    (ROOT/"results/prior-predictive.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run_t3()
    print(json.dumps({"status":report["status"],"events":report["events"]},indent=2))
    raise SystemExit(0 if report["status"]=="pass" else 1)
