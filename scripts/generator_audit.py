"""Validate the conditional training generator, then measure its CPU rate."""
import cProfile
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import numpy as np
from scipy.stats import ks_2samp
from frbsbi.generator import (generate_batch,population_tables,cosmic_tables,
                              cosmic_from_uniforms,Z_MAX)
from frbsbi.preflight import SEED_RANGES, SeedRange
from frbsbi.selection import SchechterLF, FLUENCE_MIN
from frbsbi.tolerances import GENERATOR_BURSTS_PER_SECOND, CONDITIONAL_KS_P_MIN
from frbsim.cosmology import luminosity_distance
from frbsim.fields import cosmic_pdf, _mean_table
from frbsim.population import sample_redshifts
from frbsim.tolerances import DISTANCE_REL,DM_MEAN_REL
from frbsim.numerics import POPULATION_GRID_SIZE

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(batch):
    h = hashlib.sha256()
    for group in (batch.observations,batch.latents):
        for field in fields(group):
            value = getattr(group,field.name)
            h.update(value.tobytes() if isinstance(value,np.ndarray) else str(value).encode())
    return h.hexdigest()


def run_audit():
    # Cold run includes construction of the numerical population/PDF tables.
    population_tables.cache_clear()
    cosmic_tables.cache_clear()
    cosmic_pdf.cache_clear()
    _mean_table.cache_clear()
    seeds = list(range(SEED_RANGES["audit"].start+10,SEED_RANGES["audit"].start+522))
    start = time.perf_counter()
    batch = generate_batch(seeds,split="audit")
    elapsed = time.perf_counter()-start
    n = len(batch.observations.dm_obs)
    start = time.perf_counter()
    replay = generate_batch(seeds,split="audit")
    warm_elapsed = time.perf_counter()-start
    identical = fingerprint(batch)==fingerprint(replay)
    first = generate_batch(seeds[:1],split="audit")
    end = batch.observations.offsets[1]
    batch_invariant = bool(np.array_equal(first.latents.dm_cosmic,batch.latents.dm_cosmic[:end])
                           and np.array_equal(first.latents.L,batch.latents.L[:end]))
    overlap_rejected = False
    try:
        generate_batch([1],split="train",seed_ranges={"train":SeedRange(0,10),"val":SeedRange(9,20)})
    except ValueError as exc:
        overlap_rejected = "overlap" in str(exc)
    # Validate the fixed-width interpolation at previously untabulated widths.
    sigmas = np.geomspace(.05/np.sqrt(1.5),1/np.sqrt(.05),25)[1:-1]
    probabilities = np.array([.005,.05,.25,.5,.75,.95,.995])
    interpolated_errors = []
    for sigma in sigmas:
        z = np.full(len(probabilities),.05)
        F = np.full(len(probabilities),sigma*np.sqrt(.05))
        actual = cosmic_from_uniforms(z,np.ones_like(z),F,probabilities)/_mean_table(Z_MAX,1.)(z)
        expected = cosmic_pdf(float(sigma)).ppf(probabilities)
        interpolated_errors.append(float(np.max(abs(actual/expected-1))))
    obs,truth = batch.observations,batch.latents
    probe = truth.z_true[::max(1,len(truth.z_true)//100)]
    zgrid,dlgrid,_,_ = population_tables()
    distance_error = float(np.max(abs(np.interp(probe,zgrid,dlgrid)/luminosity_distance(probe)-1)))
    support = bool(np.all(obs.fluence_jy_ms>=FLUENCE_MIN*(1-np.finfo(float).eps)))
    finite = all(np.all(np.isfinite(getattr(group,f.name)))
                 for group in (obs,truth) for f in fields(group)
                 if isinstance(getattr(group,f.name),np.ndarray))
    sizes = np.diff(obs.offsets)
    schema_ok = bool(len(obs.z_obs)==np.count_nonzero(obs.is_localized)
                     and np.array_equal(obs.z_obs,truth.z_true[obs.is_localized])
                     and not hasattr(obs,"theta") and not hasattr(obs,"family"))
    # Independent naive rejection validates joint z,L via both marginals and
    # correlations/conditional summaries. No rejection in production.
    lf = SchechterLF()
    rng = np.random.default_rng(SEED_RANGES["audit"].start+2)
    n_reference = 1_000_000
    intrinsic_L = lf.sample_above(np.full(n_reference,lf.minimum),rng)
    family_per_burst = np.repeat(truth.family,sizes)
    joint = {}
    for flag,name in enumerate(("sfr","constant_comoving")):
        z_ref = sample_redshifts(n_reference,rng,family=name)
        dl_ref = np.interp(z_ref,zgrid,dlgrid)
        selected = intrinsic_L>=4*np.pi*dl_ref**2*FLUENCE_MIN
        rz,rL = z_ref[selected],intrinsic_L[selected]
        mask = family_per_burst==flag
        # Conditional survival PIT must be independent of z in the joint law.
        lower_ref = 4*np.pi*np.interp(rz,zgrid,dlgrid)**2*FLUENCE_MIN
        lower_new = 4*np.pi*np.interp(truth.z_true[mask],zgrid,dlgrid)**2*FLUENCE_MIN
        ref_pit = 1-np.exp(lf.log_tail(rL)-lf.log_tail(lower_ref))
        new_pit = 1-np.exp(lf.log_tail(truth.L[mask])-lf.log_tail(lower_new))
        ks_z = ks_2samp(rz,truth.z_true[mask])
        ks_L = ks_2samp(np.log(rL),np.log(truth.L[mask]))
        ks_pit = ks_2samp(ref_pit,new_pit)
        joint[name] = {"reference_n_detected":len(rz),"generated_n_detected":int(np.count_nonzero(mask)),
                       "ks_z_pvalue":float(ks_z.pvalue),"ks_logL_pvalue":float(ks_L.pvalue),
                       "ks_conditional_L_pvalue":float(ks_pit.pvalue),
                       "reference_z_pit_correlation":float(np.corrcoef(rz,ref_pit)[0,1]),
                       "generated_z_pit_correlation":float(np.corrcoef(truth.z_true[mask],new_pit)[0,1]),
                       "fraction_below_z05":float(np.mean(truth.z_true[mask]<.5))}
    profile = cProfile.Profile()
    profile.runcall(generate_batch,seeds[:64],split="audit")
    stream = io.StringIO()
    pstats.Stats(profile,stream=stream).sort_stats("cumulative").print_stats(20)
    (ROOT/"results/generator-profile.txt").write_text(stream.getvalue(),encoding="utf-8")
    numeric = bool(identical and batch_invariant and overlap_rejected and support and finite and schema_ok
                   and distance_error<=DISTANCE_REL and max(interpolated_errors)<=DM_MEAN_REL
                   and all(min(j["ks_z_pvalue"],j["ks_logL_pvalue"],j["ks_conditional_L_pvalue"])>CONDITIONAL_KS_P_MIN
                           for j in joint.values()))
    report = {"metadata":batch.metadata,"status":"pass" if numeric and n/elapsed>=GENERATOR_BURSTS_PER_SECOND else "fail",
              "n_catalogs":len(seeds),"n_bursts":n,"observed_size_min":int(sizes.min()),"observed_size_max":int(sizes.max()),
              "cold_seconds":elapsed,"cold_bursts_per_second":n/elapsed,
              "warm_seconds":warm_elapsed,"warm_bursts_per_second":n/warm_elapsed,
              "performance_tolerance":GENERATOR_BURSTS_PER_SECOND,"cold_setup_included":True,
              "bit_identical_replay":identical,"batch_grouping_invariant":batch_invariant,
              "seed_overlap_rejected_at_entry":overlap_rejected,"schema_ok":schema_ok,"finite":bool(finite),
              "detection_support":support,"distance_relative_error":distance_error,
              "max_cosmic_quantile_interpolation_relative_error":max(interpolated_errors),
              "numeric_checks_pass":numeric,"conditional_joint_vs_rejection":joint,
              "fingerprint":fingerprint(batch),"profiling_evidence":"results/generator-profile.txt",
              "inspection":"No per-burst Python loops: RNG allocation loops over catalogs; cosmic interpolation loops over fixed PDF nodes; LF inversion uses fixed vectorized bisection. Background quadrature loops only over cached grid nodes.",
              "not_run":"No training, network or posterior acceptance. This run generates audit catalogs only."}
    (ROOT/"results/generator-audit.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","n_catalogs","n_bursts","cold_bursts_per_second","warm_bursts_per_second","numeric_checks_pass","max_cosmic_quantile_interpolation_relative_error")}))
    return report


if __name__ == "__main__":
    result = run_audit()
    raise SystemExit(0 if result["status"]=="pass" else 1)
