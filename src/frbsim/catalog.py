"""Seeded L0 forward catalog; host parameters use NATURAL LOG (Macquart Eq. 3).

Only detected records enter the catalog. Metadata and latents are simulator
side; the observation-only export excludes both. Surveys, LF and instrumental
noise are explicitly configured pending verification of scientific defaults.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from .schema import Parameters, Catalog, GroundTruth, FRBObservation
from .population import sample_redshifts, RedshiftPopulation
from .fields import sample_cosmic
from .host import sample_host, observed_host
from .cosmology import luminosity_distance
from .selection import fluence_from_luminosity, detected
from .mw import GalacticISM
from .constants import MW_HALO

def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

def git_provenance():
    root = Path(__file__).resolve().parents[2]
    def run(*args):
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}

def generate_catalog(n_intrinsic, theta, surveys, luminosity_function, seed, *,
                     population=RedshiftPopulation(), mw_model=None, halo=MW_HALO):
    """Generate n intrinsic bursts and return those passing their survey cut.

    Multiple survey configurations are mixed equally as an explicit simulator
    design; this is not a prediction of survey exposure or yield. Each RNG
    component has its own spawned stream. No redshift errors are yet modeled.
    """
    if not isinstance(theta, Parameters) or not surveys or n_intrinsic < 1:
        raise ValueError("validated theta, nonempty surveys and n_intrinsic>=1 required")
    if type(seed) is not int or seed < 0 or not 50 <= halo <= 80:
        raise ValueError("explicit nonnegative integer seed and halo in [50,80] required")
    mw_model = GalacticISM() if mw_model is None else mw_model
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8)]
    rz, rl, rsky, rcosmic, rhost, rnoise, rlocal, rsurvey = rngs
    z = sample_redshifts(n_intrinsic, rz, theta.population_family, population)
    L = luminosity_function.sample(n_intrinsic, rl)
    ra = rsky.uniform(0., 360., n_intrinsic)
    dec = np.rad2deg(np.arcsin(rsky.uniform(-1., 1., n_intrinsic)))
    fluence = fluence_from_luminosity(L, luminosity_distance(z))
    survey_index = rsurvey.integers(0, len(surveys), n_intrinsic)
    keep = np.zeros(n_intrinsic, dtype=bool)
    for i, survey in enumerate(surveys):
        mask = survey_index == i
        keep[mask] = detected(fluence[mask], survey.fluence_min)
    z, L, ra, dec, fluence, survey_index = [a[keep] for a in (z, L, ra, dec, fluence, survey_index)]
    n = len(z)
    cosmic = sample_cosmic(z, rcosmic, theta.f_d, theta.F)
    host = sample_host(n, rhost, theta.host_median, theta.host_sigma_ln)
    ism = mw_model.evaluate(ra, dec)
    errors = np.array([surveys[i].dm_error for i in survey_index])
    dm_obs = cosmic + observed_host(host, z) + ism + halo + rnoise.normal(0., errors)
    localized = rlocal.random(n) < np.array([surveys[i].localized_fraction for i in survey_index])
    obs, truth = [], []
    for i in range(n):
        s = surveys[survey_index[i]]
        obs.append(FRBObservation(float(ra[i]), float(dec[i]), float(dm_obs[i]), float(errors[i]),
            float(fluence[i]), float(fluence[i]/s.noise_equivalent_fluence), bool(localized[i]),
            float(z[i]) if localized[i] else None, s.survey_id))
        truth.append(GroundTruth(float(z[i]), float(cosmic[i]), float(host[i]),
                                float(ism[i]), float(halo), float(L[i]), theta))
    config = {"surveys": [asdict(s) for s in surveys], "luminosity": asdict(luminosity_function),
              "population": asdict(population), "mw": {"class": type(mw_model).__name__, **asdict(mw_model)},
              "halo": float(halo), "n_intrinsic": n_intrinsic, "level": "L0",
              "survey_mixture": "equal_probability"}
    metadata = {"seed": seed, **git_provenance(), "theta": asdict(theta), "config": config,
                "config_hash": hashlib.sha256(canonical_json(config).encode()).hexdigest(),
                "n_detected": n, "acceptance_level": "L0; not L2 certified"}
    return Catalog(tuple(obs), tuple(truth), metadata)
