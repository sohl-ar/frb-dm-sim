"""Vectorized L0 training catalogs, conditioned on observed count.

Host DM uses NATURAL LOG parameters per Macquart Eq.3. The ISM=30,
halo=50, zero DM error and unit noise-equivalent fluence are explicitly
inherited engineering fixtures, not calibrated survey measurements.
Only the Schechter LF and fluence threshold have the D4/D5 authorization.
No ML is implemented here. Independent per-catalog seeds persist across
batch grouping. Tables use the existing asymmetric cosmic PDF unchanged.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
import numpy as np
from scipy.integrate import cumulative_trapezoid
from frbsim.cosmology import luminosity_distance, differential_volume
from frbsim.population import sfr_shape
from frbsim.fields import cosmic_pdf, _mean_table
from frbsim.catalog import git_provenance
from frbsim.numerics import POPULATION_GRID_SIZE
from .selection import SchechterLF, FLUENCE_MIN, SURVEY_SOURCE
from .preflight import SEED_RANGES, assert_disjoint_seed_ranges

SIGMA_GRID_SIZE = 128  # Same interpolation strategy/count as Phase 1; fixed domain for batch replay.
Z_MIN, Z_MAX = .05, 1.5
PRIOR_LO = np.array([.6,.05,20.,.2],dtype=np.float64)
PRIOR_HI = np.array([1.,1.,200.,2.],dtype=np.float64)
LOG_PRIOR = np.array([False,True,True,False])
ISM_FIXTURE, HALO_FIXTURE, DM_ERROR_FIXTURE, NOISE_FLUENCE_FIXTURE = 30.,50.,0.,1.


@dataclass(frozen=True)
class Observations:
    """No latents. z_obs holds ONLY localized values, in mask order.

    Compact redshifts avoid both NaN sentinels and hidden unlocalized z.
    The future encoder may scatter these values using is_localized.
    Host parameters, population family and theta are absent.
    """
    offsets: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    dm_obs: np.ndarray
    dm_err: np.ndarray
    fluence_jy_ms: np.ndarray
    snr: np.ndarray
    is_localized: np.ndarray
    z_obs: np.ndarray
    survey_id: str = "phase2a-CHIME-selection-proxy-with-engineering-noise"

    def __post_init__(self):
        n = len(self.dm_obs)
        if (self.offsets.ndim != 1 or self.offsets.dtype.kind not in "iu" or self.offsets[0] != 0
                or self.offsets[-1] != n or np.any(np.diff(self.offsets)<=0)):
            raise ValueError("invalid catalog offsets")
        if self.is_localized.dtype != np.bool_ or self.is_localized.shape != (n,):
            raise ValueError("localized mask required")
        if self.z_obs.shape != (np.count_nonzero(self.is_localized),):
            raise ValueError("redshift must exist iff localized; no unlocalized redshift slots")
        for a in (self.ra_deg,self.dec_deg,self.dm_obs,self.dm_err,self.fluence_jy_ms,self.snr):
            if a.shape != (n,) or a.dtype != np.float64 or not np.all(np.isfinite(a)):
                raise ValueError("finite float64 observable arrays required")
        if (self.z_obs.dtype != np.float64 or not np.all(np.isfinite(self.z_obs))
                or np.any(self.z_obs<Z_MIN) or np.any(self.z_obs>Z_MAX)
                or np.any(self.ra_deg<0) or np.any(self.ra_deg>=360)
                or np.any(abs(self.dec_deg)>90) or np.any(self.dm_err<0)
                or np.any(self.fluence_jy_ms<0) or np.any(self.snr<0)):
            raise ValueError("invalid observable domain")


@dataclass(frozen=True)
class Latents:
    """Simulator only; host-median and host-sigma_ln are natural-log convention."""
    theta: np.ndarray
    family: np.ndarray
    z_true: np.ndarray
    dm_cosmic: np.ndarray
    dm_host_rest: np.ndarray
    L: np.ndarray
    dm_mw_ism: np.ndarray
    dm_mw_halo: np.ndarray


@dataclass(frozen=True)
class TrainingBatch:
    observations: Observations
    latents: Latents
    metadata: dict


@lru_cache(maxsize=8)
def population_tables(lf=SchechterLF(), threshold=FLUENCE_MIN, selection_on=True):
    if not np.isfinite(threshold) or threshold<=0:
        raise ValueError("positive finite fluence cut required")
    z = np.linspace(Z_MIN,Z_MAX,POPULATION_GRID_SIZE,dtype=np.float64)
    dl = luminosity_distance(z)
    volume = differential_volume(z)
    survival = lf.tail(4*np.pi*dl**2*threshold) if selection_on else np.ones_like(z)
    cdfs = []
    fractions = []
    for family in ("sfr","constant_comoving"):
        p = volume*(sfr_shape(z) if family == "sfr" else 1.)
        cdf = cumulative_trapezoid(p*survival,z,initial=0.)
        fractions.append(float(cdf[-1]/np.trapezoid(p,z)))
        cdf /= cdf[-1]
        cdf.setflags(write=False)
        cdfs.append(cdf)
    z.setflags(write=False)
    dl.setflags(write=False)
    return z,dl,tuple(cdfs),tuple(fractions)


@lru_cache(maxsize=1)
def cosmic_tables():
    nodes = np.geomspace(PRIOR_LO[1]/np.sqrt(Z_MAX),PRIOR_HI[1]/np.sqrt(Z_MIN),SIGMA_GRID_SIZE)
    # Loop over fixed numerical grid nodes, never individual bursts.
    pdfs = tuple(cosmic_pdf(float(s)) for s in nodes)
    return nodes,pdfs


def cosmic_from_uniforms(z, fd, F, uniforms):
    nodes,pdfs = cosmic_tables()
    sigma = F/np.sqrt(z)
    if np.any(sigma<nodes[0]) or np.any(sigma>nodes[-1]):
        raise ValueError("cosmic width outside configured prior/grid domain")
    j = np.clip(np.searchsorted(nodes,sigma)-1,0,len(nodes)-2)
    fraction = (sigma-nodes[j])/(nodes[j+1]-nodes[j])
    delta = np.empty_like(z)
    for i in np.unique(j):
        mask = j==i
        delta[mask] = ((1-fraction[mask])*pdfs[i].ppf(uniforms[mask])
                       +fraction[mask]*pdfs[i+1].ppf(uniforms[mask]))
    return _mean_table(Z_MAX,1.)(z)*fd*delta


def generate_batch(seeds, *, split, seed_ranges=None, lf=SchechterLF(), threshold=FLUENCE_MIN,
                   forced_n=None, localized_fraction=None, forced_family=None, selection_on=True):
    """Generate selected catalogs directly; N is observed, not intrinsic.

    Per-catalog RNG allocation is the only data-dependent Python loop.
    Luminosities, cosmic/host DM, sky, masks and observations are arrays.
    """
    ranges = SEED_RANGES if seed_ranges is None else seed_ranges
    assert_disjoint_seed_ranges(ranges)  # Mandatory at the actual training entry point.
    seeds = tuple(seeds)
    if split not in ranges or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique catalog seeds and known split required")
    r = ranges[split]
    if any(type(s) is not int or not r.start<=s<r.stop for s in seeds):
        raise ValueError("catalog seed outside its assigned split")
    if forced_n is not None and (type(forced_n) is not int or not 1<=forced_n<=1024):
        raise ValueError("diagnostic forced_n must be an integer in 1..1024")
    if localized_fraction is not None and (not np.isfinite(localized_fraction) or not 0<=localized_fraction<=1):
        raise ValueError("localized_fraction must be in [0,1]")
    if forced_family not in (None,0,1):
        raise ValueError("forced_family must be None, 0 (SFR), or 1 (constant)")
    zgrid,dlgrid,cdfs,detected_fractions = population_tables(lf,threshold,selection_on)
    sizes,theta,families,parts = [],[],[],[]
    bounds_lo = np.where(LOG_PRIOR,np.log(PRIOR_LO),PRIOR_LO)
    bounds_hi = np.where(LOG_PRIOR,np.log(PRIOR_HI),PRIOR_HI)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1,513))
        p = bounds_lo+rng.random(4)*(bounds_hi-bounds_lo)
        p = np.where(LOG_PRIOR,np.exp(p),p)
        family = int(rng.integers(0,2))  # Equal family mixture: explicit engineering allocation.
        local_probability = rng.beta(2.,6.)
        # Always consume the original draws, preserving default replay and paired
        # localized/unlocalized experiments' theta, DM, sky, and fluence streams.
        n = n if forced_n is None else forced_n
        family = family if forced_family is None else forced_family
        local_probability = local_probability if localized_fraction is None else localized_fraction
        # Independent coordinates of the RNG stream; never percentile grids.
        arrays = rng.random((6,n))
        z = np.interp(arrays[0],cdfs[family],zgrid)
        dl = np.interp(z,zgrid,dlgrid)
        host = rng.lognormal(np.log(p[2]),p[3],n)
        parts.append((z,dl,arrays[5],host,arrays[1],arrays[2],arrays[3],arrays[4]<local_probability))
        sizes.append(n)
        theta.append(p)
        families.append(family)
    # These comprehensions iterate over columns/catalogs, not bursts.
    z,dl,u_L,host,u_cosmic,u_ra,u_dec,localized = (
        np.concatenate([part[i] for part in parts]) for i in range(8))
    L = lf.quantile_above(4*np.pi*dl**2*threshold if selection_on else np.full_like(dl,lf.minimum),u_L)
    theta = np.asarray(theta,dtype=np.float64)
    per_burst_theta = np.repeat(theta,np.asarray(sizes),axis=0)
    cosmic = cosmic_from_uniforms(z,per_burst_theta[:,0],per_burst_theta[:,1],u_cosmic)
    n_total = len(z)
    offsets = np.concatenate(([0],np.cumsum(sizes,dtype=np.int64)))
    ism,halo = np.full(n_total,ISM_FIXTURE),np.full(n_total,HALO_FIXTURE)
    fluence = L/(4*np.pi*dl**2)
    obs = Observations(offsets,360*u_ra,np.rad2deg(np.arcsin(2*u_dec-1)),
                       cosmic+host/(1+z)+ism+halo,np.full(n_total,DM_ERROR_FIXTURE),
                       fluence,fluence/NOISE_FLUENCE_FIXTURE,localized,z[localized])
    truth = Latents(theta,np.asarray(families,dtype=np.int64),z,cosmic,host,L,ism,halo)
    config = {"LF":asdict(lf),"fluence_min":threshold,"survey_source":SURVEY_SOURCE,
              "redshift_interval":[Z_MIN,Z_MAX],"family_mixture":"equal probabilities, marginalized",
              "N":"DiscreteUniform inclusive 1..512, observed count","localization":"Beta(2,6) then Bernoulli",
              "prior_lo":PRIOR_LO.tolist(),"prior_hi":PRIOR_HI.tolist(),"log_prior":LOG_PRIOR.tolist(),
              "ISM_fixture":ISM_FIXTURE,"halo_fixture":HALO_FIXTURE,"dm_error_fixture":DM_ERROR_FIXTURE,
              "noise_equivalent_fluence_fixture":NOISE_FLUENCE_FIXTURE,
              "cosmic_sigma_nodes":SIGMA_GRID_SIZE,"population_grid_points":POPULATION_GRID_SIZE}
    if forced_n is not None or localized_fraction is not None or forced_family is not None or not selection_on:
        config["diagnostic_overrides"] = {"forced_n":forced_n,"localized_fraction":localized_fraction,
                                           "forced_family":forced_family,"selection_on":selection_on}
    metadata = {**git_provenance(),"seeds":seeds,"split":split,"config":config,
                "seed_ranges":{k:asdict(v) for k,v in ranges.items()},
                "config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                "detected_fraction_by_family":detected_fractions,
                "acceptance":"L0 engineering fixture; not Phase 1 acceptance"}
    return TrainingBatch(obs,truth,metadata)
