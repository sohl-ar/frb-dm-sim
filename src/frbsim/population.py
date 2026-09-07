"""Population redshifts and bounded phenomenological pseudo-luminosity.

The authorized p(z) is proportional to R(z) dVc/dz. R is used as a shape
proxy as specified; interpreting it as a source-clock transient rate would
require an observer-time conversion, which this implementation does not add.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.integrate import cumulative_trapezoid
from .cosmology import differential_volume
from .numerics import POPULATION_GRID_SIZE

SFR_SOURCE = "https://arxiv.org/html/1403.0007 Eq. 15; user-authorized exponent correction"
SFR_RISE, SFR_TURNOVER, SFR_FALL = 2.7, 2.9, 5.6

def sfr_shape(z):
    z = np.asarray(z, dtype=np.float64)
    return (1+z)**SFR_RISE/(1+((1+z)/SFR_TURNOVER)**SFR_FALL)

@dataclass(frozen=True)
class RedshiftPopulation:
    z_min: float = .05
    z_max: float = 1.5

    def __post_init__(self):
        if not np.isfinite(self.z_min+self.z_max) or not 0 < self.z_min < self.z_max:
            raise ValueError("finite 0<z_min<z_max required")

@lru_cache(maxsize=16)
def _redshift_cdf(population, family):
    z = np.linspace(population.z_min, population.z_max, POPULATION_GRID_SIZE)
    if family not in ("sfr", "constant_comoving"):
        raise ValueError("unknown population family")
    density = differential_volume(z)
    if family == "sfr":
        density *= sfr_shape(z)
    cdf = cumulative_trapezoid(density, z, initial=0)
    return z, cdf/cdf[-1]

def sample_redshifts(n, rng, family="sfr", population=RedshiftPopulation()):
    z, cdf = _redshift_cdf(population, family)
    return np.interp(rng.random(n), cdf, z)

@dataclass(frozen=True)
class LuminosityFunction:
    """L in Jy ms Mpc^2. Bounds and slope are explicit, with provenance.

    No unverified empirical default bounds or slope are supplied.
    kind='delta' is the spec-authorized geometry-test mode.
    """
    kind: str
    minimum: float
    maximum: float
    alpha: float
    source: str

    def __post_init__(self):
        if not self.source or not np.all(np.isfinite([self.minimum, self.maximum, self.alpha])):
            raise ValueError("finite LF parameters and provenance required")
        if not 0 < self.minimum <= self.maximum or self.kind not in ("delta", "power_law"):
            raise ValueError("invalid bounded luminosity function")
        if self.kind == "power_law" and self.minimum == self.maximum:
            raise ValueError("power-law LF needs distinct bounds")
        if self.kind == "delta" and self.minimum != self.maximum:
            raise ValueError("delta LF must have identical bounds")

    def sample(self, n, rng):
        if self.kind == "delta":
            return np.full(n, self.minimum, dtype=np.float64)
        p = rng.random(n)
        power = 1-self.alpha
        if power == 0:
            return self.minimum*np.exp(p*np.log(self.maximum/self.minimum))
        return (self.minimum**power+p*(self.maximum**power-self.minimum**power))**(1/power)
