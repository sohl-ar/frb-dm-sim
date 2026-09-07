"""L0 Macquart Eq. 4 inverse-CDF sampler and uniform-shell ray validation.

Halos are already included in the cosmic mean. No additive cosmic halo term.
The parameter F/sqrt(z) is an effective PDF width, NOT the distribution's
standard deviation. For beta=3 the untruncated second moment diverges.
L2 is not implemented or certified by this module.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from numpy.polynomial.legendre import leggauss
from .mean_dm import mean_dm, prefactor
from .cosmology import redshift, E
from .constants import FD, F_DEFAULT, C, G, MP, OB, H0_SI, ELECTRON_FRACTION, DM_SI
from .numerics import CDF_GRID_SIZE, DELTA_MIN, DELTA_MAX, ROOT_XTOL, POPULATION_GRID_SIZE

@dataclass(frozen=True)
class CosmicPDF:
    sigma: float
    c0: float
    log_A: float
    delta_max: float
    log_delta: np.ndarray
    cdf_values: np.ndarray
    density_in_log_delta: np.ndarray

    @property
    def mean(self):
        return float(np.trapezoid(np.exp(self.log_delta)*self.density_in_log_delta, self.log_delta))

    @property
    def truncated_std(self):
        second = np.trapezoid(np.exp(2*self.log_delta)*self.density_in_log_delta, self.log_delta)
        return float(np.sqrt(second-self.mean**2))

    def ppf(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be in [0,1]")
        # Remove flat underflow sections, retaining both CDF endpoints.
        keep = np.concatenate(([True], np.diff(self.cdf_values) > 0))
        return np.exp(np.interp(p, self.cdf_values[keep], self.log_delta[keep]))

    def sample(self, n, rng):
        return self.ppf(rng.random(n))

@lru_cache(maxsize=512)
def cosmic_pdf(sigma, delta_max=DELTA_MAX, grid_size=CDF_GRID_SIZE):
    """Solve normalization and <Delta>=1 on a documented finite domain.

    Work in x=ln Delta: p(x) proportional to exp[-2x -
    (exp(-3x)-C0)^2/(18 sigma^2)]. The Jacobian is included. A log
    density shift prevents underflow without altering the published PDF.
    """
    sigma, delta_max = float(sigma), float(delta_max)
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(delta_max) or delta_max <= 1:
        raise ValueError("positive sigma and finite delta_max>1 required")
    x = np.linspace(np.log(DELTA_MIN), np.log(delta_max), grid_size, dtype=np.float64)
    delta = np.exp(x)
    def weights(c0):
        logw = -2*x - (np.exp(-3*x)-c0)**2/(18*sigma**2)
        shift = np.max(logw)
        w = np.exp(logw-shift)
        norm = np.trapezoid(w, x)
        return w/norm, shift+np.log(norm)
    def residual(c0):
        w, _ = weights(c0)
        return np.trapezoid(delta*w, x)-1
    lo, hi = -1., 1.
    for _ in range(60):
        if residual(lo) >= 0 >= residual(hi):
            break
        lo *= 2
        hi *= 2
    else:
        raise RuntimeError("STOP: C0 root could not be bracketed")
    c0 = brentq(residual, lo, hi, xtol=ROOT_XTOL)
    density, lognorm = weights(c0)
    cdf = cumulative_trapezoid(density, x, initial=0)
    cdf /= cdf[-1]
    for a in (x, cdf, density):
        a.setflags(write=False)
    return CosmicPDF(sigma, float(c0), float(-lognorm), delta_max, x, cdf, density)

@lru_cache(maxsize=16)
def _mean_table(zmax, f_d):
    z = np.linspace(0., zmax, POPULATION_GRID_SIZE)
    cumulative = cumulative_trapezoid((1+z)/E(z), z, initial=0)*prefactor()*f_d
    return PchipInterpolator(z, cumulative, extrapolate=False)

def sample_cosmic(z, rng, f_d=FD, F=F_DEFAULT):
    """Seeded independent L0 draws; tabulate in sigma for heterogeneous z.

    Quantiles are linearly interpolated between 128 logarithmic sigma nodes.
    At fixed redshift the exact corresponding CDF table is used. Neither
    samples nor individual catalogs are rescaled to force the sample mean.
    """
    z = redshift(z)
    if np.any(z <= 0) or not .02 <= F <= 1 or not .6 <= f_d <= 1:
        raise ValueError("z>0, F in [.02,1], f_d in [.6,1] required")
    flat = z.ravel()
    if flat.size == 0:
        return np.empty_like(z)
    sigma = F/np.sqrt(flat)
    p = rng.random(flat.size)
    if np.ptp(sigma) == 0:
        delta = cosmic_pdf(float(sigma[0])).ppf(p)
    else:
        nodes = np.geomspace(sigma.min(), sigma.max(), 128)
        j = np.clip(np.searchsorted(nodes, sigma)-1, 0, len(nodes)-2)
        fraction = (sigma-nodes[j])/(nodes[j+1]-nodes[j])
        delta = np.empty(flat.size)
        for i in np.unique(j):
            mask = j == i
            q0 = cosmic_pdf(float(nodes[i])).ppf(p[mask])
            q1 = cosmic_pdf(float(nodes[i+1])).ppf(p[mask])
            delta[mask] = q0*(1-fraction[mask])+q1*fraction[mask]
    means = _mean_table(float(flat.max()), float(f_d))(flat)
    return (means*delta).reshape(z.shape)

def uniform_shell_rays(z, n_sightlines=1000, n_shells=24, f_d=FD):
    """Integrate physical n_e, proper length, and observed delay per shell.

    Three-point Gauss-Legendre integration inside each shell; all directions
    through a spatially uniform shell must give identical columns.
    """
    z = float(redshift(z))
    if n_shells < 1 or n_sightlines < 1:
        raise ValueError("positive ray and shell counts required")
    edges = np.linspace(0., z, n_shells+1)
    nodes, weights = leggauss(3)
    n0 = f_d*OB*3*H0_SI**2/(8*np.pi*G*MP)*ELECTRON_FRACTION
    columns = np.zeros(n_sightlines, dtype=np.float64)
    for left, right in zip(edges[:-1], edges[1:]):
        t = (right+left)/2 + (right-left)/2*nodes
        density = n0*(1+t)**3
        proper_length_per_z = C/(H0_SI*(1+t)*E(t))
        observed_column = density*proper_length_per_z/(1+t)/DM_SI
        columns += (right-left)/2*np.dot(weights, observed_column)
    return columns
