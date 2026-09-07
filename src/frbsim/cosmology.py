"""Pinned flat matter + Lambda background; float64 and distances in Mpc."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
from .constants import H0, OM, OB, OL, C
from .numerics import QUAD_EPS

ASTROPY_COSMO = FlatLambdaCDM(H0=H0, Om0=OM, Ob0=OB, Tcmb0=0)

def redshift(z):
    z = np.asarray(z, dtype=np.float64)
    if np.any(~np.isfinite(z)) or np.any(z < 0):
        raise ValueError("redshift must be finite and nonnegative")
    return z

def E(z):
    return np.sqrt(OM * (1 + redshift(z))**3 + OL)

@lru_cache(maxsize=32768)
def _chi(z):
    return C / 1000 / H0 * quad(lambda t: 1/E(t), 0, z,
                              epsabs=QUAD_EPS, epsrel=QUAD_EPS)[0]

def comoving_distance(z):
    a = redshift(z)
    out = np.array([_chi(float(x)) for x in a.ravel()], dtype=np.float64).reshape(a.shape)
    return float(out) if out.ndim == 0 else out

def luminosity_distance(z):
    return (1 + redshift(z)) * comoving_distance(z)

def differential_volume(z):
    """Full-sky dVc/dz in Mpc^3."""
    return 4*np.pi * comoving_distance(z)**2 * (C/1000/H0) / E(z)
