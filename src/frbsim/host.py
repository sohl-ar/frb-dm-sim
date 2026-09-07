"""Host rest DM is lognormal in NATURAL LOG per Macquart Eq. 3.

zdm log10 parameters must both be multiplied by ln(10) before use.
The median and ln-width do not evolve with z; observed DM divides by 1+z.
"""
import numpy as np
from .constants import HOST_MEDIAN, HOST_SIGMA

def sample_host(n, rng, median=HOST_MEDIAN, sigma_ln=HOST_SIGMA):
    if not np.isfinite(median) or not 20 <= median <= 200:
        raise ValueError("host median must be in [20,200] pc cm^-3")
    if not np.isfinite(sigma_ln) or not .2 <= sigma_ln <= 2:
        raise ValueError("natural-log host width must be in [.2,2]")
    return rng.lognormal(np.log(median), sigma_ln, n).astype(np.float64)

def observed_host(dm_rest, z):
    z = np.asarray(z, dtype=np.float64)
    if np.any(~np.isfinite(z)) or np.any(z < 0):
        raise ValueError("finite nonnegative host z required")
    return np.asarray(dm_rest, dtype=np.float64)/(1+z)

def from_log10(mu10, sigma10):
    return float(mu10*np.log(10)), float(sigma10*np.log(10))
