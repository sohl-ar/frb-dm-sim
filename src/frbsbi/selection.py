"""Authorized Schechter LF and exact conditioning on a hard fluence cut.

James 2022b, doi:10.1093/mnras/stac2524, Eq.14 and section 2.2.2;
fiducial Emax from Baptista Table 2. SPEC-02a consolidated authorization
defines L=4*pi*DL^2*fluence in Jy ms Mpc^2, with the fixed 1 GHz energy
conversion below. No extra 4*pi or redshift correction is inserted.
L_max is an exponential scale, not a sharp upper bound.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import exp1, gammaincc, gammaln

LF_SOURCE = "James 2022b doi:10.1093/mnras/stac2524 Eq.14, sec.2.2.2; Baptista Table 2; SPEC-02a authorization"
SURVEY_SOURCE = "CHIME Catalog 1 ApJS 257,59; SPEC-02a authorization: selection proxy"
ENERGY_PER_L = 9.52e31  # erg / (Jy ms Mpc^2), authorized 1 GHz convention
FLUENCE_MIN = 5.0  # Jy ms, authorized D4
FLUENCE_AXIS = 3.5  # Catalog 2 lower-limit sensitivity proxy, not a measured hard cut
INVERSE_ITERATIONS = 48  # float64 numerical bisection, not a physical cutoff


def log_upper_gamma(s, x):
    """Log Gamma(s,x), valid for s>-1 and x>0, including nonpositive s.

    For negative s use downward recurrence with log-space subtraction;
    s=0 uses E1. Never pass a nonpositive shape to scipy.gammaincc.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(s) or s <= -1 or np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("finite s>-1 and x>0 required")
    if s > 0:
        result = gammaln(s) + np.log(gammaincc(s, x))
    elif s == 0:
        result = np.log(exp1(x))
    else:
        log_b = s * np.log(x) - x
        log_a = gammaln(s + 1) + np.log(gammaincc(s + 1, x))
        result = log_b + np.log(-np.expm1(log_a - log_b)) - np.log(-s)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("STOP: upper-gamma evaluation outside verified numeric domain")
    return result


@dataclass(frozen=True)
class SchechterLF:
    gamma: float = -1.16
    log10_energy_scale: float = 41.84
    log10_energy_minimum: float = 30.0
    source: str = LF_SOURCE

    def __post_init__(self):
        if (not np.all(np.isfinite([self.gamma, self.log10_energy_scale, self.log10_energy_minimum]))
                or not -2 < self.gamma < 0 or self.log10_energy_minimum >= self.log10_energy_scale
                or not self.source):
            raise ValueError("finite -2<gamma<0, Emin<Escale and citation required")

    @property
    def minimum(self):
        return 10.0**self.log10_energy_minimum / ENERGY_PER_L

    @property
    def scale(self):
        return 10.0**self.log10_energy_scale / ENERGY_PER_L

    def log_tail(self, lower):
        lower = np.maximum(np.asarray(lower, dtype=np.float64), self.minimum)
        return (log_upper_gamma(self.gamma + 1, lower / self.scale)
                - log_upper_gamma(self.gamma + 1, self.minimum / self.scale))

    def tail(self, lower):
        return np.exp(self.log_tail(lower))

    def sample_above(self, lower, rng):
        return self.quantile_above(lower, rng.random(np.shape(lower)))

    def quantile_above(self, lower, probabilities):
        """Inverse survival sampling; only fixed vectorized bisection loops.

        For s<1 the kernel decreases faster than exp(-x). Therefore
        x_hi=x_lower-log(1-u) brackets the conditional quantile. There is
        no intrinsic rejection loop and no imposed maximum luminosity.
        """
        lower = np.maximum(np.asarray(lower, dtype=np.float64), self.minimum)
        probabilities = np.asarray(probabilities,dtype=np.float64)
        lower,probabilities = np.broadcast_arrays(lower,probabilities)
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities<0) or np.any(probabilities>=1):
            raise ValueError("finite probabilities in [0,1) required")
        if np.any(~np.isfinite(lower)):
            raise ValueError("finite luminosity threshold required")
        x0 = lower / self.scale
        log_survival = np.log1p(-probabilities)
        target = log_upper_gamma(self.gamma + 1, x0) + log_survival
        left = np.log(x0)
        right = np.log(x0 - log_survival)
        for _ in range(INVERSE_ITERATIONS):
            middle = (left + right) / 2
            above = log_upper_gamma(self.gamma + 1, np.exp(middle)) > target
            left = np.where(above, middle, left)
            right = np.where(above, right, middle)
        # Enforce support against the last floating-point exp/log rounding.
        return np.maximum(lower, self.scale * np.exp((left + right) / 2))
