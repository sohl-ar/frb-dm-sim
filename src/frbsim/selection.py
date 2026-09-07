"""Uniform-sky fluence threshold with explicitly configured survey assumptions.

Authorized v1: L [Jy ms Mpc^2] / (4 pi DL[Mpc]^2) is fluence [Jy ms].
Physical emitted energy, bandwidth and spectral K corrections are deferred.
"""
from dataclasses import dataclass
import numpy as np

def fluence_from_luminosity(L, distance_mpc):
    L, distance_mpc = np.broadcast_arrays(np.asarray(L, dtype=np.float64),
                                         np.asarray(distance_mpc, dtype=np.float64))
    if np.any(~np.isfinite(L)) or np.any(L < 0) or np.any(~np.isfinite(distance_mpc)) or np.any(distance_mpc <= 0):
        raise ValueError("finite L>=0 and positive finite DL required")
    return L/(4*np.pi*distance_mpc**2)

def detected(fluence, threshold):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("nonnegative finite threshold required")
    return np.asarray(fluence, dtype=np.float64) >= threshold

@dataclass(frozen=True)
class Survey:
    survey_id: str
    fluence_min: float
    localized_fraction: float
    dm_error: float
    noise_equivalent_fluence: float
    source: str

    def __post_init__(self):
        if not self.survey_id or not self.source:
            raise ValueError("survey identifier and provenance required")
        if not np.all(np.isfinite([self.fluence_min, self.localized_fraction,
                                   self.dm_error, self.noise_equivalent_fluence])):
            raise ValueError("finite survey parameters required")
        if self.fluence_min < 0 or self.dm_error < 0 or self.noise_equivalent_fluence <= 0:
            raise ValueError("invalid threshold or measurement-noise configuration")
        if not 0 <= self.localized_fraction <= 1:
            raise ValueError("localization fraction must be in [0,1]")
