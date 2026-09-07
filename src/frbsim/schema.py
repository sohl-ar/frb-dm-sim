"""Strict observation/latent boundary. Host theta uses NATURAL LOG width.

Observations never embed simulator metadata, parameters or ground truth.
Use Catalog.observation_payload() for the downstream inference boundary.
"""
from dataclasses import dataclass, asdict
from typing import Literal
import math
from .constants import FD, F_DEFAULT, HOST_MEDIAN, HOST_SIGMA

@dataclass(frozen=True, slots=True)
class Parameters:
    f_d: float = float(FD)
    F: float = float(F_DEFAULT)
    host_median: float = float(HOST_MEDIAN)
    host_sigma_ln: float = float(HOST_SIGMA)
    population_family: Literal["sfr", "constant_comoving"] = "sfr"

    def __post_init__(self):
        for name, bounds in [("f_d", (.6, 1)), ("F", (.02, 1)),
                             ("host_median", (20, 200)), ("host_sigma_ln", (.2, 2))]:
            value = getattr(self, name)
            if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{name} must be in {bounds}")
        if self.population_family not in ("sfr", "constant_comoving"):
            raise ValueError("unknown population family")

@dataclass(frozen=True, slots=True)
class FRBObservation:
    ra_deg: float
    dec_deg: float
    dm_obs: float
    dm_err: float
    fluence_jy_ms: float
    snr: float
    is_localized: bool
    z_obs: float | None
    survey_id: str

    def __post_init__(self):
        for name in ("ra_deg", "dec_deg", "dm_obs", "dm_err", "fluence_jy_ms", "snr"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"nonfinite observation {name}")
        if not 0 <= self.ra_deg < 360 or not -90 <= self.dec_deg <= 90:
            raise ValueError("invalid sky coordinate")
        if self.dm_err < 0 or self.fluence_jy_ms < 0 or self.snr < 0:
            raise ValueError("errors, fluence and SNR must be nonnegative")
        # Explicit exception survives python -O, unlike a bare assert.
        if type(self.is_localized) is not bool or self.is_localized != (self.z_obs is not None):
            raise ValueError("z_obs must be present iff localized")
        if self.z_obs is not None and (not math.isfinite(self.z_obs) or self.z_obs < 0):
            raise ValueError("invalid observed redshift")
        if not isinstance(self.survey_id, str) or not self.survey_id:
            raise ValueError("survey_id required")

@dataclass(frozen=True, slots=True)
class GroundTruth:
    z_true: float
    dm_cosmic: float
    dm_host_rest: float
    dm_mw_ism: float
    dm_mw_halo: float
    L: float
    theta: Parameters

@dataclass(frozen=True, slots=True)
class Catalog:
    observations: tuple[FRBObservation, ...]
    ground_truth: tuple[GroundTruth, ...]
    metadata: dict

    def __post_init__(self):
        if len(self.observations) != len(self.ground_truth):
            raise ValueError("observation/latent record count mismatch")

    def observation_payload(self):
        return [asdict(row) for row in self.observations]

    def simulator_payload(self):
        return {"observations": self.observation_payload(),
                "ground_truth": [asdict(row) for row in self.ground_truth], "metadata": self.metadata}
