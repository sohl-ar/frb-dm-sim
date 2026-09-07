"""Galactic ISM via pygedm; local halo is separate from cosmological halos.

Official API: dist_to_dm(gl,gb,dist_pc,mode='gal',method='ymw16').
https://pygedm.readthedocs.io/en/latest/pygedm.html
Full-column convention: finite 30 kpc endpoint, pending 30/50 kpc
convergence validation. No fixed ISM is silently substituted for pygedm.
"""
from dataclasses import dataclass
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

@dataclass(frozen=True)
class GalacticISM:
    method: str = "ymw16"
    endpoint_pc: float = 30_000.

    def __post_init__(self):
        if self.method not in ("ymw16", "ne2001") or not 0 < self.endpoint_pc <= 50_000:
            raise ValueError("invalid pygedm method or Galactic endpoint")

    def evaluate(self, ra, dec):
        try:
            import pygedm
        except ImportError as exc:
            raise RuntimeError("BLOCKED: pygedm unavailable; no substitute ISM model is authorized") from exc
        coords = SkyCoord(ra=np.asarray(ra)*u.deg, dec=np.asarray(dec)*u.deg).galactic
        result = [pygedm.dist_to_dm(float(l), float(b), self.endpoint_pc,
                                   mode="gal", method=self.method)[0].to_value(u.pc/u.cm**3)
                  for l, b in zip(coords.l.deg.ravel(), coords.b.deg.ravel())]
        return np.asarray(result, dtype=np.float64).reshape(np.asarray(ra).shape)

@dataclass(frozen=True)
class FixedISMFixture:
    """Explicit replication/test-only fixture; never a scientific survey default."""
    dm: float
    source: str

    def __post_init__(self):
        if not np.isfinite(self.dm) or self.dm < 0 or not self.source:
            raise ValueError("nonnegative finite fixture DM and source required")

    def evaluate(self, ra, dec):
        return np.full(np.broadcast_shapes(np.shape(ra), np.shape(dec)), self.dm, dtype=np.float64)
