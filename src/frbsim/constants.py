"""Central physical constants, units, sources, and confidence.

All dimensional computations use float64. Mathematical integers and numerical
algorithm settings are not empirical physical constants.
"""
from dataclasses import dataclass
import numpy as np
from astropy import constants as ac, units as u

@dataclass(frozen=True)
class Constant:
    name: str
    value: float
    unit: str
    source: str
    confidence: str

PLANCK_SOURCE = "https://arxiv.org/abs/1807.06209; SPEC-FINAL v2.1 pinned values"
MACQUART_SOURCE = "https://arxiv.org/abs/2005.13161"
REGISTRY = {}

def register(name, value, unit, source, confidence="paper-verified"):
    REGISTRY[name] = Constant(name, float(value), unit, source, confidence)
    return np.float64(value)

C = register("c", ac.c.si.value, "m s^-1", ac.c.reference, "SI exact")
G = register("G", ac.G.si.value, "m^3 kg^-1 s^-2", ac.G.reference, "CODATA")
MP = register("m_p", ac.m_p.si.value, "kg", ac.m_p.reference, "CODATA")
PC_M = register("pc", u.pc.to(u.m), "m", "Astropy parsec definition", "unit definition")
MPC_M = register("Mpc", u.Mpc.to(u.m), "m", "Astropy SI unit conversion", "unit definition")
DM_SI = register("one_DM_in_SI", (u.pc/u.cm**3).to(u.m**-2), "m^-2", "1 pc cm^-3", "unit definition")
H0 = register("H0", 67.4, "km s^-1 Mpc^-1", PLANCK_SOURCE)
OM = register("Omega_m", .315, "1", PLANCK_SOURCE)
OB = register("Omega_b", .049, "1", PLANCK_SOURCE)
OL = register("Omega_Lambda", .685, "1", PLANCK_SOURCE)
YHE = register("Y_He", .25, "1", MACQUART_SOURCE + " Eq. 2")
FD = register("f_d", .85, "1", "SPEC-FINAL v2.1; " + MACQUART_SOURCE + " Methods diffuse fraction")
F_DEFAULT = register("F", .32, "1", "SPEC-FINAL v2.1; James et al. 2022b MNRAS 516 4862")
HOST_MEDIAN = register("host_median", 100, "pc cm^-3", "SPEC-FINAL v2.1 choice within Macquart Eq. 3 ranges")
HOST_SIGMA = register("host_sigma_ln", .8, "1", "SPEC-FINAL v2.1 choice within Macquart Eq. 3 ranges")
MW_HALO = register("MW_halo", 50, "pc cm^-3", MACQUART_SOURCE)
H0_SI = H0 * u.km.to(u.m) / MPC_M
ELECTRON_FRACTION = 1 - YHE / 2

def column_dm(density_cm3, length_pc):
    """An unredshifted column; 1 cm^-3 over 1 pc is exactly 1 DM."""
    return np.asarray(density_cm3, dtype=np.float64) * np.asarray(length_pc, dtype=np.float64)
