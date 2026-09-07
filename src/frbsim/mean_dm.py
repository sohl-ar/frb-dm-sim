"""Four computation paths for observed cosmic DM; diffuse fraction includes halos.

(a) SI transcription, (b) CGS scale-factor path derivation, (c) comoving
sphere ray ODE, (d) unmodified authors' density and summation functions under
explicitly matched cosmology, diffuse fraction and ionization assumptions.
"""
from functools import lru_cache
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad, solve_ivp
from astropy import constants as ac, units as u
from .constants import C, G, MP, H0_SI, OM, OL, OB, FD, ELECTRON_FRACTION, DM_SI, MPC_M
from .cosmology import E, redshift, comoving_distance, ASTROPY_COSMO
from .numerics import QUAD_EPS, ODE_RTOL, ODE_ATOL, AUTHOR_NEVAL

def prefactor():
    return 3*C*H0_SI*OB*ELECTRON_FRACTION/(8*np.pi*G*MP)/DM_SI

def integral(z):
    return quad(lambda x: (1+x)/E(x), 0, float(redshift(z)),
                epsabs=QUAD_EPS, epsrel=QUAD_EPS)[0]

def _validate(z, f_d):
    z = float(redshift(z))
    if not np.isfinite(f_d) or not 0 <= f_d <= 1:
        raise ValueError("f_d must be finite in [0,1]")
    return z

def transcription(z, f_d=FD):
    """(a) n_e(z) and Eq. (2), with SI column-to-DM conversion."""
    z = _validate(z, f_d)
    rho_c0 = 3*H0_SI**2 / (8*np.pi*G)
    def integrand(t):
        rho_b = OB*rho_c0*(1+t)**3
        ne_physical = f_d*rho_b/MP*ELECTRON_FRACTION
        return C*ne_physical/(H0_SI*(1+t)**2*E(t))/DM_SI
    return quad(integrand, 0, z, epsabs=QUAD_EPS, epsrel=QUAD_EPS)[0]

def first_principles(z, f_d=FD):
    """(b) CGS + scale factor a: n=n0/a^3, dl=c da/(aH), weight=a."""
    z = _validate(z, f_d)
    h = ASTROPY_COSMO.H0.to_value(u.s**-1)
    rho = (3*(h/u.s)**2/(8*np.pi*ac.G)).to_value(u.g/u.cm**3)
    n0 = f_d*OB*rho/ac.m_p.cgs.value * (3/4 + 2*(1/4)/4)
    def integrand(a):
        ne = n0/a**3
        dl_da_cm = ac.c.cgs.value/(a*h*np.sqrt(OM/a**3+OL))
        return ne*dl_da_cm*a / u.pc.to(u.cm)
    return quad(integrand, 1/(1+z), 1, epsabs=QUAD_EPS, epsrel=QUAD_EPS)[0]

def box_raytrace(z, f_d=FD):
    """(c) Uniform comoving-density sphere, radial ray in comoving Mpc.

    Evolve z(chi) alongside DM. A proper segment is dchi/(1+z),
    followed by another delay factor 1/(1+z). No analytic DM is called.
    """
    z = _validate(z, f_d)
    if z == 0:
        return 0.
    n0 = f_d*OB*3*H0_SI**2/(8*np.pi*G*MP)*ELECTRON_FRACTION
    def rhs(chi, state):
        zr = state[0]
        physical = n0*(1+zr)**3
        return [H0_SI*np.sqrt(OM*(1+zr)**3+OL)*MPC_M/C,
                physical*MPC_M/(1+zr)**2/DM_SI]
    out = solve_ivp(rhs, (0, comoving_distance(z)), [0., 0.],
                    rtol=ODE_RTOL, atol=ODE_ATOL)
    if not out.success:
        raise RuntimeError(out.message)
    return float(out.y[1, -1])

@lru_cache(maxsize=1)
def _author_tree():
    root = Path(__file__).parent / "vendor"
    raw = (root / "igm.py.txt").read_bytes()
    provenance = json.loads((root / "provenance.json").read_text())
    if hashlib.sha256(raw).hexdigest() != provenance["sha256"]:
        raise RuntimeError("Pinned authors' source hash mismatch")
    tree = ast.parse(raw.decode())
    names = {"ne_cosmic", "average_DM"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError("Authors' function interface changed")
    return compile(ast.Module(body=selected, type_ignores=[]), "FRBs/FRB/igm.py", "exec")

def authors_code(z, f_d=FD):
    """(d) Execute pinned upstream functions, not a rewritten formula.

    Boundary callbacks select constant f_d and fully ionized H/He, as required
    by this spec. The upstream default evolving baryon census is a different
    model. Its density assembly and rectangular summation remain unmodified.
    """
    z = _validate(z, f_d)
    def diffuse(z, cosmo, return_rho=False):
        fraction = np.full_like(np.asarray(z, dtype=np.float64), f_d)
        rho = cosmo.Ob0*cosmo.critical_density0*(1+z)**3*fraction
        return (fraction, rho) if return_rho else fraction
    ns = {"np": np, "constants": ac, "defs": SimpleNamespace(frb_cosmo=ASTROPY_COSMO),
          "f_diffuse": diffuse, "average_fHI": lambda x: np.zeros_like(x),
          "average_He_nume": lambda x: np.full_like(x, 2.)}
    exec(_author_tree(), ns)
    return ns["average_DM"](z, cosmo=ASTROPY_COSMO, neval=AUTHOR_NEVAL).to_value(u.pc/u.cm**3)

def mean_dm(z, f_d=FD):
    a = redshift(z)
    values = np.array([transcription(float(x), f_d) for x in a.ravel()]).reshape(a.shape)
    return float(values) if values.ndim == 0 else values
