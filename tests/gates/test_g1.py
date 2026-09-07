import numpy as np
import pytest
from frbsim import mean_dm as dm
from frbsim import tolerances as t
from frbsim.constants import FD, column_dm
from frbsim.cosmology import ASTROPY_COSMO, comoving_distance, luminosity_distance

pytestmark = pytest.mark.gates

def test_g1(gate):
    gate("G1", "unit_column", float(column_dm(1, 1)), {"equals": 1.}, column_dm(1, 1) == 1)
    for label, value, bounds in [("K", dm.prefactor(), t.K_RANGE),
                                  ("integral_at_1", dm.integral(1), t.INTEGRAL_RANGE),
                                  ("DM_at_1", dm.transcription(1, FD), t.DM1_RANGE)]:
        gate("G1", label, value, {"range": bounds}, bounds[0] <= value <= bounds[1])
    zsmall = 1e-7
    slope = dm.box_raytrace(zsmall)/zsmall
    gate("G1", "low_z_slope", slope, {"anchor": t.LOW_Z_ANCHOR*FD, "relative": t.FORMULA_REL},
         abs(slope/(t.LOW_Z_ANCHOR*FD)-1) <= t.FORMULA_REL)
    for z in [.1, .5, 1., 1.5]:
        vals = [f(z) for f in [dm.transcription, dm.first_principles, dm.box_raytrace, dm.authors_code]]
        rel = (max(vals)-min(vals))/min(vals)
        gate("G1", f"four_way_z={z}", {"a_b_c_d": vals, "max_relative": rel},
             {"relative": t.FORMULA_REL}, rel <= t.FORMULA_REL)
        gate("G1", f"authors_z={z}", abs(vals[3]/vals[0]-1), {"relative": t.AUTHOR_REL},
             abs(vals[3]/vals[0]-1) <= t.AUTHOR_REL)
    zs = np.linspace(.001, 1.5, 30)
    for label, ours, oracle in [
        ("chi_astropy", comoving_distance(zs), ASTROPY_COSMO.comoving_distance(zs).value),
        ("DL_astropy", luminosity_distance(zs), ASTROPY_COSMO.luminosity_distance(zs).value)]:
        err = float(np.max(np.abs(ours/oracle-1)))
        gate("G1", label, err, {"relative": t.DISTANCE_REL}, err <= t.DISTANCE_REL)
