import numpy as np
import pytest
from frbsim.fields import uniform_shell_rays
from frbsim.mean_dm import transcription
from frbsim import tolerances as t

pytestmark = pytest.mark.gates

def test_g2(gate):
    for z in [.1, .5, 1., 1.5]:
        a, b = [uniform_shell_rays(z, n_shells=n) for n in [24, 48]]
        relative = float(np.max(np.abs(a/transcription(z)-1)))
        gate("G2", f"uniform_1000_rays_z={z}", relative, {"relative": t.UNIFORM_RAY_REL},
             relative <= t.UNIFORM_RAY_REL)
        convergence = float(np.max(np.abs(b/a-1)))
        gate("G2", f"shell_doubling_z={z}", convergence, {"relative": t.SHELL_CONVERGENCE_REL},
             convergence <= t.SHELL_CONVERGENCE_REL)
