from pathlib import Path
import numpy as np
import pytest
from frbsim.population import RedshiftPopulation, sample_redshifts

pytestmark = pytest.mark.gates

def test_g9(gate):
    p = RedshiftPopulation()
    z = sample_redshifts(10_000, np.random.default_rng(9), population=p)
    conventions = (Path(__file__).resolve().parents[2]/"CONVENTIONS.md").read_text()
    gate("G9", "default_redshift_floor", {"configured_minimum": p.z_min, "sample_minimum": float(z.min())},
         {"minimum": .05}, bool(p.z_min == .05 and np.all(z >= .05)), hard=False)
    gate("G9", "peculiar_velocity_documentation", {"documented": "peculiar velocities" in conventions},
         {"required": True}, "peculiar velocities" in conventions, hard=False)
