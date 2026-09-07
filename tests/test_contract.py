"""Engineering tests, not substitutes for L2 acceptance gates."""
import numpy as np
import pytest
from frbsim.schema import FRBObservation, Parameters
from frbsim.catalog import generate_catalog, canonical_json
from frbsim.demo import fixture_config
from frbsim.selection import fluence_from_luminosity

def test_no_latent_redshift_in_unlocalized_observation():
    args = dict(ra_deg=1., dec_deg=2., dm_obs=100., dm_err=0., fluence_jy_ms=1.,
                snr=1., is_localized=False, z_obs=None, survey_id="test")
    row = FRBObservation(**args)
    with pytest.raises(ValueError, match="iff localized"):
        FRBObservation(**{**args, "z_obs": .5})
    with pytest.raises((AttributeError, TypeError)):
        row.z_true = .5
    with pytest.raises(ValueError):
        FRBObservation(**{**args, "is_localized": True})

def test_seed_and_component_accounting():
    surveys, lf, mw = fixture_config()
    a = generate_catalog(100, Parameters(), surveys, lf, 2020, mw_model=mw)
    b = generate_catalog(100, Parameters(), surveys, lf, 2020, mw_model=mw)
    assert canonical_json(a.simulator_payload()) == canonical_json(b.simulator_payload())
    for obs, latent in zip(a.observations, a.ground_truth):
        expected = latent.dm_cosmic+latent.dm_host_rest/(1+latent.z_true)+latent.dm_mw_ism+latent.dm_mw_halo
        assert obs.dm_obs == expected
    exported = a.observation_payload()
    assert all("theta" not in x and "z_true" not in x for x in exported)
    assert all(x["z_obs"] is None for x in exported if not x["is_localized"])

def test_fluence_units():
    distances = np.array([1., 10., 100.])
    fluence = np.array([1., 2., 3.])
    np.testing.assert_allclose(fluence_from_luminosity(4*np.pi*distances**2*fluence, distances), fluence)
