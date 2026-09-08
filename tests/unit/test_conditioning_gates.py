"""Small positive/negative controls; no trained-posterior acceptance run."""
import json
import numpy as np
import pytest
import torch
from frbsbi.gate_metrics import contraction,information,tarp,smoke_summary,prior_quantiles
from frbsbi.generator import generate_batch,PRIOR_LO,PRIOR_HI
from frbsbi.data import encode_observations,pad_catalogs,ROOT
from frbsbi.evaluate import validation_catalogs
from frbsbi.macquart_smoke import smoke_features
from frbsbi.model import PosteriorFlow,ModelConfig
from frbsbi.train_conditioned import RUN_DIR


def prior_fixture(n=4):
    return np.repeat(prior_quantiles((np.arange(1000)+.5)/1000)[None,:,:],n,axis=0)


def test_contraction_prior_fails_and_contracted_control_passes():
    p = prior_fixture()
    assert contraction(p,.7)["status"]=="fail"
    assert contraction(p,.5)["status"]=="fail"
    center = p.mean(1,keepdims=True)
    assert contraction(center+.4*(p-center),.5)["status"]=="pass"


def test_information_positive_negative_controls():
    p = prior_fixture()
    center = p.mean(1,keepdims=True)
    assert information(p,p)["status"]=="fail"
    assert information(center+.4*(p-center),p)["status"]=="pass"


def test_tarp_sbi_diagonal_and_biased_controls():
    # Construct exact uniform distance ranks around the SAME independently drawn
    # references the adapter will use. This tests ECP plumbing, not calibration.
    seed,n,draws = 73300,100,1000
    refs = np.random.default_rng(seed).random((n,4))
    direction = np.zeros((n,4)); direction[:,0] = np.where(refs[:,0]<.5,1.,-1.)
    radii = (np.arange(draws)+.5)/draws/4
    unit_p = refs[:,None,:]+radii[None,:,None]*direction[:,None,:]
    truth = refs+((np.arange(n)+.5)/n/4)[:,None]*direction
    physical = lambda x: PRIOR_LO+x*(PRIOR_HI-PRIOR_LO)
    result = tarp(physical(unit_p),physical(truth),reference_seed=seed)
    assert result["status"]=="pass"
    assert result["at_68_95"]==pytest.approx([.68,.95])
    wrong = tarp(physical(unit_p),physical(refs),reference_seed=seed)
    assert wrong["status"]=="fail"


def test_forced_generator_masks_and_default_cache_replay():
    seeds = [50000,50001]
    default = generate_batch(seeds,split="validation")
    f,o,t,_ = validation_catalogs(len(seeds))
    assert np.array_equal(encode_observations(default.observations),f)
    assert np.array_equal(default.latents.theta,t)
    a = generate_batch(seeds,split="validation",forced_n=8,localized_fraction=1.)
    b = generate_batch(seeds,split="validation",forced_n=8,localized_fraction=0.)
    assert a.observations.is_localized.all() and not b.observations.is_localized.any()
    assert len(b.observations.z_obs)==0
    for key in ("dm_obs","fluence_jy_ms","ra_deg","dec_deg"):
        assert np.array_equal(getattr(a.observations,key),getattr(b.observations,key))
    assert np.array_equal(a.latents.theta,b.latents.theta)
    ignored = generate_batch(seeds,split="validation",forced_n=8,selection_on=False)
    assert np.isfinite(encode_observations(ignored.observations)).all()
    assert np.all(np.diff(ignored.observations.offsets)==8)


def test_actual_new_model_tiny_inference_and_smoke():
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(71010)
    model = PosteriorFlow(ModelConfig(statistics_bypass=True))
    model.statistics.set_normalization(json.loads((RUN_DIR/"normalization.json").read_text()))
    f = smoke_features()
    x,m,_ = pad_catalogs(f,np.array([0,len(f)]),np.zeros((1,4)),[0])
    p,_ = model.sample(x,m,16,seed=73500,steps=2,check_steps=False)
    result = smoke_summary(p.numpy())
    assert result["status"]=="reported" and np.isfinite(result["posterior_mean"]).all()
    assert contraction(p.numpy(),.7)["status"] in ("pass","fail")
    assert information(p.numpy(),p.numpy())["status"]=="fail"
    assert tarp(p.numpy(),p[:,0,:].numpy(),reference_seed=73300)["status"] in ("pass","fail")
