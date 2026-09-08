"""Model engineering checks before full training; no posterior calibration claim."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from frbsim.catalog import git_provenance
from frbsbi.generator import generate_batch,PRIOR_LO,PRIOR_HI
from frbsbi.data import encode_observations,pad_catalogs,require_pretraining
from frbsbi.model import PosteriorFlow,ModelConfig
from frbsbi.tolerances import PERMUTATION_W1,TRANSFORM_ROUNDTRIP_REL

ROOT = Path(__file__).resolve().parents[1]


def state_hash(model):
    h = hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_audit():
    require_pretraining()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    batch = generate_batch(list(range(62000,62008)),split="audit")
    features = encode_observations(batch.observations)
    offsets,theta = batch.observations.offsets,batch.latents.theta
    x,mask,y = pad_catalogs(features,offsets,theta,list(range(8)))
    def train_short(seed):
        torch.manual_seed(seed)
        model = PosteriorFlow()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        rng = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x,mask,y,generator=rng)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        return model,losses
    model,losses = train_short(71000)
    replay,replay_losses = train_short(71000)
    weights_equal = state_hash(model)==state_hash(replay)
    transformed = model.transform(y)
    roundtrip = float((model.transform.inverse(transformed)-y).abs().div(torch.from_numpy(PRIOR_HI-PRIOR_LO)).max())
    model.eval()
    with torch.no_grad():
        base_summary = model.encoder(x,mask)
        rng = np.random.default_rng(71001)
        permutations = [rng.permutation(offsets[i+1]-offsets[i]) for i in range(8)]
        shuffled,shuffled_mask,_ = pad_catalogs(features,offsets,theta,list(range(8)),permutations=permutations,extra_padding=17)
        shuffled[shuffled_mask] = 7.  # finite arbitrary padding content must have no influence
        summary_error = float((model.encoder(shuffled,shuffled_mask)-base_summary).abs().max())
        p1,ode = model.sample(x,mask,32,seed=71002,steps=32,check_steps=True)
        p2,_ = model.sample(shuffled,shuffled_mask,32,seed=71002,steps=32,check_steps=False)
        p3,_ = model.sample(x,mask,32,seed=71002,steps=32,check_steps=False)
        p4,_ = model.sample(x,mask,32,seed=71003,steps=32,check_steps=False)
        w1 = float(((p1.sort(dim=1).values-p2.sort(dim=1).values).abs().mean(dim=1)/torch.from_numpy(PRIOR_HI-PRIOR_LO)).max())
    grad_missing = model.encoder.missing_z.grad
    finite = bool(all(np.isfinite(losses)) and torch.all(torch.isfinite(p1)))
    report = {**git_provenance(),"scope":"architecture and five-step replay only; not G-P1 full sample or calibrated inference",
              "config":asdict(ModelConfig()),"torch":torch.__version__,"threads":torch.get_num_threads(),
              "dtype":"float32 network; float64 physical transform boundaries and sampler",
              "seeds":{"weights":71000,"permutation":71001,"posterior":71002,"different_posterior":71003},
              "losses":losses,"replay_losses":replay_losses,"weights_bit_identical":weights_equal,
              "weights_hash":state_hash(model),"posterior_bit_identical":bool(torch.equal(p1,p3)),
              "different_seed_differs":not bool(torch.equal(p1,p4)),"parameter_roundtrip_prior_width_error":roundtrip,
              "padding_and_shuffle_summary_error":summary_error,"shuffle_wasserstein_prior_widths":w1,
              "missing_embedding_receives_gradient":grad_missing is not None and bool(torch.any(grad_missing!=0)),
              "ode":ode,"finite":finite}
    passed = (weights_equal and report["posterior_bit_identical"] and report["different_seed_differs"]
              and finite and roundtrip<TRANSFORM_ROUNDTRIP_REL and w1<PERMUTATION_W1 and report["missing_embedding_receives_gradient"])
    report["tolerances"] = {"parameter_roundtrip_prior_width_error":TRANSFORM_ROUNDTRIP_REL,
                             "shuffle_wasserstein_prior_widths":PERMUTATION_W1}
    report["status"] = "pass" if passed else "fail"
    (ROOT/"results/model-audit.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report


if __name__ == "__main__":
    report = run_audit()
    raise SystemExit(0 if report["status"]=="pass" else 1)
