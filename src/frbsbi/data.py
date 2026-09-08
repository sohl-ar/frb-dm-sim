"""Regenerable on-disk training data, with observables and targets separated.

Sampler stays float64. Features are cast to float32 only at the explicit
training boundary. Host theta remains in physical median / natural-log width.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import torch
from .generator import generate_batch, NOISE_FLUENCE_FIXTURE
from .preflight import SEED_RANGES,assert_disjoint_seed_ranges

ROOT = Path(__file__).resolve().parents[2]


def encode_observations(obs):
    if np.any(obs.dm_obs<=0):
        raise ValueError("logDM encoding requires positive observed DM")
    sky = SkyCoord(ra=obs.ra_deg*u.deg,dec=obs.dec_deg*u.deg).galactic.cartesian
    z = np.zeros(len(obs.dm_obs),dtype=np.float64)
    z[obs.is_localized] = obs.z_obs
    return np.column_stack((np.log10(obs.dm_obs),sky.x.value,sky.y.value,sky.z.value,
                            np.log10(obs.fluence_jy_ms+NOISE_FLUENCE_FIXTURE),z,
                            ~obs.is_localized)).astype(np.float32)


def require_pretraining():
    ledger = json.loads((ROOT/"results/phase2a_gates.json").read_text(encoding="utf-8"))
    if ledger["pretraining"]["T1"]["status"]!="pass":
        raise RuntimeError("STOP: T1 RNG audit is not passed")
    for name,key in (("prior-predictive.json","status"),("generator-audit.json","status")):
        report = json.loads((ROOT/"results"/name).read_text(encoding="utf-8"))
        if report[key]!="pass":
            raise RuntimeError(f"STOP: pretraining requirement failed: {name}")
    selection = json.loads((ROOT/"results/selection-audit.json").read_text(encoding="utf-8"))
    if not selection["numeric_checks_pass"] or not selection["prints_within_factor_two"]:
        raise RuntimeError("STOP: selection pretraining unresolved")


def prepare_dataset(destination=None):
    require_pretraining()
    assert_disjoint_seed_ranges(SEED_RANGES)
    destination = ROOT/"work/training-data" if destination is None else Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    manifest = {"sampler_precision":"float64","features_precision":"float32",
                "seed_ranges":{k:asdict(v) for k,v in SEED_RANGES.items()},"shards":[],"complete":False}
    start = time.perf_counter()
    for split in ("training","validation","test"):
        interval = SEED_RANGES[split]
        for first in range(interval.start,interval.stop,512):
            seeds = list(range(first,min(first+512,interval.stop)))
            batch = generate_batch(seeds,split=split)
            features = encode_observations(batch.observations)
            name = f"{split}-{first:06d}.npz"
            target = destination/name
            np.savez(target,features=features,offsets=batch.observations.offsets,
                     theta=batch.latents.theta,family=batch.latents.family,seeds=np.array(seeds))
            manifest["shards"].append({"path":name,"split":split,"n_catalogs":len(seeds),"n_bursts":len(features),
                                       "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"metadata":batch.metadata})
            manifest["elapsed_seconds"] = time.perf_counter()-start
            (ROOT/"results/training-data-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            print(json.dumps({"split":split,"catalogs_done":seeds[-1]-interval.start+1,
                              "elapsed_seconds":manifest["elapsed_seconds"]}),flush=True)
    manifest["complete"] = True
    manifest["total_catalogs"] = sum(s["n_catalogs"] for s in manifest["shards"])
    manifest["total_bursts"] = sum(s["n_bursts"] for s in manifest["shards"])
    (ROOT/"results/training-data-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def pad_catalogs(features,offsets,theta,indices,*,permutations=None,extra_padding=0):
    """Shuffle real elements if requested, THEN pad. True padding is ignored."""
    sizes = offsets[np.array(indices)+1]-offsets[indices]
    output = np.zeros((len(indices),int(max(sizes))+extra_padding,7),dtype=np.float32)
    padding = np.ones(output.shape[:2],dtype=np.bool_)
    for row,i in enumerate(indices):
        real = features[offsets[i]:offsets[i+1]]
        if permutations is not None:
            real = real[permutations[row]]
        output[row,:len(real)] = real
        padding[row,:len(real)] = False
    return torch.from_numpy(output),torch.from_numpy(padding),torch.from_numpy(theta[indices])


if __name__ == "__main__":
    prepare_dataset()
