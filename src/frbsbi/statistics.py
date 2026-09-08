"""Observable bypass; host width uses NATURAL LOG conventions.

Recover DM/fluence from the existing float32 observable feature boundary,
then reduce in float64. No latent or missing redshift is read. Standard
deviations use ddof=0. Empty subsets are zero with explicit availability.
OLS uses observed localized DM-z, with no cosmological/host subtraction.
"""
import hashlib
import json
import numpy as np
import torch
from torch import nn
from .data import ROOT, pad_catalogs

STAT_NAMES = (
    "dm_mean", "dm_std", "dm_median", "dm_q25", "dm_q75", "dm_min", "dm_max",
    "fluence_mean", "fluence_std", "fluence_median",
    "localized_dm_mean", "localized_dm_std", "localized_z_mean", "localized_count",
    "unlocalized_count", "has_localized", "has_unlocalized", "ols_available",
    "localized_z_std", "dm_z_slope", "dm_z_intercept", "residual_std",
    "residual_q25", "residual_median", "residual_q75",
) + tuple(f"zbin{i}_{s}" for i in range(3) for s in ("count", "dm_mean", "dm_std"))
Z_EDGES = (.05, .5, 1., 1.5)  # Observable diagnostic bins, not physics parameters.


def catalog_statistics(features, padding):
    if features.ndim != 3 or features.shape[-1] != 7 or padding.shape != features.shape[:2]:
        raise ValueError("expected features [batch,N,7] and padding [batch,N]")
    rows = []
    for f, mask in zip(features, padding):
        f = f[~mask].to(torch.float64)
        if not len(f) or not bool(torch.isfinite(f).all()):
            raise ValueError("statistics require nonempty finite observable catalogs")
        dm, fluence = 10.**f[:,0], 10.**f[:,4]-1.
        local = f[:,6] < .5
        ld, z = dm[local], f[local,5]
        zero = f.new_zeros(())
        q = lambda a: torch.quantile(a, a.new_tensor([.25,.5,.75]))
        quart = q(dm)
        values = [dm.mean(),dm.std(correction=0),quart[1],quart[0],quart[2],dm.min(),dm.max(),
                  fluence.mean(),fluence.std(correction=0),q(fluence)[1]]
        ols = len(z)>=2 and bool((z-z.mean()).square().sum()>0)
        values += [ld.mean() if len(ld) else zero,ld.std(correction=0) if len(ld) else zero,
                   z.mean() if len(z) else zero,zero+len(z),zero+len(f)-len(z),
                   zero+bool(len(z)),zero+bool(len(f)-len(z)),zero+ols,
                   z.std(correction=0) if len(z) else zero]
        if ols:
            dz = z-z.mean()
            slope = (dz*(ld-ld.mean())).sum()/dz.square().sum()
            intercept = ld.mean()-slope*z.mean()
            residual = ld-(intercept+slope*z)
            values += [slope,intercept,residual.std(correction=0),*q(residual)]
        else:
            values += [zero]*6
        for i,(lo,hi) in enumerate(zip(Z_EDGES[:-1],Z_EDGES[1:])):
            subset = ld[(z>=lo)&((z<=hi) if i==2 else (z<hi))]
            values += [zero+len(subset),subset.mean() if len(subset) else zero,
                       subset.std(correction=0) if len(subset) else zero]
        rows.append(torch.stack(values))
    result = torch.stack(rows)
    if result.shape[1]!=len(STAT_NAMES) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid observable statistics")
    return result


class StatisticsBypass(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center",torch.zeros(len(STAT_NAMES),dtype=torch.float64))
        self.register_buffer("scale",torch.ones(len(STAT_NAMES),dtype=torch.float64))
        self.register_buffer("fitted",torch.tensor(False))

    def set_normalization(self, report):
        if tuple(report["names"])!=STAT_NAMES or report["fit_split"]!="training":
            raise ValueError("statistics schema / training-only normalization mismatch")
        center = torch.tensor(report["center"],dtype=torch.float64)
        scale = torch.tensor(report["scale"],dtype=torch.float64)
        if not bool(torch.isfinite(center).all() & torch.isfinite(scale).all() & (scale>0).all()):
            raise ValueError("invalid normalization")
        self.center.copy_(center); self.scale.copy_(scale); self.fitted.fill_(True)

    def forward(self, features, padding):
        if not bool(self.fitted):
            raise RuntimeError("STOP: training-only normalization required")
        return ((catalog_statistics(features,padding)-self.center)/self.scale).to(features.dtype)


def canonical_manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def fit_training_statistics(data_root=None):
    """One fixed pass over training shards only; no model optimization."""
    data_root = ROOT/"work/training-data" if data_root is None else data_root
    manifest = json.loads((ROOT/"results/training-data-manifest.json").read_text())
    if not manifest["complete"]:
        raise RuntimeError("incomplete dataset")
    chunks,hashes = [],[]
    for shard in manifest["shards"]:
        if shard["split"]!="training":
            continue
        path = data_root/shard["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=shard["sha256"]:
            raise RuntimeError("STOP: training shard hash mismatch")
        with np.load(path,allow_pickle=False) as d:
            for first in range(0,len(d["theta"]),64):
                x,mask,_ = pad_catalogs(d["features"],d["offsets"],d["theta"],
                                      np.arange(first,min(first+64,len(d["theta"]))))
                chunks.append(catalog_statistics(x,mask).numpy())
        hashes.append({"path":shard["path"],"sha256":shard["sha256"]})
    values = np.concatenate(chunks)
    center,scale = values.mean(0),values.std(0)
    constant = scale==0
    scale[constant] = 1.  # Exact constants become zero after centering.
    return {"schema_version":1,"names":STAT_NAMES,"fit_split":"training",
            "n_catalogs":len(values),"center":center.tolist(),"scale":scale.tolist(),
            "constant_dimensions":np.flatnonzero(constant).tolist(),"shards":hashes,
            "dataset_manifest_sha256":canonical_manifest_hash(manifest),
            "manifest_hash_algorithm":"sha256 of sorted compact JSON UTF-8",
            "precision":"float64 statistics recovered from existing float32 observable features"}
