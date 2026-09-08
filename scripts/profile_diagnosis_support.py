"""Read training composition support and saved diagnostic conditioning values."""
import json
import numpy as np
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.statistics import catalog_statistics,STAT_NAMES
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from complete_phase2a_diagnosis import OUT,sha,load_npz

torch.set_num_threads(4)
manifest = json.loads((ROOT/"results/training-data-manifest.json").read_text())
sizes,localized = [],[]
for shard in manifest["shards"]:
    if shard["split"]!="training":
        continue
    path = ROOT/"work/training-data"/shard["path"]
    assert sha(path)==shard["sha256"]
    with np.load(path,allow_pickle=False) as d:
        sizes.append(np.diff(d["offsets"]))
        localized.append(np.add.reduceat((d["features"][:,6]<.5).astype(np.int64),d["offsets"][:-1]))
n,loc = np.concatenate(sizes),np.concatenate(localized)
report = {"scope":"training support only; no training updates or inference",
    "training_catalogs":len(n),"all_localized_count":int((n==loc).sum()),
    "all_localized_N_max":int(n[n==loc].max()),"all_unlocalized_count":int((loc==0).sum()),
    "N_ge64_count":int((n>=64).sum()),"all_localized_N_ge64":int(((loc==n)&(n>=64)).sum()),
    "all_unlocalized_N_ge64":int(((loc==0)&(n>=64)).sum()),
    "N_ge64_localized_fraction_quantiles":np.quantile((loc/n)[n>=64],[0,.01,.5,.99,1]).tolist(),
    "standardized_probe_features":{}}
norm = json.loads((RUN_DIR/"normalization.json").read_text())
for name,path in (("mixed",RUN_DIR/"posterior-artifacts/contraction-256.npz"),
                  ("all_localized",OUT/"all_localized.npz"),("all_unlocalized",OUT/"all_unlocalized.npz")):
    d = load_npz(path)
    x,mask,_ = pad_catalogs(d["features"],d["offsets"],d["truth"],np.arange(len(d["truth"])))
    z = (catalog_statistics(x,mask).numpy()-np.array(norm["center"]))/np.array(norm["scale"])
    med = np.median(z,axis=0)
    ids = np.argsort(abs(med))[-6:][::-1]
    report["standardized_probe_features"][name] = [
        {"statistic":STAT_NAMES[i],"median_standardized_value":float(med[i]),
         "min":float(z[:,i].min()),"max":float(z[:,i].max())} for i in ids]
write_json(OUT/"training-support.json",report)
print(json.dumps(report,indent=2))
