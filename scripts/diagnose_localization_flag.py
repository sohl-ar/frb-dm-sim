"""Counterfactual summary ablation, not a valid posterior or accepted remedy.

All-localized observables stay fixed. Only the standardized has_unlocalized
entry is replaced with its value in the paired mixed catalogs, to attribute
the model response. This deliberately inconsistent summary is diagnostic only.
"""
import json
import numpy as np
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.inference import load_checkpoint
from frbsbi.statistics import STAT_NAMES
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from complete_phase2a_diagnosis import OUT,load_npz,sha,summarize

training = json.loads((RUN_DIR/"training.json").read_text())
checkpoint = ROOT/training["checkpoint"]
assert sha(checkpoint)==training["checkpoint_sha256"]
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
torch.use_deterministic_algorithms(True)
model,_ = load_checkpoint(checkpoint)
d = load_npz(OUT/"all_localized.npz")
index = STAT_NAMES.index("has_unlocalized")
replacement = float((1-model.statistics.center[index])/model.statistics.scale[index])
def replace_flag(module,args,output):
    result = output.clone()
    result[:,index] = replacement
    return result
hook = model.statistics.register_forward_hook(replace_flag)
sampling = json.loads((OUT/"diagnosis.json").read_text())["host_probe"]["artifacts"]["all_localized"]
posterior = np.empty_like(d["posterior"])
try:
    for first in range(0,len(d["truth"]),sampling["batch_size"]):
        ids = np.arange(first,min(first+sampling["batch_size"],len(d["truth"])))
        x,mask,_ = pad_catalogs(d["features"],d["offsets"],d["truth"],ids)
        p,_ = model.sample(x,mask,sampling["draws"],seed=sampling["base_sampling_seed"]+first,
                          steps=sampling["rk4_steps"],check_steps=False)
        posterior[ids] = p.numpy()
finally:
    hook.remove()
report = {"scope":"counterfactual summary ablation only; not a valid posterior or gate evidence",
    "changed_entry":STAT_NAMES[index],"replacement_standardized_value":replacement,
    "changed_parameters":False,"checkpoint_unchanged":sha(checkpoint)==training["checkpoint_sha256"],
    "sampling":sampling,"result":summarize(posterior,d["truth"]),
    "limitation":"Flag contradicts all-localized composition; isolates a model input sensitivity, not a calibrated fix"}
target = OUT/"counterfactual-flag.npz"
np.savez_compressed(target,posterior=posterior,truth=d["truth"])
report["artifact"] = {"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target)}
write_json(OUT/"flag-ablation.json",report)
print(json.dumps({"replacement":replacement,"ratios":report["result"]["contraction_diagnostic"]["median_posterior_prior_ratio"],
    "coverage":{r["parameter"]:[r['0.68']['coverage'],r['0.95']['coverage']] for r in report["result"]["coverage"]}},indent=2))
