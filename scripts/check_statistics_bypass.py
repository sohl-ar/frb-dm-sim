"""Allowed quick verification: 100 validation catalogs, one backward, no update."""
import json
import numpy as np
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.evaluate import validation_catalogs
from frbsbi.model import PosteriorFlow,ModelConfig
from frbsbi.statistics import STAT_NAMES,catalog_statistics
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from frbsim.catalog import git_provenance

torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
torch.use_deterministic_algorithms(True)
torch.manual_seed(71010)
f,o,t,seeds = validation_catalogs(100)
x,mask,y = pad_catalogs(f,o,t,np.arange(100))
model = PosteriorFlow(ModelConfig(statistics_bypass=True))
model.statistics.set_normalization(json.loads((RUN_DIR/"normalization.json").read_text()))
raw = catalog_statistics(x,mask).numpy()
scaled = model.statistics(x,mask).detach().numpy()
corr = np.zeros((len(STAT_NAMES),4))
for i in range(len(STAT_NAMES)):
    if raw[:,i].std()>0:
        corr[i] = np.corrcoef(raw[:,i],t.T)[0,1:]
loss = model.loss(x[:8],mask[:8],y[:8],generator=torch.Generator().manual_seed(71011))
loss.backward()
grad = model.velocity[0].weight.grad[:,-len(STAT_NAMES):]
assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm()>0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
report = {**git_provenance(),"scope":"quick engineering check; no optimizer step or training",
          "status":"pass","catalog_seeds":seeds,"statistics":STAT_NAMES,
          "raw_variance":raw.var(0).tolist(),"standardized_variance":scaled.var(0).tolist(),
          "correlations":corr.tolist(),"parameter_order":["f_d","F","host_median","host_sigma_ln"],
          "flagged_abs_r_above_0_3":[{"statistic":STAT_NAMES[i],"parameter_index":int(j),"r":float(corr[i,j])}
                                     for i,j in zip(*np.where(abs(corr)>.3))],
          "constant_validation_dimensions":[STAT_NAMES[i] for i in np.flatnonzero(raw.std(0)==0)],
          "summary_dimension":int(model.condition(x[:1],mask[:1]).shape[1]),
          "velocity_input_dimension":model.velocity[0].in_features,
          "forward_loss":float(loss.detach()),"bypass_first_layer_gradient_norm":float(grad.norm()),
          "normalization_fit":"all training catalogs only; no validation fitting"}
write_json(RUN_DIR/"statistics-quickcheck.json",report)
print(json.dumps({k:report[k] for k in ("status","summary_dimension","velocity_input_dimension","forward_loss","bypass_first_layer_gradient_norm","constant_validation_dimensions")}))
