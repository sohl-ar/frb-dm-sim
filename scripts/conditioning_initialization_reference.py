"""Forward-only initialization reference; no training or new catalogs."""
import json
import numpy as np
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.evaluate import validation_catalogs
from frbsbi.model import PosteriorFlow,ModelConfig


def main():
    train=json.loads((ROOT/'results/training.json').read_text())
    torch.set_num_threads(train['training_config']['cpu_threads'])
    torch.use_deterministic_algorithms(True); torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(train['training_config']['initialization_seed'])
    initial=PosteriorFlow(ModelConfig(**train['model_config'])).eval()
    f,o,truth,seeds=validation_catalogs(100)
    x,mask,_=pad_catalogs(f,o,truth,np.arange(100))
    stages={}
    def capture(name):
        def hook(module,args,out):
            if out.ndim==3: out=(out*(~mask)[...,None]).sum(1)/(~mask).sum(1)[:,None]
            a=out.detach().double().numpy()
            stages[name]={'catalog_variation_RMS':float(np.sqrt(np.mean((a-a.mean(0))**2))),
                          'value_RMS':float(np.sqrt(np.mean(a*a)))}
        return hook
    handles=[m.register_forward_hook(capture(n)) for n,m in
        [('per_burst_projection_mean',initial.encoder.project),('ISAB_1_element_mean',initial.encoder.blocks[0]),
         ('ISAB_2_element_mean',initial.encoder.blocks[1]),('PMA_summary',initial.encoder.pool)]]
    with torch.no_grad(): initial.encoder(x,mask)
    for h in handles: h.remove()
    trained=json.loads((ROOT/'results/conditioning-diagnosis.json').read_text())
    comparison={k:{'initialized_variation_RMS':v['catalog_variation_RMS'],
        'trained_variation_RMS':trained['stages'][k]['catalog_variation_RMS'],
        'trained_over_initialized':trained['stages'][k]['catalog_variation_RMS']/v['catalog_variation_RMS']}
        for k,v in stages.items()}
    report={'scope':'Forward-only reconstruction of original seeded initialization; no updates or fitting',
        'initialization_seed':train['training_config']['initialization_seed'],'catalog_seeds':seeds,
        'model_config':train['model_config'],'stages':comparison,
        'limitation':'Endpoint comparison establishes learned attenuation, not when or why training entered it; intermediate encoder checkpoints are unavailable.'}
    (ROOT/'results/conditioning-initialization-reference.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__': main()
