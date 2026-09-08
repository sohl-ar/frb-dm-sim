"""Locate frozen PMA attenuation, including a float64 evaluation cross-check."""
import copy,json,hashlib
import numpy as np
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.evaluate import validation_catalogs
from frbsbi.inference import load_checkpoint


def main():
    train=json.loads((ROOT/'results/training.json').read_text())
    model,payload=load_checkpoint(ROOT/train['checkpoint'])
    torch.set_num_threads(train['training_config']['cpu_threads'])
    torch.use_deterministic_algorithms(True); torch.backends.mha.set_fastpath_enabled(False)
    f,o,truth,seeds=validation_catalogs(100)
    x,mask,_=pad_catalogs(f,o,truth,np.arange(100))
    report={'catalog_seeds':seeds,'checkpoint_sha256':hashlib.sha256((ROOT/train['checkpoint']).read_bytes()).hexdigest(),
        'scope':'Same 100 existing catalogs, frozen PMA internal activations; double precision is diagnostic only',
        'precisions':{}}
    for dtype,label in ((torch.float32,'float32'),(torch.float64,'float64')):
        candidate=copy.deepcopy(model).to(dtype=dtype)
        pool=candidate.encoder.pool.pool
        captured={}
        def hook(name):
            def capture(module,inputs,output):
                if isinstance(output,tuple): output=output[0]
                value=output.detach().double().squeeze(1).numpy()
                captured[name]={'catalog_variation_RMS':float(np.sqrt(np.mean((value-value.mean(0))**2))),
                    'value_RMS':float(np.sqrt(np.mean(value**2)))}
            return capture
        handles=[module.register_forward_hook(hook(name)) for name,module in
            [('attention_output',pool.attention),('after_norm1',pool.norm1),('feedforward_output',pool.ff),('after_norm2',pool.norm2)]]
        with torch.no_grad():
            summary=candidate.encoder(x.to(dtype),mask)
            h=candidate.encoder.project(torch.cat((x.to(dtype)[...,:5],x.to(dtype)[...,6:7],
                torch.where((x[...,6]>.5)[...,None],candidate.encoder.missing_z,candidate.encoder.z_project(x.to(dtype)[...,5:6])),
                candidate.encoder.type_embed((x[...,6]>.5).long())),dim=-1))
            for block in candidate.encoder.blocks: h=block(h,mask)
            # Direct value projection plus output projection separates its
            # attenuation from softmax weighting and LayerNorm effects.
            width=candidate.config.width
            wv=pool.attention.in_proj_weight[2*width:]
            bv=pool.attention.in_proj_bias[2*width:]
            value=torch.nn.functional.linear(h,wv,bv)
            average=(value*(~mask)[...,None]).sum(1)/(~mask).sum(1)[:,None]
            uniform=torch.nn.functional.linear(average,pool.attention.out_proj.weight,pool.attention.out_proj.bias)
            centered=uniform-uniform.mean(0)
            captured['uniform_value_projection']={'catalog_variation_RMS':float(centered.square().mean().sqrt())}
        for handle in handles: handle.remove()
        report['precisions'][label]={'activations':captured,'pooled_std_median':float(summary[:,:-2].double().std(0).median()),
            'norm1_gain_min_max':pool.norm1.weight.detach().double().numpy()[[np.argmin(pool.norm1.weight.detach().numpy()),np.argmax(pool.norm1.weight.detach().numpy())]].tolist(),
            'norm2_gain_min_max':[float(pool.norm2.weight.detach().min()),float(pool.norm2.weight.detach().max())],
            'value_projection_Frobenius':float(wv.detach().norm()),'output_projection_Frobenius':float(pool.attention.out_proj.weight.detach().norm())}
    (ROOT/'results/pma-collapse-detail.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    print(json.dumps(report['precisions'],indent=2))


if __name__=='__main__': main()
