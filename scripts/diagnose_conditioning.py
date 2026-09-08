"""Frozen-checkpoint conditioning diagnosis on existing validation catalogs.

No optimizer, training, simulator draws, ODE sampling, or src/ modifications.
Correlations are descriptive, with raw and prior-coordinate targets retained.
Loss differences use common x0/t draws and catalog-cluster uncertainty.
"""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.evaluate import validation_catalogs
from frbsbi.inference import load_checkpoint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlations(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    ac=a-a.mean(0); bc=b-b.mean(0)
    denominator=np.sqrt((ac*ac).sum(0)[:,None]*(bc*bc).sum(0)[None,:])
    return np.divide(ac.T@bc,denominator,out=np.zeros_like(denominator),where=denominator>0)


def stats(a):
    a=np.asarray(a,dtype=np.float64)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'max':float(a.max())}


def main():
    out=ROOT/'results/conditioning-diagnosis.json'
    if out.exists():
        raise RuntimeError('Diagnosis already saved; inspect it instead of silently repeating')
    started=time.perf_counter()
    train=json.loads((ROOT/'results/training.json').read_text())
    path=ROOT/train['checkpoint']; original_hash=digest(path)
    assert original_hash==train['checkpoint_sha256']
    assert digest(ROOT/'results/training-data-manifest.json')==train['dataset_manifest_sha256']
    torch.set_num_threads(train['training_config']['cpu_threads'])
    torch.use_deterministic_algorithms(True); torch.backends.mha.set_fastpath_enabled(False)
    model,payload=load_checkpoint(path)
    before={k:v.clone() for k,v in model.state_dict().items()}
    features,offsets,truth,seeds=validation_catalogs(100)
    x,mask,theta=pad_catalogs(features,offsets,truth,np.arange(100))
    stages={}
    def hook(name,pool=False):
        def capture(module,inputs,output):
            value=output.detach()
            if value.ndim==3:
                value=(value*(~mask)[...,None]).sum(1)/(~mask).sum(1)[:,None]
            stages[name]=value.to(torch.float64).numpy()
        return capture
    handles=[model.encoder.project.register_forward_hook(hook('per_burst_projection_mean')),
             model.encoder.blocks[0].register_forward_hook(hook('ISAB_1_element_mean')),
             model.encoder.blocks[1].register_forward_hook(hook('ISAB_2_element_mean')),
             model.encoder.pool.register_forward_hook(hook('PMA_summary'))]
    with torch.no_grad():
        summary=model.encoder(x,mask)
    for h in handles: h.remove()
    c=summary.double().numpy(); labels=[f'pooled_{j:02d}' for j in range(c.shape[1]-2)]+['log1p_N','localized_fraction']
    physical_corr=correlations(c,truth)
    target_prior=model.transform(theta).double().sigmoid().numpy()
    prior_corr=correlations(c,target_prior)
    stage_stats={}
    for name,values in stages.items():
        stage_stats[name]={'shape':list(values.shape),'dimension_std':values.std(0,ddof=1).tolist(),
            'catalog_variation_RMS':float(np.sqrt(np.mean((values-values.mean(0))**2))),
            'value_RMS':float(np.sqrt(np.mean(values**2))),
            'max_abs_physical_correlation_per_parameter':abs(correlations(values,truth)).max(0).tolist()}
    rng=np.random.default_rng(75002)
    permutations=[]
    for _ in range(8):
        perm=rng.permutation(len(c))
        while np.any(perm==np.arange(len(c))): perm=rng.permutation(len(c))
        permutations.append(perm)
    mean=summary.mean(0,keepdim=True).expand_as(summary)
    alternatives={'zero':torch.zeros_like(summary),'mean':mean,'shuffled':summary[permutations[0]],
        'pooled_shuffled_only':torch.cat((summary[permutations[0],:-2],summary[:,-2:]),dim=1),
        'extras_shuffled_only':torch.cat((summary[:,:-2],summary[permutations[0],-2:]),dim=1)}
    fixed_rng=torch.Generator().manual_seed(75000)
    states=[torch.zeros(1,4),torch.randn(1,4,generator=fixed_rng)]
    sensitivity=[]
    with torch.no_grad():
        for state_index,state in enumerate(states):
            for t_value in (.1,.5,.9):
                xt=state.expand(len(c),-1); tt=torch.full((len(c),1),t_value)
                real=model.vector_field(xt,tt,summary)
                row={'state':state.tolist()[0],'state_index':state_index,'t':t_value,
                    'real_velocity_norm':stats(real.norm(dim=1).numpy()),
                    'across_catalog_velocity_std':real.double().std(0).tolist(),'alternatives':{}}
                for name,alternative in alternatives.items():
                    other=model.vector_field(xt,tt,alternative)
                    row['alternatives'][name]=stats((real-other).norm(dim=1).numpy())
                sensitivity.append(row)
    repetitions=256
    noise=torch.Generator().manual_seed(75001)
    with torch.no_grad():
        x1=model.transform(theta).repeat_interleave(repetitions,dim=0)
        x0=torch.randn(x1.shape,generator=noise)
        tt=torch.rand((len(x1),1),generator=noise)
        xt=(1-tt)*x0+tt*x1; target=x1-x0
        def loss(condition):
            v=model.vector_field(xt,tt,condition.repeat_interleave(repetitions,dim=0))
            return (v-target).square().reshape(len(c),repetitions,4).mean(1).double().numpy()
        real_loss=loss(summary)
        alternatives_loss={key:loss(value) for key,value in alternatives.items()}
        shuffled_loss=np.stack([loss(summary[perm]) for perm in permutations]).mean(0)
        alternatives_loss['shuffled_average_8']=shuffled_loss
    loss_report={'per_parameter_real':real_loss.mean(0).tolist(),'total_real':float(real_loss.mean()),'alternatives':{}}
    for name,value in alternatives_loss.items():
        delta=value-real_loss; total=delta.mean(1)
        loss_report['alternatives'][name]={'per_parameter_loss':value.mean(0).tolist(),'total_loss':float(value.mean()),
            'per_parameter_delta_alternative_minus_real':delta.mean(0).tolist(),
            'per_parameter_paired_catalog_SE':(delta.std(0,ddof=1)/np.sqrt(len(c))).tolist(),
            'delta_alternative_minus_real':float(total.mean()),'paired_catalog_SE':float(total.std(ddof=1)/np.sqrt(len(c))),
            'relative_loss_delta':float(total.mean()/real_loss.mean())}
    # Autograd is used only to measure the frozen model, never to update it.
    cond=summary.detach().clone().requires_grad_(True)
    real_v=model.vector_field(torch.zeros(len(c),4),torch.full((len(c),1),.5),cond)
    jac=[]
    for j in range(4):
        jac.append(torch.autograd.grad(real_v[:,j].sum(),cond,retain_graph=j<3)[0].detach().numpy())
    jac=np.stack(jac,axis=1)
    scaled_jac=jac*c.std(0,ddof=1)[None,None,:]
    jacobian={'mean_absolute_raw_by_output_and_summary':abs(jac).mean(0).tolist(),
        'one_summary_std_scaled_RMS_by_output_pooled':np.sqrt(np.mean(scaled_jac[:,:,:-2]**2,axis=(0,2))).tolist(),
        'one_summary_std_scaled_RMS_by_output_extras':np.sqrt(np.mean(scaled_jac[:,:,-2:]**2,axis=(0,2))).tolist(),
        'nonzero_summary_derivative_entries':int(np.count_nonzero(jac))}
    # Record decoder first-layer relative sensitivity to variation vs mean offsets.
    first=model.velocity[0]
    w=first.weight.detach().double().numpy()[:,5:]
    first_layer={'condition_weight_column_norms':np.linalg.norm(w,axis=0).tolist(),
        'pooled_varying_contribution_RMS':float(np.sqrt(np.mean(((c[:,:-2]-c[:,:-2].mean(0))@w[:,:-2].T)**2))),
        'extras_varying_contribution_RMS':float(np.sqrt(np.mean(((c[:,-2:]-c[:,-2:].mean(0))@w[:,-2:].T)**2))),
        'mean_condition_contribution_RMS':float(np.sqrt(np.mean((c.mean(0)@w.T)**2)))}
    high=[{'dimension':labels[i],'parameter':name,'r':float(physical_corr[i,j])}
        for i in range(len(labels)) for j,name in enumerate(['f_d','F','host_median','host_sigma_ln']) if abs(physical_corr[i,j])>.3]
    report={'git_sha':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'checkpoint_sha256':original_hash,'checkpoint_epoch':payload['epoch'],'catalog_seeds':seeds,
        'diagnostic_seeds':{'fixed_state':75000,'loss_noise':75001,'permutations':75002},
        'scope':'Frozen model on 100 existing validation catalogs; no new simulation/training/ODE samples',
        'architecture':{'raw_features':int(x.shape[-1]),'per_burst_projection_input':model.encoder.project.in_features,
            'per_burst_width':model.encoder.project.out_features,'pooled_width':model.config.width,
            'condition_width':int(c.shape[1]),'velocity_linear_layers':[[m.in_features,m.out_features] for m in model.velocity if isinstance(m,torch.nn.Linear)]},
        'summary_labels':labels,'parameter_labels':['f_d','F','host_median','host_sigma_ln'],
        'summary_mean':c.mean(0).tolist(),'summary_std':c.std(0,ddof=1).tolist(),
        'physical_correlation_matrix':physical_corr.tolist(),'prior_coordinate_correlation_matrix':prior_corr.tolist(),
        'flag_abs_r_gt_0_3':high,'constant_summary_dimensions':[labels[i] for i,v in enumerate(c.std(0)) if v==0],
        'stages':stage_stats,'velocity_sensitivity':sensitivity,'loss':loss_report,
        'loss_noise_draws_per_catalog':repetitions,'shuffle_count':len(permutations),
        'loss_uncertainty':'Paired SE across catalog-average differences; permutation dependence and small catalog count limit interpretation.',
        'jacobian':jacobian,'first_velocity_layer':first_layer,
        'limitations':['Linear correlations are not a nonlinear information test; abs(r)>0.3 is descriptive across many comparisons.',
          'Zero condition is off-distribution; shuffled conditions are the primary catalog-information comparison.',
          'No benefit in this frozen model does not prove no incentive in the CFM objective or no information in the data.'],
        'elapsed_seconds':time.perf_counter()-started}
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert digest(path)==original_hash
    report['diagnostic_config']={'catalog_count':len(seeds),'loss_draws_per_catalog':repetitions,
        'shuffle_count':len(permutations),'times':[.1,.5,.9],'seeds':report['diagnostic_seeds']}
    report['config_hash']=hashlib.sha256(json.dumps(report['diagnostic_config'],sort_keys=True).encode()).hexdigest()
    report['model_state_unchanged']=True
    artifact=ROOT/'results/conditioning-diagnostic-arrays.npz'
    np.savez_compressed(artifact,summary=c,theta=truth,real_loss=real_loss,shuffled_loss=shuffled_loss,
        catalog_seeds=np.array(seeds),permutations=np.array(permutations),**stages)
    report['array_artifact_sha256']=digest(artifact)
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    print(json.dumps({'architecture':report['architecture'],'stages':{k:{kk:vv for kk,vv in v.items() if kk!='dimension_std'} for k,v in stage_stats.items()},
        'max_physical_abs_corr':abs(physical_corr).max(0).tolist(),'flag_count':len(high),
        'sensitivity':sensitivity,'loss':loss_report,'first_layer':{k:v for k,v in first_layer.items() if k!='condition_weight_column_norms'}},indent=2))


if __name__=='__main__': main()
