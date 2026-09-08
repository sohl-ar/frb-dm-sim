"""Read-only model audit: ONE authorized 50-catalog replay, no new simulations.

Does not call training, run_gates, or modify original evidence. Conditional
coverage of the original 1000 trials is bounded from their archived ranks.
"""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.stats import kstest
import torch
from frbsbi.data import ROOT,pad_catalogs
from frbsbi.evaluate import validation_catalogs
from frbsbi.inference import load_checkpoint
from frbsbi.generator import PRIOR_LO,PRIOR_HI


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def save(name,value):
    (ROOT/'results'/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8',newline='\n')


def wilson(count,n):
    # Descriptive 95% binomial uncertainty, not a new acceptance tolerance.
    z=1.959963984540054
    p=count/n; denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [float(center-radius),float(center+radius)]


def rank_coverage_bounds(ranks,level,samples):
    """np.quantile default linear: ranks lose the two boundary interpolations.

    For q*(L-1) between order-statistic indices j and j+1, rank j+1 is
    ambiguous. All other ranks decide coverage exactly (continuous draws).
    """
    q=(1-level)/2
    left=q*(samples-1); right=(1-q)*(samples-1)
    definitely=(ranks>=int(np.floor(left))+2)&(ranks<=int(np.floor(right)))
    possibly=(ranks>=int(np.floor(left))+1)&(ranks<=int(np.floor(right))+1)
    return definitely,possibly


def main():
    target=ROOT/'results/sbc-spotcheck-50.json'
    if target.exists():
        raise RuntimeError('Spot-check already exists; do not repeat the authorized inference')
    training=json.loads((ROOT/'results/training.json').read_text())
    original=json.loads((ROOT/'results/posterior-gates.json').read_text())
    ledger=json.loads((ROOT/'results/phase2a_gates.json').read_text())
    gp4=original['gates']['G-P4']; raw=np.array(gp4['raw_ranks'])
    provenance={'audit_git_sha':git('rev-parse','HEAD'),'checkpoint_sha256':sha(ROOT/training['checkpoint']),
        'original_sbc_sha256':sha(ROOT/'results/posterior-gates.json'),
        'training_json_sha256':sha(ROOT/'results/training.json'),
        'manifest_sha256':sha(ROOT/'results/training-data-manifest.json')}
    assert provenance['checkpoint_sha256']==training['checkpoint_sha256']==original['checkpoint_sha256']
    assert provenance['manifest_sha256']==training['dataset_manifest_sha256']==original['dataset_manifest_sha256']
    model,payload=load_checkpoint(ROOT/training['checkpoint'])
    last=torch.load(ROOT/'results/checkpoints/phase2a-last.pt',map_location='cpu',weights_only=True)
    features,offsets,truth,seeds=validation_catalogs(gp4['trials'])
    assert seeds==gp4['catalog_seeds']==list(range(50000,51000))
    lengths=np.diff(offsets)
    localized=np.add.reduceat(1-features[:,6].astype(np.int64),offsets[:-1])/lengths
    jitter=(raw+np.random.default_rng(gp4['rank_jitter_seed']).random(raw.shape))/(gp4['posterior_samples_per_trial']+1)
    ks=[]
    for j,p in enumerate(gp4['parameters']):
        check=kstest(jitter[:,j],'uniform')
        ks.append({'parameter':p['parameter'],'D':float(check.statistic),'raw_p':float(check.pvalue),
                   'matches_archived':bool(check.statistic==p['KS_D'] and check.pvalue==p['raw_p'])})
    # Compare model/loss/validation/early-stop ASTs, ignoring only formatting.
    train_tree=ast.parse((ROOT/'src/frbsbi/train.py').read_text())
    resume_tree=ast.parse((ROOT/'work/resume_train.py').read_text())
    train_epoch=next(n for n in ast.walk(train_tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    resume_epoch=next(n for n in ast.walk(resume_tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    train_val=next(n for n in ast.walk(train_epoch) if isinstance(n,ast.With) and any('torch.no_grad' in ast.unparse(i.context_expr) for i in n.items))
    resume_val=next(n for n in ast.walk(resume_epoch) if isinstance(n,ast.With) and any('torch.no_grad' in ast.unparse(i.context_expr) for i in n.items))
    train_batches=next(n for n in ast.walk(train_epoch) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='first')
    resume_batches=next(n for n in ast.walk(resume_epoch) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='first')
    # First ten statements cover IDs through optimizer update/count; omit progress logging.
    common=min(len(train_batches.body),len(resume_batches.body))-1
    training_math_same=all(ast.dump(a)==ast.dump(b) for a,b in zip(train_batches.body[:common],resume_batches.body[:common]))
    epochs=training['epochs']; best=min(epochs,key=lambda e:e['validation_loss'])
    required_stop=best['epoch']+training['training_config']['patience']
    searched=[p for folder in ('src','tests') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.txt')]
    simps=[str(p.relative_to(ROOT)) for p in searched if 'simps(' in p.read_text(encoding='utf-8') or 'import simps' in p.read_text(encoding='utf-8')]
    packages={name:metadata.version(name) for name in ('numpy','scipy','torch','torchdiffeq','astropy','pytest')}
    lock=(ROOT/'requirements-training-lock.txt').read_text()
    audit={**provenance,'status':'findings_require_human_review','training_epochs':epochs,
        'best_epoch':payload['epoch'],'best_validation_loss':payload['validation_loss'],'last_epoch':last['epoch'],
        'checkpoint_hash_check':'pass','config_hash_check':payload['config_hash']==training['config_hash']==original['training_config_hash'],
        'manifest_hash_check':'pass','seed_check':'pass','SBC_KS_recomputation':ks,
        'ledger_matches_posterior':all(all(ledger['gates'][k].get(kk)==vv for kk,vv in v.items()) for k,v in original['gates'].items()),
        'resume':{'verdict':'FAIL: not fully faithful stopping behavior; optimization and validation math match',
            'sha256':sha(ROOT/'work/resume_train.py'),'training_batch_math_ast_matches':training_math_same,
            'validation_loop_ast_matches':ast.dump(train_val)==ast.dump(resume_val),
            'loads_last_and_optimizer_and_both_RNG_states':True,'validation_noise_seed':training['training_config']['validation_noise_seed'],
            'training_noise_seed':training['training_config']['training_noise_seed'],'recorded':training['resume'],
            'expected_patience_stop_epoch':required_stop,'actual_stop_epoch':epochs[-1]['epoch'],
            'failure':'Resume starts another epoch before checking already-exhausted patience. Best epoch 13 remains the evaluated checkpoint.',
            'provenance_issue':'from_epoch=18 contradicts docstring/note claiming epoch 18 replay. Progress counts describe last periodic update, not a discarded partial epoch. Resume source/executor SHA is not recorded in checkpoint git_sha.'},
        'git_log':git('log','--oneline','-15').splitlines(),'initial_status':git('status','--short').splitlines(),
        'changes_since_913c9fa_in_src_tests':git('diff','913c9fa..HEAD','--','src','tests'),
        'unpublished_commits':git('log','origin/main..HEAD','--oneline'),
        'remote_diff':git('diff','origin/main..HEAD','--stat'),
        'notebook':{'status':'FLAG: missing','path':'notebooks/frb_dm_sim_colab.ipynb'},
        'README_assessment':'FIX: stale Phase 2a boundary; no Colab badge exists',
        'COLAB_RUN_assessment':'FIX: combined native install should not be the default; current state needs updating',
        'simps_matches_src_tests':simps,'packages':packages,
        'packages_match_lock':all(f'{k}=={v}' in lock for k,v in packages.items()),
        'archived_posterior_samples_or_intervals':False,
        'original_evidence_preserved':True}
    save('repository-audit.json',audit)
    print(json.dumps({'part1_read_checks':audit['status'],'resume':audit['resume'],'KS':ks}),flush=True)
    torch.set_num_threads(training['training_config']['cpu_threads'])
    torch.use_deterministic_algorithms(True); torch.backends.mha.set_fastpath_enabled(False)
    posterior=np.empty((50,gp4['posterior_samples_per_trial'],4),dtype=np.float64)
    start=time.perf_counter(); ode=[]
    for first in range(0,50,16):
        ids=np.arange(first,min(first+16,50))
        x,mask,_=pad_catalogs(features,offsets,truth,ids)
        values,info=model.sample(x,mask,gp4['posterior_samples_per_trial'],seed=72100+first,steps=32,check_steps=False)
        posterior[ids]=values.numpy(); ode.append(info)
    replay_ranks=(posterior<truth[:50,None,:]).sum(axis=1)
    rows=[]
    widths={}
    prior_quantiles=[]
    for q in (.16,.84):
        values=PRIOR_LO+q*(PRIOR_HI-PRIOR_LO)
        values[[1,2]]=np.exp(np.log(PRIOR_LO[[1,2]])+q*np.log(PRIOR_HI[[1,2]]/PRIOR_LO[[1,2]]))
        prior_quantiles.append(values)
    prior68=prior_quantiles[1]-prior_quantiles[0]
    q16,q84=np.quantile(posterior,[.16,.84],axis=1)
    for j,p in enumerate(gp4['parameters']):
        row={'parameter':p['parameter']}
        for level in (.68,.95):
            low,high=np.quantile(posterior,[(1-level)/2,(1+level)/2],axis=1)
            count=int(((truth[:50,j]>=low[:,j])&(truth[:50,j]<=high[:,j])).sum())
            ref=p['coverage'+str(round(100*level))]
            row[str(level)]={'covered':count,'n':50,'coverage':count/50,'original_1000':ref,
                'difference':count/50-ref,'wilson95':wilson(count,50),
                'subset_difference_standard_error':float(np.sqrt(ref*(1-ref)*(1/50-1/1000)))}
        widths[p['parameter']]={'median_physical_central68_width':float(np.median(q84[:,j]-q16[:,j])),
            'ratio_to_prior_support':float(np.median(q84[:,j]-q16[:,j])/(PRIOR_HI[j]-PRIOR_LO[j])),
            'prior_central68_width':float(prior68[j]),
            'ratio_to_matched_prior_central68_width':float(np.median(q84[:,j]-q16[:,j])/prior68[j])}
        rows.append(row)
    artifact=ROOT/'results/audit-spotcheck-50.npz'
    np.savez_compressed(artifact,posterior=posterior,truth=truth[:50],seeds=np.array(seeds[:50]),localized_fraction=localized[:50])
    spot={**provenance,'scope':'Only authorized 50-trial replay on existing validation catalogs; no new simulations',
        'trials':50,'samples_per_trial':gp4['posterior_samples_per_trial'],'catalog_seeds':seeds[:50],
        'solver_calls':ode,'elapsed_seconds':time.perf_counter()-start,'parameters':rows,
        'archived_rank_matches':int((raw[:50]==replay_ranks).sum()),'archived_rank_comparisons':int(replay_ranks.size),
        'posterior_artifact':str(artifact.relative_to(ROOT)),'posterior_artifact_sha256':sha(artifact),
        'posterior_widths':widths,'not_a_replacement_for_full_SBC':True}
    save('sbc-spotcheck-50.json',spot)
    print(json.dumps({'part1_spotcheck':rows,'rank_replay':[spot['archived_rank_matches'],spot['archived_rank_comparisons']]}),flush=True)
    def summarize(mask):
        n=int(mask.sum()); parameters={}
        for j,p in enumerate(gp4['parameters']):
            levels={}
            for level in (.68,.95):
                lower,upper=rank_coverage_bounds(raw[mask,j],level,gp4['posterior_samples_per_trial'])
                lo,hi=int(lower.sum()),int(upper.sum())
                levels[str(level)]={'covered_count_bounds':[lo,hi],'coverage_bounds':[lo/n,hi/n],
                    'ambiguous_boundary_trials':hi-lo,'wilson95_outer_bounds':[wilson(lo,n)[0],wilson(hi,n)[1]]}
            parameters[p['parameter']]=levels
        return {'n':n,'median_catalog_N':float(np.median(lengths[mask])),
            'median_true_F':float(np.median(truth[mask,1])),
            'median_localized_fraction':float(np.median(localized[mask])), 'parameters':parameters}
    F=truth[:,1]; u=np.log(F/PRIOR_LO[1])/np.log(PRIOR_HI[1]/PRIOR_LO[1])
    groups_F={'F_lt_0.1':F<.1,'F_0.1_to_0.5':(F>=.1)&(F<=.5),'F_gt_0.5':F>.5}
    groups_l={'localized_lt_0.15':localized<.15,'localized_0.15_to_0.35':(localized>=.15)&(localized<=.35),'localized_gt_0.35':localized>.35}
    diagnostics={**provenance,'scope':'Existing archived ranks and existing catalog metadata; width proxy limited to authorized 50 replay',
        'full_SBC_gate_status_unchanged':gp4['status'],
        'H1':{'status':'NOT IDENTIFIABLE from epoch log; limited width proxy reported',
            'per_parameter_loss_history_available':False,'full_1000_widths_available':False,
            'proxy_trials':50,'proxy_widths':widths,
            'limitation':'Endpoint width is not a loss trajectory or an achievable-information reference. No architecture-versus-physics attribution follows.'},
        'H2':{'true_F_prior':[float(PRIOR_LO[1]),float(PRIOR_HI[1])],
            'requested_groups':{k:summarize(m) for k,m in groups_F.items()},
            'exact_prior_quartile_thresholds':[float(PRIOR_LO[1]*(PRIOR_HI[1]/PRIOR_LO[1])**q) for q in (.25,.75)],
            'prior_quartile_groups':{k:summarize(m) for k,m in {'bottom':u<.25,'middle':(u>=.25)&(u<=.75),'top':u>.75}.items()},
            'limitation':'Parameter-conditioned coverage need not be nominal even for exact Bayes. Variation cannot establish logit-gradient compression.'},
        'H3':{'groups':{k:summarize(m) for k,m in groups_l.items()},
            'limitation':'Observed group associations do not identify encoder causality or an information exchange rate; no reference posterior is available.'},
        'rank_bound_method':'For linear interpolated quantiles, ranks at the two boundary order-statistic gaps are ambiguous. Report definite/possible inclusion, not invented exact intervals.',
        'uncertainty':'Wilson 95% intervals are descriptive sampling uncertainty, not modified acceptance criteria.',
        'original_rank_rows':gp4['trials']}
    save('f-diagnostics.json',diagnostics)
    print(json.dumps({'widths':widths,'F_groups':{k:{'n':v['n'],'F':v['parameters']['F']} for k,v in diagnostics['H2']['requested_groups'].items()},
        'localized_groups':{k:{'n':v['n'],'F':v['parameters']['F']} for k,v in diagnostics['H3']['groups'].items()}}),flush=True)
    assert sha(ROOT/'results/posterior-gates.json')==provenance['original_sbc_sha256']
    assert sha(ROOT/'results/training.json')==provenance['training_json_sha256']


if __name__=='__main__':
    main()
