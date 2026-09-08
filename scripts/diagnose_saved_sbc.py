"""Finish diagnostic interpretation from archived ranks; never sample a model."""
import json
from pathlib import Path
import numpy as np
from frbsbi.evaluate import validation_catalogs
from frbsbi.generator import PRIOR_LO,PRIOR_HI


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'results/f-diagnostics.json'
    d=json.loads(path.read_text())
    p=json.loads((root/'results/posterior-gates.json').read_text())['gates']['G-P4']
    _,_,truth,_=validation_catalogs(p['trials'])
    u=(truth-PRIOR_LO)/(PRIOR_HI-PRIOR_LO)
    for j in (1,2):
        u[:,j]=np.log(truth[:,j]/PRIOR_LO[j])/np.log(PRIOR_HI[j]/PRIOR_LO[j])
    rank=np.array(p['raw_ranks'])/p['posterior_samples_per_trial']
    comparison={}
    for j,name in enumerate(['f_d','F','host_median','host_sigma_ln']):
        comparison[name]={'n':len(truth),
            'prior_CDF_vs_posterior_rank_correlation':float(np.corrcoef(u[:,j],rank[:,j])[0,1]),
            'mean_absolute_CDF_difference':float(np.mean(abs(u[:,j]-rank[:,j]))),
            'prior_only_central68_coverage':float(np.mean((u[:,j]>=.16)&(u[:,j]<=.84))),
            'prior_only_central95_coverage':float(np.mean((u[:,j]>=.025)&(u[:,j]<=.975)))}
    d['H1']['archived_1000_rank_prior_comparison']=comparison
    d['H1']['interpretation']='Weak contraction is not isolated to F: all four 50-trial width ratios are approximately one, and archived ranks closely track prior CDF positions. Consistent with prior-dominated predictions; not a causal architecture diagnosis.'
    d['H2']['status']='SUPPORTED: coverage varies by true F; NOT ESTABLISHED: logit-gradient mechanism'
    d['H3']['status']='NOT SUPPORTED: proposed recovery in localized-heavy catalogs is absent'
    d['conclusion']='F overcoverage is verified, but an architectural-versus-physical cause is not identified. Broadly prior-like posteriors are the larger warning; do not interpret marginal SBC passes as successful conditional learning.'
    path.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
