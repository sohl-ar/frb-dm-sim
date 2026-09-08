"""Acceptance metrics in physical theta coordinates; host width is sigma_ln.

TARP: Lemos et al. 2023, https://arxiv.org/abs/2302.03026. We use sbi
0.27.0's sample-based implementation, with fixed histogram boundaries that
include both tested probabilities. Distances use physical prior-support
normalization, not the flow's logit coordinates. References are independent
uniform points in that support. A sample exactly on an ECP boundary belongs
to the next histogram bin (sbi / torch histogram convention).
"""
import numpy as np
import torch
from .generator import PRIOR_LO,PRIOR_HI,LOG_PRIOR
from .tolerances import WIDTH_QUANTILES,COVERAGE_ABS,INFORMATION_RATIO_MIN

PARAMETERS = ("f_d","F","host_median","host_sigma_ln")


def checked(samples):
    p = np.asarray(samples,dtype=np.float64)
    if p.ndim!=3 or p.shape[2]!=4 or p.shape[1]<2 or not np.isfinite(p).all():
        raise ValueError("finite [catalog,draw,4] posterior samples required")
    return p


def prior_quantiles(q):
    q = np.asarray(q)[:,None]
    lo,hi = np.where(LOG_PRIOR,np.log(PRIOR_LO),PRIOR_LO),np.where(LOG_PRIOR,np.log(PRIOR_HI),PRIOR_HI)
    v = lo+q*(hi-lo)
    return np.where(LOG_PRIOR,np.exp(v),v)


def widths(p):
    lo,hi = np.quantile(checked(p),WIDTH_QUANTILES,axis=1)
    return hi-lo


def contraction(p,limit):
    prior = np.diff(prior_quantiles(WIDTH_QUANTILES),axis=0)[0]
    individual = widths(p)/prior
    ratios = np.median(individual,axis=0)
    return {"status":"pass" if bool(np.all(ratios<limit)) else "fail",
            "parameter_order":PARAMETERS,"median_posterior_prior_ratio":ratios.tolist(),
            "per_catalog_ratios":individual.tolist(),"prior_width":prior.tolist(),
            "width_quantiles":WIDTH_QUANTILES,"tolerance":{"strict_upper_bound":limit}}


def information(localized,unlocalized):
    a,b = widths(localized)[:,0],widths(unlocalized)[:,0]
    if a.shape!=b.shape or np.any(a<=0):
        raise ValueError("paired positive localized widths required")
    ratios = b/a
    median = float(np.median(ratios))
    return {"status":"pass" if median>INFORMATION_RATIO_MIN else "fail",
            "median_unlocalized_localized_fd_width_ratio":median,"per_pair_ratios":ratios.tolist(),
            "width_quantiles":WIDTH_QUANTILES,"tolerance":{"strict_lower_bound":INFORMATION_RATIO_MIN}}


def tarp(p,truth,*,reference_seed):
    import sbi
    from sbi.diagnostics.tarp import _run_tarp
    if sbi.__version__!="0.27.0":
        raise RuntimeError("STOP: TARP adapter requires verified sbi==0.27.0")
    p = checked(p)
    truth = np.asarray(truth,dtype=np.float64)
    if truth.shape!=(len(p),4) or not np.isfinite(truth).all():
        raise ValueError("TARP truth must match samples")
    samples = torch.from_numpy(((p-PRIOR_LO)/(PRIOR_HI-PRIOR_LO)).transpose(1,0,2).copy())
    theta = torch.from_numpy((truth-PRIOR_LO)/(PRIOR_HI-PRIOR_LO))
    refs = torch.from_numpy(np.random.default_rng(reference_seed).random(theta.shape))
    # Explicit tensor bin edges are accepted by torch.histogram in the pinned
    # sbi implementation. Avoid its default empirical min/max histogram range.
    # sbi divides integer distance ranks into float32, irrespective of input
    # dtype; torch.histogram requires edges to have that same dtype.
    edges = torch.linspace(0,1,101,dtype=torch.float32)
    ecp,alpha = _run_tarp(samples,theta,refs,num_bins=edges,z_score_theta=False)
    counts = [int(round(float(ecp[int(round(q*100))])*len(truth))) for q in (.68,.95)]
    values = [count/len(truth) for count in counts]
    deviation = np.abs(np.asarray(values)-[.68,.95])
    passed = all(abs(count-q*len(truth))<=COVERAGE_ABS*len(truth) for count,q in zip(counts,(.68,.95)))
    return {"status":"pass" if passed else "fail",
            "ecp":ecp.tolist(),"alpha":alpha.tolist(),"at_68_95":values,
            "absolute_deviations":deviation.tolist(),"tolerance":COVERAGE_ABS,
            "reference_seed":reference_seed,"references":refs.tolist(),"sbi_version":sbi.__version__,
            "ecp_counts_at_68_95":counts,"ecp_count_convention":"recover integer counts from sbi float32 ECP, divide by trial count",
            "coordinate_space":"physical theta divided by prior support widths; independent uniform references",
            "implementation":"sbi.diagnostics.tarp._run_tarp; fixed 0..1 histogram edges"}


def smoke_summary(p):
    p = checked(p)
    return {"status":"reported","parameter_order":PARAMETERS,"posterior_mean":p.mean(1).tolist(),
            "central68_width":widths(p).tolist(),"interpretation":"schema and execution smoke test only; not science"}
