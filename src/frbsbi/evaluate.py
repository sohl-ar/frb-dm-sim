"""First trained-posterior gates: permutation, determinism, then SBC.

Ranks and coverage are in PHYSICAL theta space (host sigma is NATURAL LOG
width). Stop on failure; remaining Phase 2a gates stay explicitly unrun.
SBC uses the prescribed disjoint validation catalog seeds. Validation data
were also used for loss-based checkpoint selection; never for gradients.
"""
import hashlib
from fractions import Fraction
from numbers import Integral
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import kstest
import torch
from frbsim.catalog import git_provenance
from .data import ROOT,pad_catalogs
from .inference import load_checkpoint
from .generator import PRIOR_LO,PRIOR_HI
from .tolerances import PERMUTATION_W1,SBC_TRIALS,SBC_SAMPLES,SBC_KS_ALPHA,COVERAGE_ABS


def coverage_in_band(covered, trials, nominal, tolerance=COVERAGE_ABS):
    """Exact inclusive coverage comparison; no epsilon or rounded deviations.

    Coverage is a count/trial ratio. Interpret the specified decimal nominal
    probability and tolerance exactly, so both endpoints of +/-2pp are included.
    """
    if (not isinstance(covered,Integral) or not isinstance(trials,Integral)
            or trials<=0 or not 0<=covered<=trials):
        raise ValueError("coverage requires integer counts within a positive trial count")
    return abs(Fraction(int(covered),int(trials))-Fraction(str(nominal)))<=Fraction(str(tolerance))


def sbc_statistics(posterior,truth,*,jitter_seed):
    """Uniform randomized ranks and central intervals, all in physical units."""
    posterior,truth = np.asarray(posterior),np.asarray(truth)
    if (posterior.ndim!=3 or posterior.shape[2]!=4 or truth.shape!=(posterior.shape[0],4)
            or not np.isfinite(posterior).all() or not np.isfinite(truth).all()):
        raise ValueError("SBC requires finite [trial, draw, four physical parameters] samples and matching truth")
    raw_ranks = np.sum(posterior<truth[:,None,:],axis=1)
    ranks = (raw_ranks+np.random.default_rng(jitter_seed).random(raw_ranks.shape))/(posterior.shape[1]+1)
    lower68,upper68 = np.quantile(posterior,[.16,.84],axis=1)
    lower95,upper95 = np.quantile(posterior,[.025,.975],axis=1)
    covered68 = np.sum((truth>=lower68)&(truth<=upper68),axis=0)
    covered95 = np.sum((truth>=lower95)&(truth<=upper95),axis=0)
    coverage68,coverage95 = covered68/len(truth),covered95/len(truth)
    tests = []
    for j,name in enumerate(("f_d","F","host_median","host_sigma_ln")):
        statistic = kstest(ranks[:,j],"uniform")
        adjusted_p = min(1.,4*statistic.pvalue)  # Four parameters, one recorded repetition.
        passed = (adjusted_p>SBC_KS_ALPHA and coverage_in_band(covered68[j],len(truth),.68)
                  and coverage_in_band(covered95[j],len(truth),.95))
        tests.append({"parameter":name,"KS_D":float(statistic.statistic),"raw_p":float(statistic.pvalue),
                      "bonferroni_adjusted_p":float(adjusted_p),"coverage68":float(coverage68[j]),
                      "coverage95":float(coverage95[j]),"status":"pass" if passed else "fail"})
    return tests,raw_ranks


def validation_catalogs(n, *, manifest_path=None, data_root=None):
    manifest_path = ROOT/"results/training-data-manifest.json" if manifest_path is None else Path(manifest_path)
    data_root = ROOT/"work/training-data" if data_root is None else Path(data_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    f,t,seeds,lengths = [],[],[],[]
    remaining = n
    for shard in manifest["shards"]:
        if shard["split"]!="validation" or remaining==0:
            continue
        path = data_root/shard["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=shard["sha256"]:
            raise RuntimeError("STOP: validation-data hash mismatch")
        with np.load(path,allow_pickle=False) as data:
            take = min(remaining,len(data["theta"]))
            f.append(data["features"][:data["offsets"][take]])
            t.append(data["theta"][:take])
            seeds.extend(data["seeds"][:take].tolist())
            lengths.extend(np.diff(data["offsets"][:take+1]).tolist())
        remaining -= take
    if remaining:
        raise RuntimeError("insufficient validation catalogs")
    return np.concatenate(f),np.concatenate(([0],np.cumsum(lengths))),np.concatenate(t),seeds


def run_gates():
    training = json.loads((ROOT/"results/training.json").read_text(encoding="utf-8"))
    if training["status"]!="training_complete_unvalidated":
        raise RuntimeError("STOP: wait for completed training before posterior acceptance tests")
    path = ROOT/training["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=training["checkpoint_sha256"]:
        raise RuntimeError("STOP: checkpoint hash mismatch")
    torch.set_num_threads(training["training_config"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    model,payload = load_checkpoint(path)
    manifest_path = ROOT/"results/training-data-manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=payload["dataset_manifest_sha256"]:
        raise RuntimeError("STOP: validation manifest differs from training provenance")
    features,offsets,truth,catalog_seeds = validation_catalogs(SBC_TRIALS)
    report = {**git_provenance(),"checkpoint_sha256":training["checkpoint_sha256"],"training_git_sha":payload["git_sha"],
              "training_config_hash":payload["config_hash"],"dataset_manifest_sha256":payload["dataset_manifest_sha256"],
              "torch":str(torch.__version__),"device":"cpu","precision":"float32 flow, float64 physical parameters",
              "scope":"G-P1, G-P2 and G-P4; remaining gates require separate evidence",
              "validation_usage":"Disjoint from gradient training; used for validation-loss checkpoint selection as specified",
              "acceptance_complete":False,"gates":{f"G-P{i}":{"status":"not_run"} for i in range(1,9)}}
    target = ROOT/"results/posterior-gates.json"
    def save():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(target)
    def stop(name):
        report["stopped_at"] = name; save(); return report
    start = time.perf_counter()
    ids = np.arange(100)
    x,mask,_ = pad_catalogs(features,offsets,truth,ids)
    p,ode = model.sample(x,mask,128,seed=72000,steps=32,check_steps=True)
    p = p.numpy()
    if not np.isfinite(p).all():
        report["gates"]["G-P1"] = {"status":"fail","reason":"Nonfinite posterior samples"}
        return stop("G-P1")
    rng = np.random.default_rng(72001)
    w1_values = []
    for _ in range(10):
        permutations = [rng.permutation(offsets[i+1]-offsets[i]) for i in ids]
        shuffled,shuffled_mask,_ = pad_catalogs(features,offsets,truth,ids,permutations=permutations)
        q,_ = model.sample(shuffled,shuffled_mask,128,seed=72000,steps=32,check_steps=False)
        if not bool(torch.isfinite(q).all()):
            report["gates"]["G-P1"] = {"status":"fail","reason":"Nonfinite shuffled posterior samples"}
            return stop("G-P1")
        w1_values.append(float(np.max(np.mean(abs(np.sort(p,axis=1)-np.sort(q.numpy(),axis=1)),axis=1)/(PRIOR_HI-PRIOR_LO))))
    report["gates"]["G-P1"] = {"status":"pass" if max(w1_values)<PERMUTATION_W1 else "fail",
        "n_catalogs":100,"shuffles_per_catalog":10,"posterior_samples":128,"seed":72000,"shuffle_seed":72001,
        "max_W1_in_prior_widths":max(w1_values),"tolerance":PERMUTATION_W1,"ode":ode,
        "elapsed_seconds":time.perf_counter()-start}
    save()
    print(json.dumps({"G-P1":report["gates"]["G-P1"]}),flush=True)
    if report["gates"]["G-P1"]["status"]!="pass":
        return stop("G-P1")
    same,_ = model.sample(x,mask,128,seed=72000,steps=32,check_steps=False)
    different,_ = model.sample(x,mask,128,seed=72002,steps=32,check_steps=False)
    equal = np.array_equal(p,same.numpy())
    distinct = not np.array_equal(p,different.numpy())
    generator_audit = json.loads((ROOT/"results/generator-audit.json").read_text(encoding="utf-8"))
    report["gates"]["G-P2"] = {"status":"pass" if equal and distinct and generator_audit["bit_identical_replay"] else "fail",
        "posterior_bit_identical":equal,"different_seed_differs":distinct,
        "catalog_bit_identical":generator_audit["bit_identical_replay"],"seed":72000,"different_seed":72002,
        "catalog_replay_evidence":"results/generator-audit.json"}
    save()
    if report["gates"]["G-P2"]["status"]!="pass":
        return stop("G-P2")
    started = time.perf_counter()
    posterior = np.empty((SBC_TRIALS,SBC_SAMPLES,4),dtype=np.float64)
    for first in range(0,SBC_TRIALS,16):
        indices = np.arange(first,min(first+16,SBC_TRIALS))
        x,mask,_ = pad_catalogs(features,offsets,truth,indices)
        values,_ = model.sample(x,mask,SBC_SAMPLES,seed=72100+first,steps=32,check_steps=False)
        if not bool(torch.isfinite(values).all()):
            report["gates"]["G-P4"] = {"status":"fail","reason":"Nonfinite SBC posterior samples",
                                       "first_trial_in_batch":first}
            return stop("G-P4")
        posterior[indices] = values.numpy()
        if first%128==0:
            print(json.dumps({"SBC_trials_done":int(indices[-1]+1),"elapsed_seconds":time.perf_counter()-started}),flush=True)
    tests,raw_ranks = sbc_statistics(posterior,truth,jitter_seed=73200)
    report["gates"]["G-P4"] = {"status":"pass" if all(t["status"]=="pass" for t in tests) else "fail",
        "trials":SBC_TRIALS,"posterior_samples_per_trial":SBC_SAMPLES,"repetitions":1,
        "sampling_seeds":[72100+i for i in range(0,SBC_TRIALS,16)],
        "solver":{"method":"torchdiffeq rk4","steps":32,"step_doubling_evidence":"G-P1"},
        "catalog_seeds":catalog_seeds,"rank_jitter_seed":73200,"parameters":tests,
        "inference_seconds":time.perf_counter()-started,"tolerance":{"adjusted_KS_p_min":SBC_KS_ALPHA,"coverage_absolute":COVERAGE_ABS},
        "coordinate_space":"physical theta; host width is sigma_ln", "raw_ranks":raw_ranks.tolist()}
    save()
    print(json.dumps({"G-P4_status":report["gates"]["G-P4"]["status"],"parameters":tests}),flush=True)
    if report["gates"]["G-P4"]["status"]!="pass":
        return stop("G-P4")
    return report


if __name__ == "__main__":
    result = run_gates()
    # This subset cannot close the full Phase 2a acceptance ledger.
    raise SystemExit(0 if result["acceptance_complete"] else 1)
