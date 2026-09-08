"""Saved-sample SBC diagnosis and matched localization probes; no retraining.

Host parameters retain physical median / NATURAL LOG width. Only the
authorized coverage comparator changes. Subgroups and matched probes are
diagnostics, never substitutes for the prescribed full acceptance gates.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.stats import binomtest,spearmanr
import torch
from frbsbi.data import ROOT,pad_catalogs,encode_observations
from frbsbi.evaluate import sbc_statistics
from frbsbi.generator import generate_batch
from frbsbi.gate_metrics import PARAMETERS,contraction,widths
from frbsbi.inference import load_checkpoint
from frbsbi.statistics import canonical_manifest_hash
from frbsbi.tolerances import COVERAGE_ABS,CONTRACTION_LIMITS
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from frbsim.catalog import git_provenance

OUT = RUN_DIR/"diagnosis-v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def coverage_detail(p,truth):
    result = []
    for j,name in enumerate(PARAMETERS):
        row = {"parameter":name}
        for nominal,quantiles in ((.68,(.16,.84)),(.95,(.025,.975))):
            lo,hi = np.quantile(p[:,:,j],quantiles,axis=1)
            count = int(((truth[:,j]>=lo)&(truth[:,j]<=hi)).sum())
            n = len(truth)
            ci = binomtest(count,n).proportion_ci(confidence_level=.95,method="exact")
            deviation = 100*(Fraction(count,n)-Fraction(str(nominal)))
            row[str(nominal)] = {"covered":count,"trials":n,"coverage":count/n,
                "deviation_pp":float(deviation),"deviation_pp_exact":str(deviation),
                "binomial_95_interval":[float(ci.low),float(ci.high)]}
        result.append(row)
    return result


def summarize(p,truth):
    corr = np.array([spearmanr(draws,axis=0).statistic for draws in p])
    return {"coverage":coverage_detail(p,truth),
            "contraction_diagnostic":contraction(p,CONTRACTION_LIMITS[256]),
            "median_within_posterior_spearman":np.median(corr,axis=0).tolist(),
            "median_absolute_within_posterior_spearman":np.median(abs(corr),axis=0).tolist(),
            "posterior_mean_bias":(p.mean(1)-truth).mean(0).tolist()}


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    ledger_path = RUN_DIR/"phase2a_gates.json"
    ledger = json.loads(ledger_path.read_text())
    shutil.copyfile(ledger_path,OUT/"ledger-before.json")
    shutil.copyfile(ROOT/"CONDITIONING_FOLLOWUP.md",OUT/"assessment-before.md")
    training = json.loads((RUN_DIR/"training.json").read_text())
    checkpoint = ROOT/training["checkpoint"]
    assert sha(checkpoint)==training["checkpoint_sha256"]==ledger["identity"]["checkpoint_sha256"]
    manifest = json.loads((ROOT/"results/training-data-manifest.json").read_text())
    assert canonical_manifest_hash(manifest)==training["dataset_manifest_sha256"]
    for record in ledger["artifacts"].values():
        assert sha(ROOT/record["path"])==record["sha256"]
    allowed_changes = subprocess.check_output(["git","diff","d83c98c","--name-only","--","src"],cwd=ROOT,text=True).splitlines()
    assert allowed_changes==["src/frbsbi/evaluate.py"],allowed_changes
    sbc = load_npz(ROOT/ledger["artifacts"]["sbc"]["path"])
    p,truth = sbc["posterior"],sbc["truth"]
    corrected,ranks = sbc_statistics(p,truth,jitter_seed=ledger["gates"]["G-P4"]["rank_jitter_seed"])
    old = ledger["gates"]["G-P4"]["parameters"]
    assert ranks.tolist()==ledger["gates"]["G-P4"]["raw_ranks"]
    for a,b in zip(old,corrected):
        assert {k:v for k,v in a.items() if k!="status"}=={k:v for k,v in b.items() if k!="status"}
    changed = [b["parameter"] for a,b in zip(old,corrected) if a["status"]!=b["status"]]
    assert changed==["f_d"]
    n = np.diff(sbc["offsets"])
    localized = np.add.reduceat((sbc["features"][:,6]<.5).astype(np.int64),sbc["offsets"][:-1])
    masks = {"low_localized":100*localized<15*n,
             "medium_localized":(100*localized>=15*n)&(100*localized<35*n),
             "high_localized":100*localized>=35*n,"small_N":n<64,"large_N":n>=64}
    analysis = {**git_provenance(),"checkpoint_sha256":sha(checkpoint),
        "original_sampling_identity":ledger["identity"],"status":"running",
        "scope":"exact-count comparison fix, saved-sample stratification, matched host diagnostics",
        "coverage_exact":coverage_detail(p,truth),"corrected_SBC":corrected,
        "boundary_verification":{"changed_parameter_statuses":changed,"all_numeric_metrics_identical":True,
            "ranks_identical":True,"method":"exact rational covered/trials against inclusive decimal nominal +/- tolerance",
            "tolerance":COVERAGE_ABS,"comparator_source_sha256":sha(ROOT/"src/frbsbi/evaluate.py")},
        "groups":{},"group_definition":"realized localized fraction from observable mask; integer comparisons at 0.15 and 0.35",
        "strata_scope":"reported diagnostics; smaller trial counts have wider sampling uncertainty; no subgroup acceptance gate"}
    for name,mask in masks.items():
        analysis["groups"][name] = {"n_trials":int(mask.sum()),"coverage":coverage_detail(p[mask],truth[mask]),
            "catalog_seeds":np.asarray(ledger["gates"]["G-P4"]["catalog_seeds"])[mask].tolist(),
            "median_N":float(np.median(n[mask])),"median_localized_fraction":float(np.median((localized/n)[mask]))}
    write_json(OUT/"diagnosis.json",analysis)
    print(json.dumps({"corrected_SBC":corrected,"groups":analysis["groups"]},default=str)[:1000],flush=True)
    torch.set_num_threads(training["training_config"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    model,_ = load_checkpoint(checkpoint)
    mixed = load_npz(ROOT/ledger["artifacts"]["contraction-256"]["path"])
    seeds = ledger["gates"]["G-P6"]["by_N"]["256"]["catalog_seeds"]
    analysis["host_probe"] = {"N":256,"trials":len(seeds),"draws":mixed["posterior"].shape[1],
        "catalog_seeds":seeds,"scope":"paired network response to localization; no theoretical information bound",
        "mixed":summarize(mixed["posterior"],mixed["truth"]),"artifacts":{}}
    sampling = ledger["artifacts"]["contraction-256"]
    for label,fraction in (("all_localized",1.),("all_unlocalized",0.)):
        batch = generate_batch(seeds,split="validation",forced_n=256,localized_fraction=fraction)
        features = encode_observations(batch.observations)
        offsets = batch.observations.offsets
        assert np.array_equal(batch.latents.theta,mixed["truth"])
        assert np.array_equal(features[:,:5],mixed["features"][:,:5])
        assert np.array_equal(offsets,mixed["offsets"])
        draws = np.empty_like(mixed["posterior"])
        for first in range(0,len(seeds),sampling["batch_size"]):
            ids = np.arange(first,min(first+sampling["batch_size"],len(seeds)))
            x,mask,_ = pad_catalogs(features,offsets,batch.latents.theta,ids)
            values,_ = model.sample(x,mask,sampling["draws"],seed=sampling["base_sampling_seed"]+first,
                                  steps=sampling["rk4_steps"],check_steps=False)
            assert torch.isfinite(values).all()
            draws[ids] = values.numpy()
        path = OUT/f"{label}.npz"
        np.savez_compressed(path,posterior=draws,truth=batch.latents.theta,features=features,offsets=offsets)
        result = summarize(draws,batch.latents.theta)
        result["median_paired_width_ratio_to_mixed"] = np.median(widths(draws)/widths(mixed["posterior"]),axis=0).tolist()
        analysis["host_probe"][label] = result
        settings = {k:sampling[k] for k in ("base_sampling_seed","batch_size","draws","rk4_steps","n_catalogs")}
        analysis["host_probe"]["artifacts"][label] = {**settings,
            "path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"generator_config":batch.metadata["config"]}
        print(json.dumps({label:result["contraction_diagnostic"]["median_posterior_prior_ratio"]}),flush=True)
        write_json(OUT/"diagnosis.json",analysis)
    assert sha(checkpoint)==training["checkpoint_sha256"]
    # Preserve the old inference identity: only assessment code was changed.
    # No cached samples are relabeled as produced by the corrected evaluator.
    assessment = {"authorization":"human Task 2 boundary fix, 2026-09-08",
        "source_git_sha":git_provenance()["git_sha"],"source_sha256":sha(ROOT/"src/frbsbi/evaluate.py"),
        "posterior_artifact_sha256":ledger["artifacts"]["sbc"]["sha256"],
        "before_ledger":(OUT/"ledger-before.json").relative_to(ROOT).as_posix(),
        "diagnostic_evidence":(OUT/"diagnosis.json").relative_to(ROOT).as_posix(),
        "note":"new coverage assessment only; original sampling identity retained; no cache guard bypass"}
    row = ledger["gates"]["G-P4"]
    row["parameters"] = corrected
    row["status"] = "pass" if all(m["status"]=="pass" for m in corrected) else "fail"
    row["measured"]["parameters"] = deepcopy(corrected)
    row["measured"]["status"] = row["status"]
    row["assessment_revision"] = assessment
    ledger["boundary_fix_diagnosis"] = assessment
    ledger["acceptance_complete"],ledger["exit_status"] = False,1
    analysis.update(status="complete",elapsed_seconds=time.perf_counter()-started,checkpoint_unchanged=True,
                    acceptance_complete=False,gate_criteria_changed=False)
    write_json(OUT/"diagnosis.json",analysis)
    write_json(ledger_path,ledger)
    write_json(ROOT/"results/phase2a_gates.json",ledger)


if __name__=="__main__":
    main()
