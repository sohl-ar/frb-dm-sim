"""Complete conditioned-run acceptance driver; fail once, stop, never repair.

Host parameters remain physical median and NATURAL LOG width. Validation seeds
are disjoint from gradient training, but were used for checkpoint selection.
Cached artifacts are tied to checkpoint, canonical dataset, and source hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from frbsim.catalog import git_provenance
from .data import ROOT,pad_catalogs,encode_observations
from .evaluate import validation_catalogs,sbc_statistics
from .generator import generate_batch,PRIOR_LO,PRIOR_HI
from .inference import load_checkpoint
from .statistics import canonical_manifest_hash
from .train import write_json
from .train_conditioned import RUN_DIR
from .gate_metrics import contraction,information,tarp,smoke_summary,PARAMETERS
from .tolerances import (SBC_TRIALS,SBC_SAMPLES,TARP_TRIALS,WIDTH_TRIALS,SOFT_TRIALS,
                         CONTRACTION_LIMITS,INFORMATION_N,PERMUTATION_W1,COVERAGE_ABS)

GATE_ORDER = ("G-P1","G-P2","G-P6","G-P4","G-P5","G-P8","G-P3","G-P7","smoke")
G_P7_AUTHORIZATION = {
    "date":"2026-09-08","status":"human_authorized",
    "scope":"per-family coverage plus selection-ignored probe; soft and reported-only",
    "cross_family_deferral":"Phase 2b: true cross-trained wrong-family comparison",
    "reason":"The current network was trained on an equal mixture of both population families",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hash():
    paths = sorted((ROOT/"src/frbsbi").glob("*.py")) + sorted((ROOT/"tests/phase2a").glob("*.py"))
    paths += [ROOT/"tests/conftest.py"]
    return hashlib.sha256(b"".join(p.relative_to(ROOT).as_posix().encode()+b"\0"+
                                   p.read_text(encoding="utf-8").encode() for p in paths)).hexdigest()


def empty_ledger():
    return {"schema_version":2,**git_provenance(),"scope":"conditioned Phase 2a run; Phase 1 remains separate",
            "acceptance_complete":False,"exit_status":1,"G-P7_authorization":G_P7_AUTHORIZATION,
            "gates":{n:{"name":n,"status":"not_run","hard":n not in ("G-P3","G-P7","smoke"),
                         "measured":None,"tolerance":None} for n in GATE_ORDER},"artifacts":{}}


class Acceptance:
    def __init__(self,run_dir=RUN_DIR):
        self.run_dir = Path(run_dir)
        training = json.loads((self.run_dir/"training.json").read_text())
        if training["status"]!="training_complete_unvalidated":
            raise RuntimeError("STOP: completed conditioned training required")
        checkpoint = ROOT/training["checkpoint"]
        if sha(checkpoint)!=training["checkpoint_sha256"]:
            raise RuntimeError("STOP: checkpoint hash mismatch")
        manifest = json.loads((ROOT/"results/training-data-manifest.json").read_text())
        manifest_hash = canonical_manifest_hash(manifest)
        if training["dataset_manifest_sha256"]!=manifest_hash:
            raise RuntimeError("STOP: canonical dataset manifest mismatch")
        torch.set_num_threads(training["training_config"]["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.mha.set_fastpath_enabled(False)
        self.model,payload = load_checkpoint(checkpoint)
        if not self.model.config.statistics_bypass or not bool(self.model.statistics.fitted):
            raise RuntimeError("STOP: conditioned checkpoint with fitted bypass required")
        self.identity = {"checkpoint_sha256":sha(checkpoint),"dataset_manifest_sha256":manifest_hash,
                         "implementation_sha256":implementation_hash(),"training_config_hash":payload["config_hash"]}
        self.target = self.run_dir/"phase2a_gates.json"
        if self.target.exists():
            self.report = json.loads(self.target.read_text())
            if self.report.get("identity")!=self.identity:
                raise RuntimeError("STOP: gate evidence identity changed; no automatic reset allowed")
        else:
            self.report = empty_ledger()
            self.report.update(identity=self.identity,trained_checkpoint=training["checkpoint"],
                               training_git_sha=payload["git_sha"],
                               validation_usage="disjoint from gradient fitting; used for checkpoint selection")
        baseline = self.run_dir/"baseline-phase2a-ledger.json"
        if not baseline.exists():
            shutil.copyfile(ROOT/"results/phase2a_gates.json",baseline)
        self.report.setdefault("pretraining",json.loads(baseline.read_text()).get("pretraining",{}))
        self.artifact_dir = self.run_dir/"posterior-artifacts"
        self.artifact_dir.mkdir(exist_ok=True)

    def save(self):
        self.report["smoke_test"] = self.report["gates"]["smoke"]
        write_json(self.target,self.report)
        write_json(ROOT/"results/phase2a_gates.json",self.report)

    def _verify_artifacts(self):
        for key,record in self.report["artifacts"].items():
            if sha(ROOT/record["path"])!=record["sha256"]:
                raise RuntimeError(f"STOP: cached artifact hash mismatch: {key}")

    def sample(self,key,features,offsets,truth,*,seed,draws=SBC_SAMPLES):
        path = self.artifact_dir/f"{key}.npz"
        if key in self.report["artifacts"]:
            if sha(path)!=self.report["artifacts"][key]["sha256"]:
                raise RuntimeError("STOP: posterior cache hash mismatch")
            with np.load(path,allow_pickle=False) as d:
                return d["posterior"]
        if path.exists():
            raise RuntimeError("STOP: unregistered posterior artifact; interrupted work requires review")
        posterior = np.empty((len(truth),draws,4),dtype=np.float64)
        start = time.perf_counter()
        for first in range(0,len(truth),16):
            ids = np.arange(first,min(first+16,len(truth)))
            x,mask,_ = pad_catalogs(features,offsets,truth,ids)
            p,_ = self.model.sample(x,mask,draws,seed=seed+first,steps=32,check_steps=False)
            if not bool(torch.isfinite(p).all()):
                raise RuntimeError("STOP: nonfinite posterior sample")
            posterior[ids] = p.numpy()
            print(json.dumps({"artifact":key,"catalogs_done":int(ids[-1]+1),"seconds":time.perf_counter()-start}),flush=True)
        np.savez_compressed(path,posterior=posterior,truth=truth,offsets=offsets,features=features)
        self.report["artifacts"][key] = {"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),
            "base_sampling_seed":seed,"batch_size":16,"draws":draws,"rk4_steps":32,
            "n_catalogs":len(truth),"seconds":time.perf_counter()-start}
        self.save()
        return posterior

    def generated(self,key,n,*,seed=74000,draws=SBC_SAMPLES,trials=WIDTH_TRIALS,**overrides):
        seeds = list(range(50000,50000+trials))
        batch = generate_batch(seeds,split="validation",forced_n=n,**overrides)
        p = self.sample(key,encode_observations(batch.observations),batch.observations.offsets,
                        batch.latents.theta,seed=seed,draws=draws)
        return p,batch.latents.theta,{"catalog_seeds":seeds,"generator_config":batch.metadata["config"]}

    def run(self,name):
        if self.report.get("stopped_at"):
            raise RuntimeError(f"STOP: earlier failure at {self.report['stopped_at']}; no retry")
        self._verify_artifacts()
        row = self.report["gates"][name]
        if row["status"] in ("pass","reported"):
            return row
        if row["status"]=="running":
            raise RuntimeError("STOP: interrupted gate requires review; no automatic restart")
        row.update(status="running",git_sha=git_provenance()["git_sha"])
        self.save()
        started = time.perf_counter()
        try:
            value = getattr(self,"gate_"+name.replace("-","_"))()
            row.update(value,measured=value,elapsed_seconds=time.perf_counter()-started)
            if row["status"]=="fail":
                self.report["stopped_at"] = name
                raise RuntimeError(f"STOP: {name}: {json.dumps(value)}")
        except BaseException as exc:
            row.update(status="fail",error=str(exc))
            self.report["stopped_at"] = name
            raise
        finally:
            self.save()
        print(json.dumps({name:row}),flush=True)
        return row

    def gate_G_P1(self):
        f,o,t,_ = validation_catalogs(100)
        ids = np.arange(100)
        x,m,_ = pad_catalogs(f,o,t,ids)
        p,ode = self.model.sample(x,m,128,seed=72000,steps=32,check_steps=True)
        if not bool(torch.isfinite(p).all()):
            raise RuntimeError("nonfinite permutation baseline")
        rng = np.random.default_rng(72001)
        values = []
        for _ in range(10):
            perms = [rng.permutation(o[i+1]-o[i]) for i in ids]
            y,ym,_ = pad_catalogs(f,o,t,ids,permutations=perms)
            q,_ = self.model.sample(y,ym,128,seed=72000,steps=32,check_steps=False)
            if not bool(torch.isfinite(q).all()):
                raise RuntimeError("nonfinite permutation posterior")
            values.append(float(np.max(np.abs(np.sort(p.numpy(),axis=1)-np.sort(q.numpy(),axis=1)).mean(1)/(PRIOR_HI-PRIOR_LO))))
        return {"status":"pass" if max(values)<PERMUTATION_W1 else "fail",
                "max_W1_in_prior_widths":max(values),"shuffle_values":values,"ode":ode,
                "n_catalogs":len(ids),"shuffles":len(values),"seed":72000,"shuffle_seed":72001,
                "tolerance":PERMUTATION_W1}

    def gate_G_P2(self):
        f,o,t,_ = validation_catalogs(4)
        x,m,_ = pad_catalogs(f,o,t,np.arange(4))
        a,_ = self.model.sample(x,m,128,seed=72000,check_steps=False)
        b,_ = self.model.sample(x,m,128,seed=72000,check_steps=False)
        c,_ = self.model.sample(x,m,128,seed=72002,check_steps=False)
        left = generate_batch([60000,60001],split="audit")
        right = generate_batch([60000,60001],split="audit")
        replay = all(np.array_equal(getattr(left.observations,k),getattr(right.observations,k))
                     for k in left.observations.__dataclass_fields__)
        equal,distinct = bool(torch.equal(a,b)),not bool(torch.equal(a,c))
        return {"status":"pass" if replay and equal and distinct else "fail",
                "posterior_bit_identical":equal,"different_seed_differs":distinct,"catalog_bit_identical":replay,
                "tolerance":"exact bit identity; different seeds must differ"}

    def gate_G_P4(self):
        f,o,t,seeds = validation_catalogs(SBC_TRIALS)
        p = self.sample("sbc",f,o,t,seed=72100)
        metrics,ranks = sbc_statistics(p,t,jitter_seed=73200)
        return {"status":"pass" if all(m["status"]=="pass" for m in metrics) else "fail",
                "parameters":metrics,"raw_ranks":ranks.tolist(),"catalog_seeds":seeds,
                "rank_jitter_seed":73200,"trials":SBC_TRIALS,"draws":SBC_SAMPLES,
                "tolerance":{"coverage_absolute":COVERAGE_ABS,"KS":"Bonferroni across four parameters; p>.01"}}

    def gate_G_P5(self):
        if "sbc" not in self.report["artifacts"]:
            raise RuntimeError("STOP: run G-P4 before G-P5")
        with np.load(ROOT/self.report["artifacts"]["sbc"]["path"],allow_pickle=False) as d:
            p,t = d["posterior"][:TARP_TRIALS],d["truth"][:TARP_TRIALS]
        return {**tarp(p,t,reference_seed=73300),"trials":len(t),"draws":p.shape[1],
                "posterior_reuse":"first prescribed TARP_TRIALS of the SBC artifact"}

    def gate_G_P6(self):
        rows = {}
        for n,limit in CONTRACTION_LIMITS.items():
            p,_,meta = self.generated(f"contraction-{n}",n)
            rows[str(n)] = {**contraction(p,limit),**meta}
        return {"status":"pass" if all(r["status"]=="pass" for r in rows.values()) else "fail",
                "by_N":rows,"tolerance":CONTRACTION_LIMITS}

    def gate_G_P8(self):
        rows = {}
        for n in INFORMATION_N:
            a,_,meta = self.generated(f"localized-{n}",n,localized_fraction=1.)
            b,_,_ = self.generated(f"unlocalized-{n}",n,localized_fraction=0.)
            rows[str(n)] = {**information(a,b),**meta,"paired_catalogs":"same theta, DM, fluence, sky; only localization differs"}
        return {"status":"pass" if all(r["status"]=="pass" for r in rows.values()) else "fail",
                "by_N":rows,"tolerance":{"strict_lower_bound":1.5}}

    def soft_probe(self,key,n,**overrides):
        p,t,meta = self.generated(key,n,trials=SOFT_TRIALS,**overrides)
        metrics,_ = sbc_statistics(p,t,jitter_seed=73400)
        return {"parameters":metrics,"posterior_mean_bias":(p.mean(1)-t).mean(0).tolist(),
                "parameter_order":PARAMETERS,**meta}

    def gate_G_P3(self):
        return {"status":"reported","by_N":{str(n):self.soft_probe(f"N-generalization-{n}",n)
                for n in (1,4,16,64,256,512,1024)},"tolerance":"soft; report degradation without changing hard criteria"}

    def gate_G_P7(self):
        baseline = self.soft_probe("misspec-baseline",64)
        probes = {"SFR":self.soft_probe("misspec-sfr",64,forced_family=0),
                  "constant":self.soft_probe("misspec-constant",64,forced_family=1),
                  "selection_ignored":self.soft_probe("misspec-no-selection",64,selection_on=False)}
        for row in probes.values():
            row["coverage68_delta_from_baseline"] = [a["coverage68"]-b["coverage68"]
                for a,b in zip(row["parameters"],baseline["parameters"])]
            row["coverage95_delta_from_baseline"] = [a["coverage95"]-b["coverage95"]
                for a,b in zip(row["parameters"],baseline["parameters"])]
        return {"status":"reported","baseline":baseline,"probes":probes,
                "authorization":G_P7_AUTHORIZATION,
                "cross_family_status":"not_identifiable_for_mixture_trained_model",
                "limitation":"Both families were trained jointly; strata are diagnostics, not a cross-trained wrong-family experiment.",
                "tolerance":"soft in Phase 2a; true cross-family experiment remains Phase 2b verification"}

    def gate_smoke(self):
        from .macquart_smoke import smoke_features,TABLE
        f = smoke_features()
        truth = np.zeros((1,4))  # Shape placeholder only; never passed to model or reported as truth.
        p = self.sample("macquart-six",f,np.array([0,len(f)]),truth,seed=73500)
        return {**smoke_summary(p),"input_table":TABLE,"source":"https://arxiv.org/html/2005.13161",
                "N":len(f),"catalogs":1,"tentative_event":"190611","tolerance":"reported, not science"}

    def finalize(self,*,preflight_passed,failures=()):
        self.report["preflight_tests_passed"] = preflight_passed
        self.report["test_failures"] = list(failures)
        if failures:
            self.report.setdefault("stopped_at",failures[0]["nodeid"])
        complete = (preflight_passed and not failures and not self.report.get("stopped_at")
                    and all(r["status"]==("pass" if r["hard"] else "reported") for r in self.report["gates"].values()))
        self.report.update(acceptance_complete=bool(complete),exit_status=0 if complete else 1)
        self.save()
        return complete


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("validation",))
    args = parser.parse_args()
    try:
        run = Acceptance()
        for name in ("G-P6","G-P4"):
            run.run(name)
    except Exception as exc:
        print(str(exc),flush=True)
        raise SystemExit(1)
    print("Contraction and SBC passed; full pytest acceptance remains required.")
