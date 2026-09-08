"""Deterministic CPU CFM training; acceptance is separate from finishing epochs.

Host labels are physical median and NATURAL LOG width. Training targets
use the approved log/prior-normalization/logit transforms inside the model.
Test shards are generated and hashed but never used for fitting or selection.
"""
from dataclasses import dataclass,asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import ROOT,require_pretraining,pad_catalogs
from .model import PosteriorFlow,ModelConfig
from frbsim.catalog import git_provenance


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 5
    minimum_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_norm_clip: float = 5.
    cpu_threads: int = 4
    initialization_seed: int = 71010
    training_noise_seed: int = 71011
    validation_noise_seed: int = 71012
    shuffle_seed: int = 71013


def write_json(path,payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8",newline="\n")
    temporary.replace(path)


def train(config=TrainConfig(),model_config=ModelConfig(),data_root=None,*,run_dir=None,normalization=None):
    require_pretraining()
    architecture = json.loads((ROOT/"results/model-audit.json").read_text(encoding="utf-8"))
    if architecture["status"]!="pass":
        raise RuntimeError("STOP: architecture audit has not passed")
    manifest_path = ROOT/"results/training-data-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise RuntimeError("STOP: training/validation/test data generation incomplete")
    data_root = ROOT/"work/training-data" if data_root is None else Path(data_root)
    for shard in manifest["shards"]:
        if hashlib.sha256((data_root/shard["path"]).read_bytes()).hexdigest()!=shard["sha256"]:
            raise RuntimeError("STOP: training-data hash mismatch")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(config.initialization_seed)
    model = PosteriorFlow(model_config)
    if model_config.statistics_bypass:
        if normalization is None:
            raise RuntimeError("STOP: bypass requires recorded training-only normalization")
        from .statistics import canonical_manifest_hash
        if normalization["dataset_manifest_sha256"]!=canonical_manifest_hash(manifest):
            raise RuntimeError("STOP: normalizer / dataset mismatch")
        model.statistics.set_normalization(normalization)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    training_rng = torch.Generator().manual_seed(config.training_noise_seed)
    order_rng = np.random.default_rng(config.shuffle_seed)
    shards = {split:[s for s in manifest["shards"] if s["split"]==split]
              for split in ("training","validation","test")}
    report = {**git_provenance(),"status":"running","acceptance_complete":False,
              "training_config":asdict(config),"model_config":asdict(model_config),
              "dataset_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "catalog_counts":{split:sum(s["n_catalogs"] for s in entries) for split,entries in shards.items()},
              "torch":torch.__version__,"device":"cpu","precision":"float32; physical transforms use float64",
              "epochs":[],"best_validation_loss":None,"best_epoch":None,
              "checkpoint":"results/checkpoints/phase2a-best.pt",
              "interpretation":"Training completion is not evidence of calibrated or informative posteriors"}
    config_hash = hashlib.sha256(json.dumps({"training":asdict(config),"model":asdict(model_config)},sort_keys=True).encode()).hexdigest()
    report["config_hash"] = config_hash
    trace = ROOT/"results/training.json" if run_dir is None else Path(run_dir)/"training.json"
    ckpts = ROOT/"results/checkpoints" if run_dir is None else Path(run_dir)/"checkpoints"
    if run_dir is not None:
        report["checkpoint"] = (ckpts/"phase2a-best.pt").relative_to(ROOT).as_posix()
        report["dataset_manifest_sha256"] = canonical_manifest_hash(manifest)
        report["manifest_hash_algorithm"] = "sha256 of sorted compact JSON UTF-8"
        report["normalization"] = normalization
    ckpts.mkdir(exist_ok=True)
    start,steps,best,stale = time.perf_counter(),0,float("inf"),0
    write_json(trace,report)
    try:
        for epoch in range(1,config.max_epochs+1):
            model.train()
            epoch_start = time.perf_counter()
            loss_sum,count = 0.,0
            for shard_index in order_rng.permutation(len(shards["training"])):
                shard = shards["training"][shard_index]
                with np.load(data_root/shard["path"],allow_pickle=False) as data:
                    features,offsets,theta = data["features"],data["offsets"],data["theta"]
                order = order_rng.permutation(len(theta))
                for first in range(0,len(order),config.batch_size):
                    ids = order[first:first+config.batch_size]
                    x,padding,y = pad_catalogs(features,offsets,theta,ids)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(x,padding,y,generator=training_rng)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("STOP: nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_norm_clip,error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach())*len(ids); count += len(ids); steps += 1
                    if steps%100==0:
                        report.update({"current_epoch":epoch,"optimizer_steps":steps,"elapsed_seconds":time.perf_counter()-start,
                                       "epoch_catalogs_processed":count})
                        write_json(trace,report)
                        print(json.dumps({"epoch":epoch,"steps":steps,"catalogs_this_epoch":count,
                                          "elapsed_seconds":report["elapsed_seconds"]}),flush=True)
            model.eval()
            val_rng = torch.Generator().manual_seed(config.validation_noise_seed)
            val_sum,val_count = 0.,0
            with torch.no_grad():
                for shard in shards["validation"]:
                    with np.load(data_root/shard["path"],allow_pickle=False) as data:
                        features,offsets,theta = data["features"],data["offsets"],data["theta"]
                    for first in range(0,len(theta),config.batch_size):
                        ids = np.arange(first,min(first+config.batch_size,len(theta)))
                        x,padding,y = pad_catalogs(features,offsets,theta,ids)
                        value = model.loss(x,padding,y,generator=val_rng)
                        if not bool(torch.isfinite(value)):
                            raise RuntimeError("STOP: nonfinite validation loss")
                        val_sum += float(value)*len(ids); val_count += len(ids)
            val_loss = val_sum/val_count
            record = {"epoch":epoch,"training_loss":loss_sum/count,"validation_loss":val_loss,
                      "seconds":time.perf_counter()-epoch_start,"training_catalogs":count,"validation_catalogs":val_count}
            report["epochs"].append(record)
            payload = {"model_state":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(config),
                       "epoch":epoch,"validation_loss":val_loss,"config_hash":config_hash,
                       "dataset_manifest_sha256":report["dataset_manifest_sha256"],"git_sha":report["git_sha"]}
            if run_dir is not None:
                payload["manifest_hash_algorithm"] = report["manifest_hash_algorithm"]
            if val_loss<best:
                best,stale = val_loss,0
                report["best_validation_loss"],report["best_epoch"] = best,epoch
                torch.save(payload,ckpts/"phase2a-best.pt")
            else:
                stale += 1
            torch.save({**payload,"optimizer_state":optimizer.state_dict(),"training_rng_state":training_rng.get_state(),
                        "shuffle_rng_state":order_rng.bit_generator.state},ckpts/"phase2a-last.pt")
            report.update({"current_epoch":epoch,"optimizer_steps":steps,"elapsed_seconds":time.perf_counter()-start})
            write_json(trace,report)
            print(json.dumps(record),flush=True)
            if epoch>=config.minimum_epochs and stale>=config.patience:
                report["stopping_reason"] = "validation patience exhausted"
                break
        else:
            report["stopping_reason"] = "configured epoch limit reached"
        report["status"] = "training_complete_unvalidated"
        report["checkpoint_sha256"] = hashlib.sha256((ckpts/"phase2a-best.pt").read_bytes()).hexdigest()
    except BaseException as exc:
        report["status"],report["error"] = "failed",str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        write_json(trace,report)
    return report


if __name__ == "__main__":
    train()
