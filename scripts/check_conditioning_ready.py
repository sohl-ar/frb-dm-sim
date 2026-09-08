"""Read-only execution readiness and provenance check; never trains or samples."""
from dataclasses import asdict
import hashlib
import json
import subprocess
import sys
import numpy as np
import torch
import sbi
from frbsbi.data import ROOT,require_pretraining
from frbsbi.train import TrainConfig,write_json
from frbsbi.model import ModelConfig
from frbsbi.statistics import canonical_manifest_hash,StatisticsBypass
from frbsbi.train_conditioned import RUN_DIR
from frbsim.catalog import git_provenance

require_pretraining()
config = json.loads((RUN_DIR/"launch.json").read_text())
normalizer_path = RUN_DIR/"normalization.json"
normalizer = json.loads(normalizer_path.read_text())
manifest = json.loads((ROOT/"results/training-data-manifest.json").read_text())
assert config["training"]==asdict(TrainConfig()),"STOP: hyperparameters changed"
assert config["model"]==asdict(ModelConfig(statistics_bypass=True)),"STOP: model configuration mismatch"
assert config["normalization_sha256"]==hashlib.sha256(normalizer_path.read_bytes()).hexdigest()
assert normalizer["dataset_manifest_sha256"]==canonical_manifest_hash(manifest)
assert normalizer["n_catalogs"]==sum(s["n_catalogs"] for s in manifest["shards"] if s["split"]=="training")
StatisticsBypass().set_normalization(normalizer)
assert sbi.__version__=="0.27.0"
for name in ("TRAINING_ATTEMPT.lock","training.json","checkpoints"):
    assert not (RUN_DIR/name).exists(),f"STOP: training already attempted: {name}"
for shard in manifest["shards"]:
    path = ROOT/"work/training-data"/shard["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==shard["sha256"],f"STOP: shard mismatch: {path}"
old = json.loads((ROOT/"results/training.json").read_text())
assert hashlib.sha256((ROOT/old["checkpoint"]).read_bytes()).hexdigest()==old["checkpoint_sha256"]
# No source or prior gate criterion changes in the verified simulator.
physics_diff = subprocess.check_output(["git","diff","b0917a9","--","src/frbsim"],cwd=ROOT,text=True)
assert not physics_diff,"STOP: verified simulator source changed"
report = {**git_provenance(),"status":"pass","scope":"read-only readiness; training not started",
          "shards_verified":len(manifest["shards"]),"normalization_catalogs":normalizer["n_catalogs"],
          "training_config":config["training"],"model_config":config["model"],
          "baseline_checkpoint_sha256":old["checkpoint_sha256"],
          "python":sys.version,"numpy":np.__version__,"torch":str(torch.__version__),"sbi":sbi.__version__,
          "physics_diff_from_baseline":"empty"}
write_json(RUN_DIR/"readiness.json",report)
print(json.dumps(report,indent=2))
