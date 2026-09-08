"""Single authorized conditioning experiment; no automatic resume or retry.

Host median / sigma_ln retain NATURAL LOG conventions. Preparation fits
observable scaling only. The launch uses the original training hyperparameters.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from .data import ROOT
from .model import ModelConfig
from .statistics import fit_training_statistics
from .train import TrainConfig,train,write_json

RUN_DIR = ROOT/"results/conditioning-v1"


def prepare():
    RUN_DIR.mkdir(parents=True,exist_ok=True)
    normalizer_path = RUN_DIR/"normalization.json"
    if normalizer_path.exists():
        raise RuntimeError("STOP: existing normalization; do not refit an authorized run")
    normalizer = fit_training_statistics()
    write_json(normalizer_path,normalizer)
    launch = {"run_directory":RUN_DIR.relative_to(ROOT).as_posix(),
              "training":asdict(TrainConfig()),"model":asdict(ModelConfig(statistics_bypass=True)),
              "normalization_sha256":hashlib.sha256(normalizer_path.read_bytes()).hexdigest(),
              "normalization_hash_algorithm":"raw file SHA256",
              "training_attempts_allowed":1,"training_started":False}
    write_json(RUN_DIR/"launch.json",launch)
    return launch


def launch():
    config = json.loads((RUN_DIR/"launch.json").read_text())
    normalizer_path = RUN_DIR/"normalization.json"
    if hashlib.sha256(normalizer_path.read_bytes()).hexdigest()!=config["normalization_sha256"]:
        raise RuntimeError("STOP: normalization hash mismatch")
    if (RUN_DIR/"training.json").exists() or (RUN_DIR/"checkpoints").exists():
        raise RuntimeError("STOP: existing training output; no retry/resume authorized")
    # Exclusive marker persists after interruption/failure. Never remove to retry.
    with (RUN_DIR/"TRAINING_ATTEMPT.lock").open("x") as f:
        f.write("One training attempt claimed. Do not delete or retry.\n")
    return train(TrainConfig(**config["training"]),ModelConfig(**config["model"]),
                 run_dir=RUN_DIR,normalization=json.loads(normalizer_path.read_text()))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","launch"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action=="prepare" else launch(),indent=2))
