"""Single authorized conditioning experiment; no automatic resume or retry.

Host median / sigma_ln retain NATURAL LOG conventions. Preparation fits
observable scaling only. The launch uses the original training hyperparameters.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
from .data import ROOT
from .model import ModelConfig
from .statistics import fit_training_statistics, NORMALIZATION_POLICY, canonical_manifest_hash
from .composition import EXPANDED_COMPOSITION
from .train import TrainConfig,train,write_json

RUN_DIR = ROOT/"results/conditioning-v1"
COMPOSITION_RUN_DIR = ROOT/"results/conditioning-v2"


def prepare(run_dir=RUN_DIR, *, data_root=None, manifest_path=None):
    run_dir = Path(run_dir).resolve()
    data_root = ROOT/"work/training-data" if data_root is None else Path(data_root).resolve()
    manifest_path = ROOT/"results/training-data-manifest.json" if manifest_path is None else Path(manifest_path).resolve()
    for path in (run_dir,data_root,manifest_path):
        path.relative_to(ROOT)  # Keep run paths portable within the repository.
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("STOP: existing run evidence; do not refit or overwrite")
    manifest = json.loads(manifest_path.read_text())
    if run_dir == COMPOSITION_RUN_DIR and manifest.get("composition",{}).get("policy") != EXPANDED_COMPOSITION:
        raise RuntimeError("STOP: composition-v2 run requires the expanded dataset")
    normalizer = fit_training_statistics(data_root,manifest_path=manifest_path)
    run_dir.mkdir(parents=True,exist_ok=True)
    normalizer_path = run_dir/"normalization.json"
    write_json(normalizer_path,normalizer)
    launch = {"run_directory":run_dir.relative_to(ROOT).as_posix(),
              "dataset_manifest_path":manifest_path.relative_to(ROOT).as_posix(),
              "data_root":data_root.relative_to(ROOT).as_posix(),
              "dataset_manifest_sha256":canonical_manifest_hash(manifest),
              "normalization_policy":NORMALIZATION_POLICY,
              "composition":manifest.get("composition",{"policy":"beta-2-6-v1"}),
              "training":asdict(TrainConfig()),"model":asdict(ModelConfig(statistics_bypass=True)),
              "normalization_sha256":hashlib.sha256(normalizer_path.read_bytes()).hexdigest(),
              "normalization_hash_algorithm":"raw file SHA256",
              "training_attempts_allowed":1,"training_started":False}
    write_json(run_dir/"launch.json",launch)
    return launch


def launch(run_dir=RUN_DIR):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir/"launch.json").read_text())
    if ROOT/config["run_directory"] != run_dir:
        raise RuntimeError("STOP: launch run-directory mismatch")
    normalizer_path = run_dir/"normalization.json"
    if hashlib.sha256(normalizer_path.read_bytes()).hexdigest()!=config["normalization_sha256"]:
        raise RuntimeError("STOP: normalization hash mismatch")
    manifest_path = ROOT/config.get("dataset_manifest_path","results/training-data-manifest.json")
    data_root = ROOT/config.get("data_root","work/training-data")
    normalizer = json.loads(normalizer_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if normalizer["dataset_manifest_sha256"] != canonical_manifest_hash(manifest):
        raise RuntimeError("STOP: launch dataset changed after normalization")
    if run_dir == COMPOSITION_RUN_DIR and (
            manifest.get("composition",{}).get("policy") != EXPANDED_COMPOSITION
            or normalizer.get("encoding_policy") != NORMALIZATION_POLICY):
        raise RuntimeError("STOP: composition-v2 requires expanded data and raw binary normalization")
    if (run_dir/"training.json").exists() or (run_dir/"checkpoints").exists():
        raise RuntimeError("STOP: existing training output; no retry/resume authorized")
    # Exclusive marker persists after interruption/failure. Never remove to retry.
    with (run_dir/"TRAINING_ATTEMPT.lock").open("x") as f:
        f.write("One training attempt claimed. Do not delete or retry.\n")
    # Preserve the baseline preflight evidence before future full pytest runs
    # regenerate those reports. This archive never overwrites existing files.
    archive = run_dir/"baseline-evidence"
    archive.mkdir()
    names = ("phase2a_gates.json","generator-audit.json","model-audit.json",
             "prior-predictive.json","selection-audit.json","training.json","posterior-gates.json")
    hashes = {}
    for name in names:
        source = ROOT/"results"/name
        shutil.copyfile(source,archive/name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(archive/"manifest.json",hashes)
    return train(TrainConfig(**config["training"]),ModelConfig(**config["model"]),
                 data_root,run_dir=run_dir,normalization=normalizer,manifest_path=manifest_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","launch"))
    parser.add_argument("--composition-v2",action="store_true",
                        help="Use isolated v2 dataset/manifest/run; never overwrite conditioning-v1")
    args = parser.parse_args()
    run_dir = COMPOSITION_RUN_DIR if args.composition_v2 else RUN_DIR
    if args.action == "prepare":
        kwargs = {"data_root":ROOT/"work/training-data-v2",
                  "manifest_path":ROOT/"results/training-data-v2-manifest.json"} if args.composition_v2 else {}
        result = prepare(run_dir,**kwargs)
    else:
        result = launch(run_dir)
    print(json.dumps(result,indent=2))
