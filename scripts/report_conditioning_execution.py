"""Read recorded execution results only; do not generate data or run inference."""
import argparse
import json
from frbsbi.train_conditioned import RUN_DIR

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--section",choices=("training","gates","all"),default="all")
args = parser.parse_args()
result = {}
if args.section in ("training","all"):
    path = RUN_DIR/"training.json"
    if path.exists():
        r = json.loads(path.read_text())
        keys = ("status","current_epoch","optimizer_steps","epoch_catalogs_processed","elapsed_seconds",
                "best_epoch","best_validation_loss","stopping_reason","checkpoint_sha256","error","epochs")
        result["training"] = {k:r[k] for k in keys if k in r}
    else:
        result["training"] = {"status":"not_started"}
if args.section in ("gates","all"):
    path = RUN_DIR/"phase2a_gates.json"
    if path.exists():
        r = json.loads(path.read_text())
        result["acceptance_complete"] = r["acceptance_complete"]
        result["stopped_at"] = r.get("stopped_at")
        result["identity"] = r.get("identity")
        result["G-P7_authorization"] = r.get("G-P7_authorization")
        result["gates"] = {}
        for name,row in r["gates"].items():
            keys = ("status","hard","error","tolerance","elapsed_seconds","parameters","at_68_95",
                    "absolute_deviations","posterior_mean","central68_width","cross_family_status")
            entry = {k:row[k] for k in keys if k in row}
            if "by_N" in row:
                entry["by_N"] = {n:{k:v for k,v in x.items() if k not in
                    ("per_catalog_ratios","per_pair_ratios","catalog_seeds","generator_config")}
                    for n,x in row["by_N"].items()}
            if "probes" in row:
                entry["probes"] = row["probes"]
            result["gates"][name] = entry
    else:
        result["gates"] = {"status":"not_run"}
print(json.dumps(result,indent=2,allow_nan=False))
