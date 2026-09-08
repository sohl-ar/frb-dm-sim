"""2026-09-08 human-authorized inference continuation after G-P6 N=16 failure.

Run only N=256 contraction and independent SBC. Existing methods, seeds,
draw counts, thresholds and checkpoint are unchanged. No training or edits
to src/tests. Preserve the original stopped ledger before appending evidence.
"""
import copy
import json
from pathlib import Path
import shutil
import time
from frbsbi.acceptance import Acceptance,sha
from frbsbi.gate_metrics import contraction
from frbsbi.tolerances import CONTRACTION_LIMITS
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from frbsbi.data import ROOT
from frbsim.catalog import git_provenance


def main():
    run = Acceptance()
    run._verify_artifacts()
    before = copy.deepcopy(run.report)
    if before["gates"]["G-P6"]["by_N"]["256"]["status"]!="not_run" or before["gates"]["G-P4"]["status"]!="not_run":
        raise RuntimeError("STOP: requested followup evidence already exists; inspect rather than rerun")
    out = RUN_DIR/"authorized-followup"
    out.mkdir(exist_ok=False)
    shutil.copyfile(run.target,out/"ledger-before.json")
    checkpoint = ROOT/run.report["trained_checkpoint"]
    checkpoint_hash = sha(checkpoint)
    authorization = {"date":"2026-09-08","source":"human request in this task",
        "scope":"N=256 contraction then independent SBC, even though G-P6 N=16 failed",
        "unchanged":"checkpoint, source, gates, tolerances, sample counts, seeds and solver",
        "prior_stop_retained":before.get("stopped_at"),"training":"forbidden; not invoked"}
    report = {**git_provenance(),"identity":run.identity,"authorization":authorization,
        "original_ledger_sha256":sha(out/"ledger-before.json"),"status":"running",
        "training":{k:json.loads((RUN_DIR/"training.json").read_text())[k] for k in
                    ("best_epoch","best_validation_loss","current_epoch","elapsed_seconds")}}
    run.report["authorized_continuation"] = authorization
    write_json(out/"results.json",report)
    start = time.perf_counter()
    try:
        p,_,metadata = run.generated("contraction-256",256)
        result = {**contraction(p,CONTRACTION_LIMITS[256]),**metadata,**git_provenance()}
        row = run.report["gates"]["G-P6"]
        row["by_N"]["256"] = result
        row["measured"]["by_N"]["256"] = result
        row["original_stop_error"] = row.pop("error",None)
        row["error"] = "G-P6 remains failed under unchanged criteria; see both measured N rows"
        report["G-P6"] = copy.deepcopy(row)
        run.save()
        write_json(out/"results.json",report)
        print(json.dumps({"N256_contraction":result["median_posterior_prior_ratio"],"status":result["status"]}),flush=True)
        # Directly execute the unchanged SBC method: the user explicitly
        # authorized independence from contraction's orchestration stop.
        sbc = run.gate_G_P4()
        run.report["gates"]["G-P4"].update(sbc,measured=copy.deepcopy(sbc),**git_provenance())
        report["G-P4"] = copy.deepcopy(run.report["gates"]["G-P4"])
        report["status"] = "requested_checks_completed"
        assert sha(checkpoint)==checkpoint_hash,"STOP: checkpoint changed during inference"
        report["checkpoint_unchanged"] = True
        report["acceptance_complete"] = False
        print(json.dumps({"SBC_status":sbc["status"],"parameters":sbc["parameters"]}),flush=True)
    except BaseException as exc:
        report.update(status="execution_error",error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        run.report["acceptance_complete"] = False
        run.report["exit_status"] = 1
        run.save()
        write_json(out/"results.json",report)


if __name__=="__main__":
    main()
