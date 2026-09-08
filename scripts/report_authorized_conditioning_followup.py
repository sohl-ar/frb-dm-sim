"""Verify persisted inference metrics and render the authorized followup report."""
from decimal import Decimal
import json
import numpy as np
from frbsbi.acceptance import Acceptance,sha
from frbsbi.data import ROOT
from frbsbi.evaluate import sbc_statistics
from frbsbi.gate_metrics import contraction,PARAMETERS
from frbsbi.tolerances import CONTRACTION_LIMITS,COVERAGE_ABS
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR

run = Acceptance()
run._verify_artifacts()
out = RUN_DIR/"authorized-followup"
r = json.loads((out/"results.json").read_text())
before = json.loads((out/"ledger-before.json").read_text())
assert r["status"]=="requested_checks_completed"
assert before["gates"]["G-P6"]["by_N"]["16"]==run.report["gates"]["G-P6"]["by_N"]["16"]
verified = {"metrics_recomputed_from_saved_samples":True,"N16_original_row_unchanged":True,
            "checkpoint_sha256":run.identity["checkpoint_sha256"],"contraction":{},"localized_counts":{}}
for n,limit in CONTRACTION_LIMITS.items():
    artifact = RUN_DIR/"posterior-artifacts"/f"contraction-{n}.npz"
    with np.load(artifact,allow_pickle=False) as d:
        metric = contraction(d["posterior"],limit)
        assert metric["median_posterior_prior_ratio"]==run.report["gates"]["G-P6"]["by_N"][str(n)]["median_posterior_prior_ratio"]
        f,o = d["features"],d["offsets"]
        counts = np.array([(f[o[i]:o[i+1],6]<.5).sum() for i in range(len(o)-1)])
    verified["contraction"][str(n)] = metric
    verified["localized_counts"][str(n)] = {"median":float(np.median(counts)),"mean":float(counts.mean()),
                                             "zero_localized_catalogs":int((counts==0).sum())}
with np.load(RUN_DIR/"posterior-artifacts/sbc.npz",allow_pickle=False) as d:
    metrics,ranks = sbc_statistics(d["posterior"],d["truth"],jitter_seed=73200)
assert metrics==r["G-P4"]["parameters"]
assert ranks.tolist()==r["G-P4"]["raw_ranks"]
verified["SBC"] = metrics
verified["literal_decimal_band_diagnostic_not_gate_replacement"] = [
    {"parameter":m["parameter"],"coverage68_within_inclusive_band":abs(Decimal(str(m["coverage68"]))-Decimal('.68'))<=Decimal(str(COVERAGE_ABS)),
     "coverage95_within_inclusive_band":abs(Decimal(str(m["coverage95"]))-Decimal('.95'))<=Decimal(str(COVERAGE_ABS))}
    for m in metrics]
verified["fd_boundary_comparison"] = {"binary_float_difference_repr":repr(metrics[0]["coverage95"]-.95),
    "tolerance_repr":repr(COVERAGE_ABS),"recorded_gate_status_preserved":metrics[0]["status"],
    "note":"No comparator, criterion or gate outcome edited; literal decimal band diagnostic only"}
write_json(out/"verification.json",verified)
lines = ["# Authorized inference followup", "", "No retraining. Checkpoint and all source, gate criteria and tolerances unchanged.","",
    f"Saved training selected epoch {r['training']['best_epoch']} of {r['training']['current_epoch']}; "
    f"validation loss {r['training']['best_validation_loss']:.9f}.","",
    "| Parameter | N=16 ratio (<0.7) | Status | N=256 ratio (<0.5) | Status |",
    "|---|---:|---|---:|---|"]
for j,name in enumerate(PARAMETERS):
    a,b = [verified["contraction"][str(n)]["median_posterior_prior_ratio"][j] for n in CONTRACTION_LIMITS]
    lines.append(f"| {name} | {a:.6f} | {'PASS' if a<CONTRACTION_LIMITS[16] else 'FAIL'} | {b:.6f} | {'PASS' if b<CONTRACTION_LIMITS[256] else 'FAIL'} |")
lines += ["",f"SBC: {r['G-P4']['trials']} mixed-N validation catalogs, {r['G-P4']['draws']} draws per catalog.","",
    "| Parameter | 68% coverage | 95% coverage | Adjusted KS p | Recorded status |",
    "|---|---:|---:|---:|---|"]
for m in metrics:
    lines.append(f"| {m['parameter']} | {m['coverage68']:.3%} | {m['coverage95']:.3%} | {m['bonferroni_adjusted_p']:.6f} | {m['status'].upper()} |")
lines += ["", "The f_d FAIL is a floating-point boundary artifact in the existing comparator: "
    f"{metrics[0]['coverage95']} - 0.95 evaluates to {verified['fd_boundary_comparison']['binary_float_difference_repr']}, "
    f"which compares greater than {COVERAGE_ABS}. The measured coverage is on the inclusive upper boundary. "
    "This is reported without changing its recorded status. F exceeds its 68% band; host_sigma_ln is below its 68% band. "
    "Every adjusted KS p passes. The host-median SBC row passes.","",
    "Assessment: the severe prior-like behavior has improved, particularly for f_d and F, but full acceptance still fails. "
    "The host-parameter contraction failures and the non-boundary calibration failures remain substantive under the current spec. "
    "These checks do not establish whether the host thresholds are attainable for the selected catalogs or whether the network is still losing information.","",
    "I would retain the present thresholds for now, flag the comparator boundary issue for a separately reviewed implementation correction, "
    "and compare a few fixed catalogs against a numerical reference posterior before proposing a host-contraction spec change. "
    "Calibration stratified by catalog size would also help; this SBC result averages over the validation N distribution and is not separate calibration at N=16 and N=256.","",
    "G-P6 remains failed and acceptance remains incomplete. The original stopped ledger is archived beside results.json; "
    "both new results were recomputed from saved posterior samples by this reporting script.",""]
(ROOT/"CONDITIONING_FOLLOWUP.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
print(json.dumps(verified,indent=2)[:2500])
