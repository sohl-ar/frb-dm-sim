"""Generate the summary page from saved execution evidence, without rerunning it."""
import json
from pathlib import Path
from frbsbi.preflight import write_phase2a_report

ROOT = Path(__file__).resolve().parents[1]


def update():
    ledger_path = ROOT/"results/phase2a_gates.json"
    previous = json.loads(ledger_path.read_text(encoding="utf-8"))
    write_phase2a_report(ROOT,previous["pretraining"]["T1"],
                        previous["preflight_tests_passed"],previous["test_failures"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    training = json.loads((ROOT/"results/training.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT/"results/training-data-manifest.json").read_text(encoding="utf-8"))
    failed = [name for name,gate in ledger["gates"].items() if gate["status"]=="fail"]
    lines = ["# Phase 2a execution checkpoint — incomplete", "",
             "This page is generated from saved run JSON; it does not rerun training or gates.", ""]
    if "G-P4" in failed:
        lines += ["**The trained posteriors are miscalibrated. Phase 2a acceptance fails,",
                  "and Phase 3 is blocked.** The hard-gate STOP protocol was applied;",
                  "no tolerance was changed and no remedial training was launched.", ""]
    elif failed:
        lines += [f"**Hard-gate STOP: {', '.join(failed)} failed. Phase 3 is blocked.**", ""]
    else:
        lines += ["**Phase 2a acceptance is incomplete; Phase 3 remains blocked.**",
                  "Training completion alone does not establish posterior calibration or information content.", ""]
    lines += ["## Pretraining and implementation", "",
              "| Check | Recorded status |", "|---|---|"]
    lines += [f"| {name} | {item['status']} |" for name,item in ledger["pretraining"].items()]
    lines += ["", "T1 established independent RNG draws and disjoint catalog seeds. T2 includes",
              "the authorized Schechter selection model and measured generator performance.",
              "T3's per-event CDFs were reported before training:", "",
              "| FRB | Prior-predictive CDF |", "|---|---:|"]
    lines += [f"| {name} | {value:.6f} |" for name,value in ledger["pretraining"]["T3"]["event_CDFs"].items()]
    lines += ["", "The native Set Transformer and conditional flow-matching decoder implement",
              "the padding mask, missing-redshift embedding and bounded-prior/logit transforms.",
              "[Architecture audit](results/model-audit.json) records short-run determinism,",
              "padding/shuffle invariance, gradients and ODE step doubling.", "",
              "## Training", "",
              f"Status: `{training['status']}`. Generated {data['total_catalogs']:,} catalogs",
              f"containing {data['total_bursts']:,} bursts; catalog counts by split:"]
    lines += [f"- {split}: {count:,}." for split,count in training["catalog_counts"].items()]
    lines += ["", f"Completed epochs: {len(training['epochs'])}. Best epoch: {training['best_epoch']};",
              f"best validation loss: {training['best_validation_loss']:.8f}.",
              f"Recorded elapsed time: {training['elapsed_seconds']/60:.2f} minutes.",
              "Float32 network operations run deterministically on CPU; the simulator and",
              "physical transform boundaries use float64. Test catalogs are excluded from",
              "gradient training and checkpoint selection.", "",
              f"Checkpoint: [{training['checkpoint']}]({training['checkpoint']}).",
              "[Run config and losses](results/training.json),",
              "[data manifest and hashes](results/training-data-manifest.json), and",
              "[regeneration instructions](TRAINING_RUN.md) preserve provenance.", "",
              "## Posterior gates", "",
              "| Gate | Status |", "|---|---|"]
    lines += [f"| {name} | {gate['status']} |" for name,gate in ledger["gates"].items()]
    if "posterior_evidence" in ledger and ledger["posterior_evidence"].get("path"):
        lines += ["", "Executed measurements, seeds and tolerances are in",
                  "[posterior-gates.json](results/posterior-gates.json)."]
    gp4 = ledger["gates"]["G-P4"]
    if gp4.get("parameters"):
        lines += ["", "| Parameter | 68% coverage | 95% coverage | Adjusted KS p | Status |",
                  "|---|---:|---:|---:|---|"]
        lines += [f"| {p['parameter']} | {p['coverage68']:.3f} | {p['coverage95']:.3f} | {p['bonferroni_adjusted_p']:.6g} | {p['status']} |"
                  for p in gp4["parameters"]]
    lines += ["", "The real-data smoke test remains unrun. Full Macquart sky positions and",
              "fluence columns must be transcribed before that test; no fields are fabricated.",
              "`pytest -m phase2a` writes the separate [Phase 2a ledger](results/phase2a_gates.json)",
              "and exits nonzero while acceptance is incomplete. Evidence aggregation preserves",
              "earlier executed preflight results; it does not claim a fresh complete test run.", "",
              "## Interpretation and independent work", "",
              "The quiet-validation observation is retired. The detected population is",
              "moderate-redshift dominated; the exchange-rate framing must reflect that regime.",
              "Comparison with the actual CHIME Catalog 1 DM histogram is a recorded,",
              "nonblocking Phase 2b/3 consistency check. See [DECISIONS.md](DECISIONS.md).", "",
              "Phase 1 remains independently incomplete: L2 and the pygedm validation battery",
              "remain open. The human/Colab diagnosis is in [PYGEDM_DIAGNOSIS.md](PYGEDM_DIAGNOSIS.md).",
              "Both phase ledgers must pass before Phase 3.", ""]
    (ROOT/"PHASE2A_STATUS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    update()
