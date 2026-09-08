# Project status — Phase 2a incomplete; Phase 3 blocked

Latest code-only update: expanded composition and raw binary availability
encoding are implemented for a future conditioning-v2 run. New data generation,
normalization fitting, training and validation are explicitly deferred; tests
for this change are written but unrun. See COMPOSITION_FIX.md. The measured
results below still belong to conditioning-v1, not the new code preparation.

Current update: the conditioned training run completed. Both prescribed
contraction sizes and independent full SBC have executed under explicit human
continuation authorization. G-P6 and G-P4 remain failed; no retraining or gate
adjustment occurred in the followup. See CONDITIONING_FOLLOWUP.md for the
measured results and the floating-point boundary finding in the existing SBC
comparator. That comparison bug is now fixed with exact trial-count arithmetic:
f_d passes SBC; F and host_sigma_ln still fail. Stratification and a matched
localization probe reveal a rare-flag/training-support problem in the model.
This does not establish a physical limit on host inference. The remaining
full gate suite has not run. Retraining is deferred for the current session;
no gate-criterion change is authorized.
The G-P7 per-family/selection-ignored scope is human-approved, with a true
cross-family experiment deferred to Phase 2b. Details and generated checks are
in CONDITIONING_FIX.md, GATE_IMPLEMENTATION.md and results/conditioning-v1/.

The following numbers describe the preserved baseline run.

Training recorded 19 epochs, with best epoch 13 and validation loss 2.845229963. The saved best checkpoint and archived SBC results pass the
integrity audit. G-P1 and G-P2 pass. Three parameters pass the recorded
marginal SBC criteria; F fails its central-68% coverage criterion.

| Parameter | 68% coverage | 95% coverage | Adjusted KS p | Recorded result |
|---|---|---|---|---|
| f_d | 0.664 | 0.944 | 1 | pass |
| F | 0.722 | 0.950 | 0.77273 | fail |
| host_median | 0.687 | 0.956 | 0.236142 | pass |
| host_sigma_ln | 0.665 | 0.942 | 1 | pass |

**The more serious finding is weak contraction across all four parameters.**
The authorized replay's central-68% posterior widths are approximately the
matching prior widths. Archived ranks also closely track prior positions.
Marginal calibration passes therefore do not establish useful conditional
inference. F's architectural versus physical cause remains unidentified.

The resume script restored model, optimizer and RNG states correctly, but
ran an extra epoch after patience was already exhausted. Its stopping
fidelity is FAIL; this does not change the evaluated best checkpoint.
See [AUDIT_REPORT.md](AUDIT_REPORT.md) for evidence and limitations.

Still required for Phase 2a: human review of the recorded contraction/SBC
failures, then authorized next steps and the remaining gates/soft reports/smoke.
The original gate ledger stays failed/incomplete.

Phase 1 remains independently incomplete: pygedm/G5 and L2/GLASS/G3 are
open, with G4/G7 partial. Colab is the human-operated Linux host; its
successful native installation is unverified. The claimed repository
notebook is absent and cannot be certified.

The detected population is moderate-redshift dominated. Phase 3 framing
must reflect that regime; the CHIME Catalog 1 DM-histogram comparison
remains a future selection consistency check. Both phase ledgers must pass
before Phase 3. No retraining, new simulations, full SBC rerun, physics
changes, test changes or tolerance changes were performed in this audit.

Next: obtain human sign-off on the resume/provenance findings and approve
a focused conditioning/information audit against a reference posterior on
existing catalogs before choosing a model repair. Recover the claimed
notebook and any missing full-SBC interval/sample artifact if they exist.
See [VERIFICATION_TODO.md](VERIFICATION_TODO.md) and [TRAINING_RUN.md](TRAINING_RUN.md).
