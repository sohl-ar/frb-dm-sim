# Phase 2a status — Phase 2a incomplete; Phase 3 blocked

Current update: conditioned training completed. Both contraction sizes and
independent full SBC are now recorded; G-P6 and G-P4 remain failed. The followup
did not retrain or alter criteria. See CONDITIONING_FOLLOWUP.md, including its
floating-point boundary correction. f_d now passes SBC; F and host_sigma_ln
still fail. A matched localization diagnostic identifies sensitivity to a rare
standardized availability flag and absent large all-localized training examples.
The current host information ceiling remains unestablished. Do not relaunch
training or change gate criteria. G-P7's approved
soft scope is per-family coverage plus selection ignored; a true cross-family
experiment is deferred to Phase 2b in the ledger. The results below remain the
baseline evidence, not results from the repaired architecture.

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

Still required for Phase 2a: human review of the contraction/SBC failures, then
authorized next steps and the remaining gates/soft reports/smoke.
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

The authoritative saved ledger is [phase2a_gates.json](results/phase2a_gates.json).
Audit summaries preserve that evidence rather than rerunning preflight or acceptance.
