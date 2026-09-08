# Project status — Phase 2a incomplete; Phase 3 blocked

Current update: the observable statistics bypass, isolated launch, full
posterior gate runner and execution handoff are implemented. Quick engineering
checks pass; no new training/full SBC/full suite was run. Follow HANDOFF.md.
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

Still required for Phase 2a: execute the prepared training run and demonstrate
calibrated, informative conditional learning through the implemented full
gate suite, soft reports and full-column Macquart smoke test.
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
