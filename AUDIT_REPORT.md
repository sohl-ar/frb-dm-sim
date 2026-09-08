# Repository verification and diagnostic audit

Evidence base: commit `87853d6782ec611a06379887b97988bbe7f2767f`; [request](docs/REPOSITORY_AUDIT_REQUEST.txt).
Computed records: [repository audit](results/repository-audit.json),
[50-trial replay](results/sbc-spotcheck-50.json), [diagnostics](results/f-diagnostics.json).
This report is generated from those JSON files.

## Part 1 — verification

| Check | Result | Finding and impact |
|---|---|---|
| Training and checkpoint | PASS | Consecutive epochs; best loss and checkpoint metadata agree. |
| Archived SBC | PASS | All KS statistics recompute exactly from saved ranks and jitter seed. |
| Gate ledger | PASS | Recorded posterior gate fields match the separate SBC report. G-P4 remains fail. |
| Code integrity | PASS | No src/ or tests/ differences since the pre-interruption commit. |
| Resume | FAIL / scoped | Optimization and validation ASTs match; state restoration is correct. Stopping fidelity and narrative provenance do not match. |
| Replay | PASS | Every authorized replay rank matches the original; coverage is consistent with sampling variation. |
| Environment | PASS / scoped | Inspected packages match the recorded lock. This does not reconstruct every historical install operation. |
| Colab notebook | FLAG | Claimed notebook absent from checkout and available Git history; no badge additions found. |
| README / Colab guide | FIXED | Replaced stale status and separated core execution from failed native installation. |
| simps search | PASS | No simps calls/imports in src/ or tests/; no simulator patch needed. |
| Initial Git state | PASS | main matched local origin/main; only untracked .opencode/ session artifacts. Live remote is checked during publication. |

Training recorded 19 epochs, with best epoch 13 and validation loss 2.845229963.

Best-checkpoint SHA256: `3260ef3eb126e81913f037971c734732672b126e58db711bf5e04d4817fcd802`. The model reports epoch
13; the last checkpoint reports epoch 19.
The original training, posterior and acceptance JSONs are preserved unchanged.

### Resume fidelity and provenance

The [reviewed resume source](docs/resume_train_reviewed.py.txt) is archived
byte-for-byte for inspection, not recommended for execution. SHA256:
`2f8669ed92554d8bf469f5332580dab8fad5c3e15fb325a4c8043c5a5cc5da92`.
It loads last.pt, optimizer state, training RNG and NumPy shuffle RNG. It
uses validation shards and validation-noise seed
71012; 71011 is the
training-noise seed. The request's validation-seed statement is incorrect.
The validation loop and optimization mathematics match the original code.
Best.pt is only written on a validation improvement.

The resume record says `from_epoch=18`.
The original process output also contains the completed preceding epoch;
the claim that the executor reran both final epochs is unsupported. The
resume docstring and note describing a partial-epoch replay are stale.
The original patience condition was satisfied at epoch
18, but the resume loop checks it
only after running the next epoch and stopped at 19.
This extra epoch did not improve the selected best checkpoint. Historical
best bytes before resumption are not separately archived, so that temporal
identity cannot be independently hash-compared; current checkpoint hashes,
epoch metadata, loss history and replay are mutually consistent.

Training JSON retains the original training-code Git SHA, not a resume
executor/source SHA. Periodic progress counts are stale at epoch end and
must not be interpreted as final catalog counts or proof of discarded work.
These provenance defects are flagged for human review; the original evidence
has not been rewritten to make the history appear cleaner.

**Additional provenance FLAG:** Git normalizes the original JSON files from
CRLF to LF. Their parsed contents match, but their byte hashes differ. The
checkpoint's raw manifest hash matches the original Windows file, not the
committed LF blob. A fresh checkout can therefore fail evaluate.py's manifest
hash guard despite identical data/configuration. Both hashes and semantic
equality are recorded in repository-audit.json. This does not invalidate the
current local measurements, but byte-portable reproduction is not verified.
No hash guard, original report or historical blob was changed to hide this.

### Authorized SBC spot-check

Only 50 existing validation catalogs were replayed, with
1000 posterior draws each. Seeds are recorded in JSON.
Ranks matched in 200/200
comparisons. This subset is a replay check, not independent new calibration.

| Parameter | 50-trial 68% | Original 68% | 50-trial 95% | Original 95% |
|---|---|---|---|---|
| f_d | 0.680 | 0.664 | 0.980 | 0.944 |
| F | 0.660 | 0.722 | 0.980 | 0.950 |
| host_median | 0.680 | 0.687 | 0.980 | 0.956 |
| host_sigma_ln | 0.740 | 0.665 | 0.940 | 0.942 |

Per-parameter Wilson intervals and subset-versus-full sampling standard
errors are in JSON. A fixed five-to-seven-point window is not a universal
sampling bound at this trial count. No new acceptance threshold was used.

## Part 2 — diagnostic findings

### H1: F-specific inefficient learning — NOT ESTABLISHED

Per-parameter loss histories and original posterior widths were not saved.
Plateau timing therefore cannot be reconstructed. The width proxy below
uses only the authorized replay; full-sample rank comparisons use existing
archived data. No full-scale inference was repeated.

| Parameter | Median posterior width / prior support | Width / matching prior 68% interval | Rank vs prior-CDF correlation |
|---|---|---|---|
| f_d | 0.6725 | 0.9889 | 0.999017 |
| F | 0.5861 | 1.0340 | 0.998767 |
| host_median | 0.6031 | 0.9917 | 0.998715 |
| host_sigma_ln | 0.6857 | 1.0084 | 0.999009 |

Widths use central-68% intervals in physical parameter units. The matched
prior interval is computed analytically, including log-uniform priors.
The support-width column is also reported as requested. Neither definition
is substituted into G-P6, which remains unimplemented and unrun.
All four predictions are largely prior-like: the issue is broader than F.
There is no reference posterior here to identify achievable information or
distinguish optimization failure, encoder limitations and data limitations.

### H2: true-F position — variation SUPPORTED; transform mechanism NOT ESTABLISHED

The actual Phase 2a F prior is [0.05, 1.0], log-uniform;
the request's lower bound belongs to the broader Phase 1 range. Requested
bins are approximate, not exact quartiles. Exact quartile thresholds and
their additional analysis are in the diagnostic JSON.

| True F group | Trials | 68% F coverage bounds | 95% F coverage bounds |
|---|---|---|---|
| F_lt_0.1 | 230 | 36.52–36.96% | 90.43–91.30% |
| F_0.1_to_0.5 | 556 | 100.00% | 100.00% |
| F_gt_0.5 | 214 | 38.32–39.25% | 86.92% |

The middle group is always covered by the broad central intervals, while
the tails are covered less often. This is consistent with prior-like
predictions. Conditioning on the true parameter destroys the usual
prior-averaged SBC coverage guarantee: even an exact prior posterior with
uninformative data excludes truths outside its central prior interval.
This pattern cannot by itself diagnose logit-gradient compression.

### H3: localization composition — proposed recovery NOT SUPPORTED

| Observed localized fraction | Trials | Parameter | 68% coverage bounds | 95% coverage bounds |
|---|---|---|---|---|
| localized_lt_0.15 | 300 | f_d | 65.67% | 93.00% |
| localized_lt_0.15 | 300 | F | 69.00–69.33% | 94.00% |
| localized_lt_0.15 | 300 | host_median | 68.33% | 94.33% |
| localized_lt_0.15 | 300 | host_sigma_ln | 65.33–66.00% | 94.67% |
| localized_0.15_to_0.35 | 473 | f_d | 67.65–67.86% | 94.71–95.14% |
| localized_0.15_to_0.35 | 473 | F | 74.21% | 95.98–96.19% |
| localized_0.15_to_0.35 | 473 | host_median | 67.65–67.86% | 96.19% |
| localized_0.15_to_0.35 | 473 | host_sigma_ln | 68.08–68.29% | 95.77–95.98% |
| localized_gt_0.35 | 227 | f_d | 64.76% | 95.59% |
| localized_gt_0.35 | 227 | F | 72.25–73.13% | 94.27–94.71% |
| localized_gt_0.35 | 227 | host_median | 70.93–71.37% | 96.04% |
| localized_gt_0.35 | 227 | host_sigma_ln | 64.76% | 90.31% |

Groups use the realized localized fraction, not the latent Beta draw.
Boundary values are assigned to the middle group. Trial counts, median N,
median true F and Wilson uncertainty bounds are in JSON. The localized-heavy
group does not recover nominal F central coverage; group intervals overlap.
The low-localization group is closest to nominal. These associations do not
establish an architectural cause or prove a physical need for localization.
Weak information alone should broaden a correct posterior without requiring
systematic prior-averaged miscalibration.

### Data limitation and interpretation

Archived ranks do not determine linearly interpolated posterior quantiles
at their two boundary gaps. Conditional results are exact **bounds** on
coverage, including every ambiguous boundary trial; they are not invented
point estimates. All original full-sample coverage values lie inside these
bounds. Full-sample contraction cannot be recovered without an additional
artifact or newly authorized inference.

The central finding is prior-like inference across parameters, with F's
overcoverage still a real gate failure. Parameter-only SBC can pass even
when an algorithm returns the prior and ignores data; see
[Modrak et al., section 3.4](https://arxiv.org/html/2211.02383v3#S3.SS4).
This is a known limitation of that diagnostic, not grounds to change it.
No claim of a measured exchange rate or a physical limitation is justified.

Recommended next step, requiring the next task's authorization: compare
conditioning sensitivity and informative-reference posteriors on existing
catalogs, record per-parameter loss/interval artifacts, then choose a
targeted training or architecture repair. Keep priors and tolerances fixed.
Implement the missing gates only in the next task; do not tune on this SBC
audit repeatedly or declare Phase 3 ready.

## Part 3 — documentation and preservation

README, PROJECT_STATUS, PHASE2A_STATUS, TRAINING_RUN, DECISIONS, CONVENTIONS,
COLAB_RUN and VERIFICATION_TODO now reflect the observed state. Historical
authorizations and original reports remain unchanged. docs/README explains
which stored texts are historical. The absent notebook is flagged, not
fabricated or advertised with a broken badge. The reviewed resume source
is archived outside ignored work/; .opencode/ remains excluded.

No files under src/ or tests/ were changed. No simps rename was needed in this repo;
the separate upstream pygedm issue remains documented in PYGEDM_DIAGNOSIS.
SciPy removed the deprecated alias in its
[release notes](https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html#expired-deprecations).

## Part 4 — Git synchronization

Audit code, diagnostic evidence and documentation are committed together.
The publication check records the source commit and live origin/main SHA
in results/repository-sync.json. The final documentation commit is also
verified against the remote before completion. work/ and .opencode/ are
not staged. No force push or history rewrite is used.
