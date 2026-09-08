# Execute the conditioning experiment once

**Execution update: this training attempt is complete. Do not relaunch these
commands.** The human subsequently authorized the remaining contraction size
and independent SBC despite the initial contraction failure. Their results,
unchanged gate statuses and next-step assessment are in CONDITIONING_FOLLOWUP.md.
The instructions below are retained as the historical execution handoff.

The architect implemented and quick-tested the statistics bypass and full gate
infrastructure. **No new training, full SBC or full gate suite has run.**
The original checkpoint and results remain the baseline. Run this handoff in
the existing Windows checkout and Python environment; no Colab runtime is
needed for this CPU experiment. Phase 1's Linux/pygedm work stays separate.

The authoritative copy is this tracked file. `work/HANDOFF.md` is an identical
local convenience copy. Do not regenerate catalogs or refit normalization:
the saved training, validation and test shards and fitted training-only scaler
are already present. If any required artifact is absent, STOP and report it.

## Rules

- Do NOT modify any file in `src/` or `tests/`.
- Do NOT change any tolerance, physics anchor, gate criterion or launch config.
- Do NOT train more than once, resume, retry, delete attempt locks, or switch
  run directories to circumvent a failed/interrupted attempt.
- Do NOT adjust anything to make a gate pass. Do not diagnose or fix a failure.
- Every reported number must come from executed code or its saved artifacts.
- If ANY command/check fails, report the error verbatim and STOP. Do not
  continue to later stages. Report remaining gates as not run.
- Soft G-P3/G-P7 findings are reported diagnostic outcomes, not numerical hard
  failures. An execution exception in a soft diagnostic still stops the run.
- No Phase 3 work and no scientific interpretation of smoke-test posteriors.

## 0. Enter the existing environment and check readiness

PowerShell, from the current host:

```powershell
Set-Location 'C:\Users\sachi\Documents\Codex\2026-09-06\below-is-the-complete-final-spec\outputs\frb-dm-sim'
$py = '..\..\work\.venv\Scripts\python.exe'
$env:MPLCONFIGDIR = "$PWD\work\matplotlib"
& $py scripts/check_conditioning_ready.py
if ($LASTEXITCODE -ne 0) { throw 'STOP: readiness failed' }
```

Readiness verifies every saved shard, normalizer provenance, pinned configuration,
baseline checkpoint, and absence of a prior training attempt. It does no model
fitting. Dependencies including `sbi==0.27.0` are installed on this host. The
installed package snapshot is `requirements-conditioning-lock.txt`; this is
environment provenance, not an instruction to reinstall packages now.

## 1. Launch training exactly once

```powershell
& $py -u -m frbsbi.train_conditioned launch 2>&1 | Tee-Object -FilePath 'results/conditioning-v1/training-console.log'
if ($LASTEXITCODE -ne 0) { throw 'STOP: training failed; do not retry' }
```

The module reads every argument from the committed
`results/conditioning-v1/launch.json`: original optimizer/hyperparameters,
seeds, batch size, epoch ceiling and early-stopping settings, with the bypass
enabled. It writes an exclusive attempt lock and archives baseline reports.
It does not resume the old run. The selected checkpoint is
`results/conditioning-v1/checkpoints/phase2a-best.pt`.

When using agent command tools, let this single command return a running
process/session ID, then poll that same process. Do not relaunch it to monitor
progress. A shell/tool timeout is not evidence that a process stopped: inspect
the original process and saved status before doing anything else.

## 2. Monitor the same run

From a separate terminal using the same working directory and `$py`:

```powershell
& $py scripts/report_conditioning_execution.py --section training
if ($LASTEXITCODE -ne 0) { throw 'STOP: status read failed' }
```

Read `results/conditioning-v1/training.json`. After every completed epoch,
report epoch number, training loss, validation loss, epoch seconds, best epoch
and best validation loss. While an epoch is running, report recorded optimizer
steps/catalogs processed if asked. Do not sample posteriors during training.

The user's runtime estimate is roughly 2–4 hours; this architecture has not
been benchmarked by full training. The configured ceiling is 20 epochs with
patience 5 after the original minimum-epoch policy. The new bypass should
permit improvement below the old approximately 2.85 loss plateau. That number
is a diagnostic expectation, not a new numerical acceptance threshold or a
reason to alter stopping settings. Report a repeated plateau or failure to
converge with the actual trace; do not retry or adjust the model. Nonfinite
losses/gradients or any exception are immediate STOP conditions.

Only move on when the command succeeds and status is
`training_complete_unvalidated`. Training completion alone is not acceptance.

## 3. Contraction, then SBC

```powershell
& $py -u -m frbsbi.acceptance validation 2>&1 | Tee-Object -FilePath 'results/conditioning-v1/validation-console.log'
if ($LASTEXITCODE -ne 0) { throw 'STOP: validation failed; report verbatim' }
```

This runs G-P6 first, then G-P4 only if contraction passes. It records physical
central-68% posterior/prior interval-width ratios, not widths divided by full
prior support. **Every parameter** must have median ratio strictly below 0.7
at N=16 and below 0.5 at N=256. Ratios near one mean the conditioning problem
persists; report the ratios and stop, without diagnosis.

SBC uses the prescribed 1,000 held-out validation catalogs and 1,000 posterior
draws each. For every parameter, report central-68% and central-95% coverage,
KS statistic and Bonferroni-adjusted p. Coverage must satisfy the unchanged
±2 percentage-point bands and the adjusted KS p criterion. Undercoverage
indicates overconfidence; overcoverage indicates underconfidence. Report
which criterion failed without adjusting it. Validation data were also used
for loss-based checkpoint selection, as in the approved original protocol.

Validation success does not set full acceptance true. Raw posterior artifacts
are saved so later checks and audits do not have to reconstruct intervals.

## 4. Full gate suite

```powershell
& $py -u -m pytest -m phase2a -q -s 2>&1 | Tee-Object -FilePath 'results/conditioning-v1/pytest-console.log'
if ($LASTEXITCODE -ne 0) { throw 'STOP: full gate suite failed or incomplete' }
```

The default pytest configuration stops on the first failing test. Preflight
tests run first, followed by G-P1–G-P8 and the smoke test. Previously successful
G-P6/G-P4 results are reused only when their checkpoint, dataset, implementation
and artifact hashes match. Do not edit source, dependencies or configuration
between stages; mismatches stop instead of silently rerunning.

G-P5 evaluates the TARP ECP curve at 68% and 95% using the prescribed held-out
catalogs and pinned sbi implementation. G-P8 requires the median paired
unlocalized/localized f_d width ratio strictly above 1.5 at N=8,32,128.
G-P3 and G-P7 are soft and reported-only.

**G-P7 human authorization:** report per-family coverage and the
selection-ignored probe. A true cross-trained wrong-family experiment is
deferred to Phase 2b because this model trained on both families. This approval
and deferral are stored in the ledger, including if earlier gates stop the run.

## 5. Report and preserve the results

This command is read-only and may also be used after a failure to gather its
saved evidence; do not launch any further numerical checks after a failure:

```powershell
& $py scripts/report_conditioning_execution.py --section all
```

Report these items in a compact table plus the verbatim first failure:

- Training: status, epoch/loss trace, best epoch/loss, stop reason, elapsed
  time, checkpoint SHA256.
- G-P1–G-P8 and smoke: status, hard/soft classification, measured values,
  tolerance, execution time; explicitly list anything not run.
- G-P6: ratios for all four parameters at both N values that executed.
- G-P4: both coverages and adjusted KS p per parameter.
- G-P5: ECP at both nominal levels and absolute deviations.
- G-P8: median paired f_d width ratio at each executed N.
- G-P3: calibration metrics by N, including extrapolation.
- G-P7: per-family and selection-ignored coverage/bias and coverage deltas;
  repeat the authorized cross-family deferral.
- Smoke: combined six-event posterior means and central-68% widths, labeled
  pipeline-only and not science.

Authoritative files are `results/conditioning-v1/training.json`,
`results/conditioning-v1/phase2a_gates.json`, and the promoted
`results/phase2a_gates.json`. Posterior NPZ paths, hashes and solver settings
are in the run ledger's `artifacts` mapping. Baseline evidence is archived in
`results/conditioning-v1/baseline-evidence/` and the baseline ledger copy.
Never hand-edit JSON or remove failed evidence. Committing/pushing generated
results is allowed; changing implementation to repair them is not.

The run is accepted only when full pytest exits zero and the run ledger says
`acceptance_complete: true`. Phase 1 and Phase 3 status do not change merely
because Phase 2a passes. Stop after reporting; the next architect session
interprets results and decides subsequent work.
