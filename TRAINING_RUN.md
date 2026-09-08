# Training record and reproduction boundaries

Training recorded 19 epochs, with best epoch 13 and validation loss 2.845229963.
Current state: training finished, G-P4 failed, Phase 2a incomplete.
The best model was selected by validation loss, not SBC outcomes.

| Epoch | Training loss | Validation loss | Epoch seconds |
|---|---|---|---|
| 1 | 3.309190667 | 3.088324755 | 180.033 |
| 2 | 3.011342111 | 2.950763713 | 233.151 |
| 3 | 2.913475012 | 2.891749748 | 262.762 |
| 4 | 2.913898163 | 2.885452676 | 258.625 |
| 5 | 2.903029635 | 2.864168986 | 163.312 |
| 6 | 2.893877203 | 2.871249992 | 153.945 |
| 7 | 2.867099206 | 2.861281324 | 257.344 |
| 8 | 2.889500118 | 2.871574783 | 199.934 |
| 9 | 2.867195829 | 2.856779402 | 237.047 |
| 10 | 2.885140039 | 2.853527340 | 253.905 |
| 11 | 2.877008276 | 2.863139700 | 260.686 |
| 12 | 2.876914220 | 2.874837920 | 265.752 |
| 13 | 2.874600873 | 2.845229963 | 254.517 |
| 14 | 2.866415242 | 2.847140633 | 260.597 |
| 15 | 2.870579090 | 2.854840151 | 253.153 |
| 16 | 2.874182378 | 2.853632056 | 262.588 |
| 17 | 2.872846725 | 2.848340042 | 258.801 |
| 18 | 2.854432290 | 2.851925226 | 265.913 |
| 19 | 2.854029966 | 2.849061755 | 112.770 |

Recorded total elapsed time: 4395.197 seconds. This is the
stored run/resume clock, not a fresh benchmark or a GPU-time claim.

## Resume audit

The original self-contained process completed epoch 18.
The executor's resume then completed epoch 19.
The request's two-epoch-resume narrative conflicts with the checkpoint
record and original process output. All model, optimizer and RNG restoration
calls are correct, with validation noise seed 71012.
The validation loop and update mathematics match the original AST.

**Resume fidelity: FAIL for the stopping boundary.** Patience was already
exhausted at epoch 18; resumption ran
one extra epoch before checking it. The best epoch 13 did not
change. The resume note and periodic catalog counter are stale. Do not
execute work/resume_train.py as an approved recovery path until that finding
is resolved. The [reviewed source](docs/resume_train_reviewed.py.txt) is an
immutable forensic copy, not an executable recommendation.

Best checkpoint SHA256: `3260ef3eb126e81913f037971c734732672b126e58db711bf5e04d4817fcd802`.
Original code/config provenance remains embedded in the checkpoint. The
resume source hash and deviations are recorded in results/repository-audit.json;
the original training JSON was not silently edited.

## Reproduction, only when a new run is authorized

```sh
python -m pip install -e '.[inference]'
python -m frbsbi.prior_predictive
python scripts/model_audit.py
python -m frbsbi.data
python -m frbsbi.train
```

These commands create new data and overwrite run artifacts. They were not
run in this audit. Preserve the current evidence first and use an isolated
checkout/output location for a newly authorized experiment. A fresh original
train.py run stops at its own patience boundary; it does not intentionally
reproduce the executor's extra epoch.

The training library versions are recorded in requirements-training-lock.txt.
It is a Windows environment snapshot containing a local editable path; use
its version pins on a compatible Python host and install this checkout with
`pip install -e '.[inference]'`, rather than blindly installing that path on Linux.
CPU determinism is scoped to the recorded environment and configuration.

The audit found that Git newline normalization changes the original JSON
byte hashes. In particular, a fresh checkout's LF manifest does not match
the checkpoint's hash of the original CRLF file, although parsed JSON is
identical. Raw-hash reproduction across a fresh checkout is blocked until
the human approves an explicit provenance repair. Do not bypass the guard
or edit checkpoint hashes silently; both forms are recorded in the audit.

Shards under ignored work/training-data are local, regenerable artifacts;
the tracked manifest contains their hashes. Checkpoints and original JSONs
are tracked. Test data were not used for gradients or checkpoint selection;
validation data were used for loss selection and the specified SBC split.

## Inspect existing evidence without rerunning training

Read PROJECT_STATUS.md, AUDIT_REPORT.md and the saved results JSONs.
`python scripts/write_audit_docs.py` regenerates documents from saved JSON only.
`python scripts/diagnose_saved_sbc.py` recomputes the prior-rank comparison
from existing validation shards without running the model.
The archived repository_audit.py contains the authorized replay and refuses
to execute again while its replay report exists. Do not remove that guard
to repeat inference without new authorization.

The original `python -m frbsbi.evaluate` executes the full implemented SBC
subset and is not an audit inspection command. `pytest -m phase2a` reruns
preflight tests, merges saved posterior evidence, and exits nonzero while
acceptance is incomplete; it does not execute missing posterior gates.
