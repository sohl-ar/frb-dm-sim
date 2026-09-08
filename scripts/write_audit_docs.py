"""Render current documentation from saved audit JSON; no model execution."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))


def write(name,text):
    (ROOT/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def bounds(value):
    a,b=value['coverage_bounds']
    return f'{100*a:.2f}%' if a==b else f'{100*a:.2f}–{100*b:.2f}%'


def main():
    a=read('repository-audit.json'); d=read('f-diagnostics.json'); s=read('sbc-spotcheck-50.json')
    t=read('training.json'); p=read('posterior-gates.json'); l=read('phase2a_gates.json')
    params=p['gates']['G-P4']['parameters']; names=[x['parameter'] for x in params]
    sbc=table(['Parameter','68% coverage','95% coverage','Adjusted KS p','Recorded result'],
        [[x['parameter'],f"{x['coverage68']:.3f}",f"{x['coverage95']:.3f}",f"{x['bonferroni_adjusted_p']:.6g}",x['status']] for x in params])
    width=table(['Parameter','Median posterior width / prior support','Width / matching prior 68% interval','Rank vs prior-CDF correlation'],
        [[name,f"{s['posterior_widths'][name]['ratio_to_prior_support']:.4f}",
          f"{s['posterior_widths'][name]['ratio_to_matched_prior_central68_width']:.4f}",
          f"{d['H1']['archived_1000_rank_prior_comparison'][name]['prior_CDF_vs_posterior_rank_correlation']:.6f}"] for name in names])
    spot=table(['Parameter','50-trial 68%','Original 68%','50-trial 95%','Original 95%'],
        [[x['parameter'],f"{x['0.68']['coverage']:.3f}",f"{x['0.68']['original_1000']:.3f}",
          f"{x['0.95']['coverage']:.3f}",f"{x['0.95']['original_1000']:.3f}"] for x in s['parameters']])
    fgroups=table(['True F group','Trials','68% F coverage bounds','95% F coverage bounds'],
        [[key,v['n'],bounds(v['parameters']['F']['0.68']),bounds(v['parameters']['F']['0.95'])]
         for key,v in d['H2']['requested_groups'].items()])
    lgroups=table(['Observed localized fraction','Trials','Parameter','68% coverage bounds','95% coverage bounds'],
        [[key,v['n'],name,bounds(v['parameters'][name]['0.68']),bounds(v['parameters'][name]['0.95'])]
         for key,v in d['H3']['groups'].items() for name in names])
    epochs=table(['Epoch','Training loss','Validation loss','Epoch seconds'],
        [[e['epoch'],f"{e['training_loss']:.9f}",f"{e['validation_loss']:.9f}",f"{e['seconds']:.3f}"] for e in t['epochs']])
    outcome=f"Training recorded {len(t['epochs'])} epochs, with best epoch {t['best_epoch']} and validation loss {t['best_validation_loss']:.9f}."
    gates=table(['Gate','Status'],[[key,v['status']] for key,v in p['gates'].items()])
    summary=f"""# Project status — Phase 2a incomplete; Phase 3 blocked

{outcome} The saved best checkpoint and archived SBC results pass the
integrity audit. G-P1 and G-P2 pass. Three parameters pass the recorded
marginal SBC criteria; F fails its central-68% coverage criterion.

{sbc}

**The more serious finding is weak contraction across all four parameters.**
The authorized replay's central-68% posterior widths are approximately the
matching prior widths. Archived ranks also closely track prior positions.
Marginal calibration passes therefore do not establish useful conditional
inference. F's architectural versus physical cause remains unidentified.

The resume script restored model, optimizer and RNG states correctly, but
ran an extra epoch after patience was already exhausted. Its stopping
fidelity is FAIL; this does not change the evaluated best checkpoint.
See [AUDIT_REPORT.md](AUDIT_REPORT.md) for evidence and limitations.

Still required for Phase 2a: resolve G-P4 and demonstrate informative
conditional learning; implement and run G-P5/TARP, G-P6/contraction and
G-P8/information contrast; report G-P3 and G-P7; run the full-column
Macquart smoke test. Missing gates were not implemented in this audit.
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
"""
    write('PROJECT_STATUS.md',summary)
    write('PHASE2A_STATUS.md',summary.replace('# Project status','# Phase 2a status')+
        '\nThe authoritative saved ledger is [phase2a_gates.json](results/phase2a_gates.json).\n'+
        'Audit summaries preserve that evidence rather than rerunning preflight or acceptance.\n')
    report=f"""# Repository verification and diagnostic audit

Evidence base: commit `{a['audit_git_sha']}`; [request](docs/REPOSITORY_AUDIT_REQUEST.txt).
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

{outcome}

Best-checkpoint SHA256: `{a['checkpoint_sha256']}`. The model reports epoch
{a['best_epoch']}; the last checkpoint reports epoch {a['last_epoch']}.
The original training, posterior and acceptance JSONs are preserved unchanged.

### Resume fidelity and provenance

The [reviewed resume source](docs/resume_train_reviewed.py.txt) is archived
byte-for-byte for inspection, not recommended for execution. SHA256:
`{a['resume']['sha256']}`.
It loads last.pt, optimizer state, training RNG and NumPy shuffle RNG. It
uses validation shards and validation-noise seed
{a['resume']['validation_noise_seed']}; {a['resume']['training_noise_seed']} is the
training-noise seed. The request's validation-seed statement is incorrect.
The validation loop and optimization mathematics match the original code.
Best.pt is only written on a validation improvement.

The resume record says `from_epoch={a['resume']['recorded']['from_epoch']}`.
The original process output also contains the completed preceding epoch;
the claim that the executor reran both final epochs is unsupported. The
resume docstring and note describing a partial-epoch replay are stale.
The original patience condition was satisfied at epoch
{a['resume']['expected_patience_stop_epoch']}, but the resume loop checks it
only after running the next epoch and stopped at {a['resume']['actual_stop_epoch']}.
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

Only {s['trials']} existing validation catalogs were replayed, with
{s['samples_per_trial']} posterior draws each. Seeds are recorded in JSON.
Ranks matched in {s['archived_rank_matches']}/{s['archived_rank_comparisons']}
comparisons. This subset is a replay check, not independent new calibration.

{spot}

Per-parameter Wilson intervals and subset-versus-full sampling standard
errors are in JSON. A fixed five-to-seven-point window is not a universal
sampling bound at this trial count. No new acceptance threshold was used.

## Part 2 — diagnostic findings

### H1: F-specific inefficient learning — NOT ESTABLISHED

Per-parameter loss histories and original posterior widths were not saved.
Plateau timing therefore cannot be reconstructed. The width proxy below
uses only the authorized replay; full-sample rank comparisons use existing
archived data. No full-scale inference was repeated.

{width}

Widths use central-68% intervals in physical parameter units. The matched
prior interval is computed analytically, including log-uniform priors.
The support-width column is also reported as requested. Neither definition
is substituted into G-P6, which remains unimplemented and unrun.
All four predictions are largely prior-like: the issue is broader than F.
There is no reference posterior here to identify achievable information or
distinguish optimization failure, encoder limitations and data limitations.

### H2: true-F position — variation SUPPORTED; transform mechanism NOT ESTABLISHED

The actual Phase 2a F prior is {d['H2']['true_F_prior']}, log-uniform;
the request's lower bound belongs to the broader Phase 1 range. Requested
bins are approximate, not exact quartiles. Exact quartile thresholds and
their additional analysis are in the diagnostic JSON.

{fgroups}

The middle group is always covered by the broad central intervals, while
the tails are covered less often. This is consistent with prior-like
predictions. Conditioning on the true parameter destroys the usual
prior-averaged SBC coverage guarantee: even an exact prior posterior with
uninformative data excludes truths outside its central prior interval.
This pattern cannot by itself diagnose logit-gradient compression.

### H3: localization composition — proposed recovery NOT SUPPORTED

{lgroups}

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
"""
    write('AUDIT_REPORT.md',report)
    write('TRAINING_RUN.md',f"""# Training record and reproduction boundaries

{outcome}
Current state: training finished, G-P4 failed, Phase 2a incomplete.
The best model was selected by validation loss, not SBC outcomes.

{epochs}

Recorded total elapsed time: {t['elapsed_seconds']:.3f} seconds. This is the
stored run/resume clock, not a fresh benchmark or a GPU-time claim.

## Resume audit

The original self-contained process completed epoch {a['resume']['recorded']['from_epoch']}.
The executor's resume then completed epoch {a['resume']['actual_stop_epoch']}.
The request's two-epoch-resume narrative conflicts with the checkpoint
record and original process output. All model, optimizer and RNG restoration
calls are correct, with validation noise seed {a['resume']['validation_noise_seed']}.
The validation loop and update mathematics match the original AST.

**Resume fidelity: FAIL for the stopping boundary.** Patience was already
exhausted at epoch {a['resume']['expected_patience_stop_epoch']}; resumption ran
one extra epoch before checking it. The best epoch {t['best_epoch']} did not
change. The resume note and periodic catalog counter are stale. Do not
execute work/resume_train.py as an approved recovery path until that finding
is resolved. The [reviewed source](docs/resume_train_reviewed.py.txt) is an
immutable forensic copy, not an executable recommendation.

Best checkpoint SHA256: `{t['checkpoint_sha256']}`.
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
""")
    decision=f"""## Repository audit and training outcome (generated from execution JSON)

{outcome}
G-P1 and G-P2 pass; G-P4 fails on F central-68% overcoverage. Three other
parameters pass the recorded marginal SBC criteria, which is not proof of
informative conditional learning.

{sbc}

H1's earlier-plateau claim cannot be assessed because per-parameter loss
histories were not recorded. Width ratios from the authorized replay and
archived rank/prior correlations show prior-like predictions across all
parameters. H2 shows true-F-position dependence but does not establish a
logit-gradient defect. H3 does not support recovery in localized-heavy
catalogs. Architectural versus physical cause is unresolved; no physics
finding, model repair, new training run or tolerance change is authorized
by this diagnosis. Details and conditional bounds are in AUDIT_REPORT.md.

The resume restored states and reproduced the update/validation mathematics,
but continued after the configured patience boundary. This is flagged for
human decision; the evaluated best model remains unchanged by the extra
epoch according to the saved metadata and loss comparison. Source and
provenance defects are archived in repository-audit.json, not erased from
the original record. The claimed Colab notebook/badge additions are absent
from this repository; no decision to use them is recorded as verified.

Git EOL normalization changes original JSON byte hashes while preserving
their values. The checkpoint's manifest hash uses the Windows CRLF bytes;
a fresh LF checkout fails that guard. This portability defect is flagged
for human-approved repair without rewriting the scientific evidence.

The quiet-validation observation remains retired. The detected population
is dominated by z>0.5; the eventual exchange-rate framing must be moderate-z.
The future actual CHIME Catalog 1 DM-histogram comparison remains unrun.
Anchor prints may be revised by explicit human sign-off and evidence; gate
tolerances, physics anchors and coverage criteria cannot be revised that way.
"""
    dec=(ROOT/'DECISIONS.md').read_text(encoding='utf-8').split('<!-- GENERATED_REPOSITORY_AUDIT -->')[0].rstrip()
    write('DECISIONS.md',dec+'\n\n<!-- GENERATED_REPOSITORY_AUDIT -->\n\n'+decision)
    write('README.md',f"""# FRB dispersion-measure simulation and amortized inference

This research repository models FRB catalogs and infers the diffuse-ionized
baryon fraction, cosmic-DM fluctuation parameter and host-DM distribution.
Background cosmology is pinned. L0 uses the published asymmetric cosmic-DM
distribution; a native Set Transformer conditions a flow-matching posterior
on localized and unlocalized observations.

**Phase 2a training is complete; acceptance is not.** {outcome}
G-P1/G-P2 pass. Three parameters pass the recorded marginal SBC checks;
F is underconfident at central-68% coverage. The audit additionally finds
largely prior-like predictions across all four parameters, so these passes
must not be advertised as successful informative inference. Phase 3 is blocked.

{sbc}

Start with [PROJECT_STATUS.md](PROJECT_STATUS.md) and the
[verification and diagnostic report](AUDIT_REPORT.md). Phase 1 remains
independently incomplete: L2 is unbuilt and pygedm validation is open.

## Quick start: inspect the saved checkpoint

Python with the dependencies in pyproject.toml is required; the tested
environment is recorded in requirements-training-lock.txt. From a clone:

```sh
python -m pip install -e '.[inference]'
python -c "import json; print(json.load(open('results/training.json'))['status'])"
```

No retraining is needed to inspect committed evidence. Checkpoints, seeds,
config hashes and reports are tracked. Large training shards live under
ignored work/training-data and are reproducible from the tracked manifest.
See [TRAINING_RUN.md](TRAINING_RUN.md) before authorizing a new run; training
and evaluation commands can overwrite run artifacts.

## Verification architecture

T1–T3 gate the RNG path, selected generator and prior predictive. Physics
gates verify mean-DM formulas, units, asymmetric sampling and replication
fixtures. Posterior gates separately check invariance, replay, calibration,
contraction and information content. Failures trigger STOP; tolerances and
physical anchors require explicit human authorization to change.

The separate saved ledgers are [Phase 1](results/gates.json) and
[Phase 2a](results/phase2a_gates.json). `pytest -m gates` remains nonzero for
incomplete L2 acceptance. `pytest -m phase2a` reruns preflight checks and
merges existing posterior evidence; it remains nonzero and does not
implement or execute the missing posterior gates. L0 smoke inputs are
engineering fixtures, not empirical survey calibrations.

## Colab and remaining work

Colab can execute the same repository without keeping scientific logic in
cells. Follow [COLAB_RUN.md](COLAB_RUN.md). The claimed repository notebook
`notebooks/frb_dm_sim_colab.ipynb` is absent; no working badge is claimed.
The human-run native-dependency diagnosis is [PYGEDM_DIAGNOSIS.md](PYGEDM_DIAGNOSIS.md).

G-P5/TARP, G-P6, G-P8, the G-P3/G-P7 diagnostics and real-data smoke remain
unimplemented/unrun. G-P4 and weak contraction must be resolved. L2/GLASS
and the Milky Way model remain a separate Phase 1 task. The actual CHIME
Catalog 1 DM-histogram comparison is deferred to Phase 2b/3.

See [CONVENTIONS.md](CONVENTIONS.md), [DECISIONS.md](DECISIONS.md),
[VERIFICATION_TODO.md](VERIFICATION_TODO.md) and the
[historical document index](docs/README.md). No physics or gate code was
changed by the repository audit.
""")


if __name__=='__main__':
    main()
