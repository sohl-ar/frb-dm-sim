# Phase 2a: preflight checkpoint, incomplete

The inspected production L0 sampler consumes independent RNG uniforms.
The new T1 tests exercise its sampling moments, dependence on RNG order,
and rejection of overlapping seed ranges. Numerical results are generated
in [the separate Phase 2a ledger](results/phase2a_gates.json).

The native Set Transformer and flow-matching network are implemented.
T2's print corrections are authorized and the z=0.2 evidence row is closed.
The selected training-data generator is implemented and has been exercised
on audit catalogs; seed, distribution, schema, replay and CPU speed
checks pass. The complete training, validation and test sets are generated
and hashed in `results/training-data-manifest.json`.
Generator audit results are in `results/generator-audit.json`. T3 passes;
per-event CDFs are recorded in `results/prior-predictive.json` and were
reported before training. `results/model-audit.json` records the initial
architecture checks and bit-identical short-run weights/samples. Full
training state and checkpoint provenance are recorded in `results/training.json`
when the training runner starts. See `TRAINING_RUN.md` for regeneration.
Statistical posterior gates and the real-data smoke test remain unrun.
See [DECISIONS.md](DECISIONS.md).

Run locally from the installed repository:

```sh
python -m pytest -m phase2a
```

The preflight tests can pass while this command deliberately exits nonzero:
Phase 2a acceptance is incomplete. It does not overwrite Phase 1 evidence.
No Colab runtime is required for this check. Both independent phase ledgers
must reach acceptance before Phase 3.

The interpretation error is resolved by human authorization: the detected
population is dominated by moderate redshifts, not z<0.5. The quiet-validation
observation is retired. Phase 3 framing must reflect that regime; the
CHIME Catalog 1 DM-histogram consistency check is recorded as future work.

The submitted Colab evidence is archived under `results/colab/` with a
separate human-reported environment record. Table-1 verification is
closed by human sign-off and its redshift amendment has been rerun.
See [PYGEDM_DIAGNOSIS.md](PYGEDM_DIAGNOSIS.md) for human-executable Linux
diagnostics and a candidate isolated compatibility environment. No Linux
installation success or full Phase 1 completion is claimed.
