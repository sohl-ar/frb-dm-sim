# Phase 2a: preflight checkpoint, incomplete

The inspected production L0 sampler consumes independent RNG uniforms.
The new T1 tests exercise its sampling moments, dependence on RNG order,
and rejection of overlapping seed ranges. Numerical results are generated
in [the separate Phase 2a ledger](results/phase2a_gates.json).

No neural network, training data, checkpoint or posterior exists. T2 is
blocked by the selection-print discrepancy under the newly approved
Schechter LF and fluence threshold. Their numerical sampling checks pass;
the supplied print comparison does not. T3 has not run. All posterior gates and the
real-data smoke test are unrun. See [DECISIONS.md](DECISIONS.md).

Run locally from the installed repository:

```sh
python -m pytest -m phase2a
```

The preflight tests can pass while this command deliberately exits nonzero:
Phase 2a acceptance is incomplete. It does not overwrite Phase 1 evidence.
No Colab runtime is required for this check. Both independent phase ledgers
must reach acceptance before Phase 3.

The submitted Colab evidence is archived under `results/colab/` with a
separate human-reported environment record. Table-1 verification is
closed by human sign-off and its redshift amendment has been rerun.
See [PYGEDM_DIAGNOSIS.md](PYGEDM_DIAGNOSIS.md) for human-executable Linux
diagnostics and a candidate isolated compatibility environment. No Linux
installation success or full Phase 1 completion is claimed.
