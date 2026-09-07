# Run this repository on Google Colab

Colab is only the Linux execution host. The simulator remains a normal Git
repository with a `src/` package, immutable tolerances, pytest gates, and Git
provenance.

Upload `frb-dm-sim.bundle` to the Colab session, then run these commands in a
single code cell:

```python
!git clone /content/frb-dm-sim.bundle /content/frb-dm-sim
%cd /content/frb-dm-sim
!python -m pip install --upgrade pip
!python -m pip install -e ".[l2,mw]"
!python -m pytest -m gates
```

The final command is intentionally strict. It must remain nonzero until all
L2 work is implemented and G1-G9 pass. Installing GLASS, healpy, CAMB, and
pygedm only removes the native-Windows dependency blockers; it does not turn
the existing L0 checkpoint into a completed L2 simulator.

For an environment and L0 smoke check before continuing L2 work:

```python
!python - <<'PY'
import importlib.util
for name in ("numpy", "scipy", "astropy", "pytest", "pygedm", "healpy", "glass", "camb"):
    print(f"{name}: {importlib.util.find_spec(name) is not None}")
PY
!python -m pytest --level l0
!python -m frbsim.demo --smoke
```

Download `results/gates.json` after each strict run. The file records measured
values, fixed tolerances, gate status, and the repository commit SHA.
