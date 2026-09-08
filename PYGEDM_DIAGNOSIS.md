# pygedm diagnosis package: Linux execution pending

The supplied Colab banner does not contain the compiler error. Therefore
the original failure cannot yet be attributed to missing system libraries,
Python 3.13, or a packaging bug. The human operates the Colab runtime.

Local inspection of the cached PyPI pygedm 3.3.0 source establishes:

- `setup.py` builds C++ extensions and links NE2001 with `-lf2c`.
  This build path does not invoke `numpy.distutils` or `f2py`. The presence
  of gfortran alone does not verify C++ headers or libf2c availability.
- `pygedm/yt2020.py` calls `integrate.simps` at module import. SciPy 1.14
  removed that function, creating a separate import-compatibility problem
  even if compilation succeeds. No halo model replacement is proposed.

Sources: [upstream installation requirements](https://github.com/FRBs/pygedm#installation),
[PyPI release](https://pypi.org/project/pygedm/3.3.0/), and
[SciPy removal notice](https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html#expired-deprecations).
The inspected 3.3.0 source hashes and matching lines are recorded in
`results/colab/pygedm-source-inspection.json`. This is source inspection,
not a fabricated verbose-build finding.

## First cell: capture the actual failure

After updating the repository to this checkpoint, execute:

```python
%cd /content/frb-dm-sim
!python scripts/colab_diagnose.py
```

The script inventories installed versions and tests imports in separate
processes, then runs a verbose `pip wheel` without installing it into the
main environment. Paste back the generated `diagnosis.json` and
`pygedm-build.log`. They persist under `results/colab/diagnosis-*/` within
the runtime; download or commit them before disconnecting.

## Candidate compatibility cell: isolated environment

This recipe has not been executed in Linux here. It addresses the known
SciPy API requirement and supplies the documented build prerequisites;
it is not a claim that it fixes the unobserved compiler failure. It leaves
the main Python packages unchanged. Run it after reviewing the diagnostic.

```bash
%%bash
set -euo pipefail
cd /content/frb-dm-sim
apt-get update -qq
apt-get install -y build-essential f2c libf2c2-dev
python -m venv /content/frb-build-tools
/content/frb-build-tools/bin/python -m pip install uv
/content/frb-build-tools/bin/uv venv --python 3.11 --seed /content/frb-mw
mkdir -p results/colab
/content/frb-mw/bin/python -m pip install -v -r requirements-mw-compat.txt 2>&1 | tee results/colab/mw-compat-install.log
/content/frb-mw/bin/python -m pip freeze > results/colab/mw-compat-freeze.txt
/content/frb-mw/bin/python -c 'import pygedm; print(pygedm.__file__)'
```

The main repository requires NumPy >=2 and SciPy >=1.14. Do not install
the repository into the compatibility environment or downgrade the main
environment to these pins. A future subprocess adapter or verified newer
pygedm release would be needed to connect environments. That integration
and the pulsar/pole battery are still open; an import does not pass G5.

## Verification cell: main packages and honest gate rerun

```python
%cd /content/frb-dm-sim
!python -m pip list | grep -E 'healpy|glass|pygedm'
!test ! -x /content/frb-mw/bin/python || /content/frb-mw/bin/python -m pip list | grep -E 'numpy|scipy|pygedm'
!FRBSIM_GATE_REPORT=results/colab/gates-followup.json python -m pytest -m gates
```

Absence from `pip list` is an installation finding. Presence is not proof
of import success; the diagnostic tests that separately. The full gate
command continues to exit nonzero while L2/G5 and full G4/G7 remain open.
