"""Human-run Linux diagnostic: record packages, imports and a verbose wheel build.

Builds in isolation without installing pygedm or changing the main packages.
The resulting log is necessary evidence; no root cause is guessed from pip's
summary banner. Not a physics gate and not a substitute Milky Way model.
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def main():
    if sys.platform != "linux":
        raise SystemExit("Run this diagnostic in the human's Linux/Colab runtime")
    root = Path(__file__).resolve().parents[1]
    stamp = str(time.time_ns())
    out = root / "results" / "colab" / f"diagnosis-{stamp}"
    out.mkdir(parents=True)
    wheel_dir = root / "work" / f"pygedm-wheels-{stamp}"
    wheel_dir.mkdir(parents=True)
    packages = subprocess.run([sys.executable,"-m","pip","list","--format=json"],capture_output=True,text=True)
    (out/"pip-list.json").write_text(packages.stdout,encoding="utf-8")
    imports = {}
    for name in ("numpy","scipy","healpy","glass","pygedm"):
        probe = subprocess.run([sys.executable,"-c",f"import {name}; print({name}.__file__)"],capture_output=True,text=True)
        imports[name] = {"returncode":probe.returncode,"stdout":probe.stdout,"stderr":probe.stderr}
    command = [sys.executable,"-m","pip","wheel","-v","--no-cache-dir","--no-deps",
               "--wheel-dir",str(wheel_dir),"pygedm==3.3.0"]
    with (out/"pygedm-build.log").open("w",encoding="utf-8") as log:
        build = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    logtext = (out/"pygedm-build.log").read_text(encoding="utf-8",errors="replace")
    report = {"python":sys.version,"platform":platform.platform(),"executable":sys.executable,
              "compiler_paths":{name:shutil.which(name) for name in ("gcc","g++","gfortran","f2c")},
              "f2c_header_present":Path("/usr/include/f2c.h").exists(),
              "imports":imports,"build_command":command,"build_exit_status":build.returncode,
              "finding":"See complete verbose log; build success is distinct from import and physics validation",
              "suspect_lines":[line for line in logtext.splitlines()
                               if any(token in line.lower() for token in ("error:","fatal error","cannot find","distutils","f2py"))]}
    (out/"diagnosis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print(f"Evidence directory: {out}")
    return build.returncode


if __name__ == "__main__":
    raise SystemExit(main())
