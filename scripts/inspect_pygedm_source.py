"""Record source evidence from an unpacked pygedm 3.3.0 sdist."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    source = Path(sys.argv[1])
    findings = {}
    for relative in ("setup.py", "pygedm/yt2020.py", "pygedm/__init__.py", "pygedm/pygedm.py"):
        path = source/relative
        content = path.read_bytes()
        text = content.decode("utf-8")
        findings[relative] = {"sha256":hashlib.sha256(content).hexdigest(),
            "matching_lines":[{"line":i,"text":line} for i,line in enumerate(text.splitlines(),1)
                              if any(token in line for token in ("f2c","distutils","simps","yt2020","__version__","language="))]}
    report = {"scope":"static inspection; original Linux verbose-build failure unknown",
              "source":"cached PyPI pygedm 3.3.0 sdist", "files":findings,
              "linux_verbose_build_executed_here":False}
    target = Path(__file__).resolve().parents[1]/"results/colab/pygedm-source-inspection.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
