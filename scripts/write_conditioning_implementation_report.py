"""Render quick-check numbers from executed JSON/XML; no science acceptance."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from frbsbi.data import ROOT
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR
from frbsim.catalog import git_provenance

r = json.loads((RUN_DIR/"statistics-quickcheck.json").read_text())
c = np.asarray(r["correlations"])
v = np.asarray(r["standardized_variance"])
xml = ET.parse(RUN_DIR/"quick-unit-tests.xml").getroot()
suites = list(xml.iter("testsuite"))
counts = {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
report = {**git_provenance(),"scope":"implementation checks only; no new training or full acceptance",
          "statistics_count":len(r["statistics"]),"validation_catalogs":len(r["catalog_seeds"]),
          "varying_dimensions":int(np.count_nonzero(v)),"constant_dimensions":r["constant_validation_dimensions"],
          "max_absolute_correlation_by_parameter":dict(zip(r["parameter_order"],np.abs(c).max(0).tolist())),
          "standardized_variance_range_among_varying_dimensions":[float(v[v>0].min()),float(v.max())],
          "unit_tests":counts,"training_started":(RUN_DIR/"TRAINING_ATTEMPT.lock").exists(),
          "new_statistical_acceptance":"not_run"}
assert not report["training_started"]
assert counts["failures"]==counts["errors"]==0
write_json(RUN_DIR/"implementation-checks.json",report)
lines = ["# Statistics bypass quick-check evidence","",
         "Generated from statistics-quickcheck.json and quick-unit-tests.xml; no training was performed.","",
         f"Validation catalogs: {report['validation_catalogs']}. Statistics: {report['statistics_count']}; "
         f"varying dimensions: {report['varying_dimensions']}. Availability flags constant in this validation subset "
         "are retained because absent/singleton subsets occur elsewhere.","",
         "Correlations are descriptive and unadjusted for multiple comparisons. They do not demonstrate posterior learning.","",
         "| Statistic | Raw variance | Standardized variance | f_d r | F r | Host median r | Host sigma_ln r |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for i,name in enumerate(r["statistics"]):
    lines.append(f"| {name} | {r['raw_variance'][i]:.6g} | {v[i]:.6g} | "+" | ".join(f"{x:.6f}" for x in c[i])+" |")
lines += ["",f"Forward/backward test loss (untrained model): {r['forward_loss']:.9g}.",
          f"Bypass first-layer gradient norm: {r['bypass_first_layer_gradient_norm']:.9g}.",
          f"Summary dimension: {r['summary_dimension']}; velocity input dimension: {r['velocity_input_dimension']}.",
          f"Unit checks: {counts['tests']} tests, {counts['failures']} failures, {counts['errors']} errors.","",
          "The full correlation matrix and all absolute-correlation flags are retained in the source JSON.",""]
(ROOT/"STATISTICS_QUICKCHECK.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
(ROOT/"work").mkdir(exist_ok=True)
shutil.copyfile(ROOT/"HANDOFF.md",ROOT/"work/HANDOFF.md")
print(json.dumps(report,indent=2))
