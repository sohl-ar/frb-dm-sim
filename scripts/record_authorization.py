"""Record human-supplied evidence and executed Table-1 before/after results."""
import hashlib
import json
from pathlib import Path
from frbsim.catalog import git_provenance

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(name):
        return json.loads((ROOT/"results"/name).read_text(encoding="utf-8"))
    def write(name, payload):
        (ROOT/"results"/name).write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    colab_path = ROOT/"results/colab/gates-e5b57d6.json"
    colab = read("colab/gates-e5b57d6.json")
    write("colab/environment-e5b57d6.json", {
        "evidence_source":"user-supplied Colab appendix; not locally executed",
        "authorization":"docs/SPEC-02a-authorization.txt",
        "gate_report_sha256":hashlib.sha256(colab_path.read_bytes()).hexdigest(),
        "environment_source":"human report in authorization section 7",
        "reported_environment":{"platform":"linux","python":"3.13.15","pytest":"8.4.2","gfortran":"11.2.0"},
        "installed_packages_status":"unverified; pip install transaction aborted at pygedm wheel build",
        "git_sha":colab["git_sha"],"git_dirty":colab["git_dirty"],
        "gate_status":{k:v["status"] for k,v in colab["gates"].items()},
        "acceptance_complete":colab["acceptance_complete"],"exit_status":colab["exit_status"]})
    before, after = read("table1-before.json"), read("table1-after.json")
    def extract(run):
        return next(c["measured"] for c in run["gates"]["G4"]["checks"] if c["name"] == "L0_Table1_zero_noise_limit")
    b, a = extract(before), extract(after)
    cb = extract(colab)
    colab_event = next(e for e in cb["events"] if e["frb"]=="190102")
    rerun_event = next(e for e in a["events"] if e["frb"]=="190102")
    changes = []
    for old,new in zip(b["events"],a["events"]):
        assert old["frb"] == new["frb"]
        changes.append({"frb":old["frb"],"before":old,"after":new,
            "delta_dm_cosmic_estimated":new["dm_cosmic_estimated"]-old["dm_cosmic_estimated"],
            "delta_analytic_mean":new["analytic_mean"]-old["analytic_mean"],
            "delta_predictive_5_95":[y-x for x,y in zip(old["predictive_5_95"],new["predictive_5_95"])],
            "coverage_changed":old["inside"] != new["inside"]})
    write("table1-amendment.json", {**git_provenance(),
        "authorization":"docs/SPEC-02a-authorization.txt section 1",
        "human_table1_spot_check":"complete: all six rows, per explicit human sign-off",
        "test_consumers":["tests/gates/test_g4.py::test_g4_table1"],
        "before_report":"table1-before.json","after_report":"table1-after.json",
        "seed":2020190608,"coverage_before":b["covered"],"coverage_after":a["covered"],
        "colab_before_vs_rerun_after_190102":{
            "before_source":"colab/gates-e5b57d6.json","before_git_sha":colab["git_sha"],
            "after_source":"table1-after.json","after_git_sha":after["git_sha"],
            "before":colab_event,"after":rerun_event,
            "delta_dm_cosmic_estimated":rerun_event["dm_cosmic_estimated"]-colab_event["dm_cosmic_estimated"],
            "delta_analytic_mean":rerun_event["analytic_mean"]-colab_event["analytic_mean"],
            "delta_predictive_5_95":[y-x for x,y in zip(colab_event["predictive_5_95"],rerun_event["predictive_5_95"])],
            "note":"Cross-environment comparison includes tiny constants/version differences; local controlled before/after also retained."},
        "outcome_unchanged":all(not e["coverage_changed"] for e in changes),"events":changes})
    print(json.dumps({"coverage_before":b["covered"],"coverage_after":a["covered"],
                      "190102":next(e for e in changes if e["frb"]=="190102")}))


if __name__ == "__main__":
    main()
