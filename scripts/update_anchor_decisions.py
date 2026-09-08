"""Render computed evidence into the decision record; never hand-type results."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/"results/selection-audit.json").read_text(encoding="utf-8"))
    lines = ["", "Source: `results/selection-audit.json`; human print correction v2.", "",
             "| z | Authorized comparison print (%) | Computed probability (%) |", "|---|---:|---:|"]
    for z,bounds,p in zip(report["z"],report["user_print_ranges"],report["detected_fraction"]):
        label = f"{bounds[0]*100:g}" if bounds[0]==bounds[1] else f"{bounds[0]*100:g}-{bounds[1]*100:g}"
        lines.append(f"| {z:g} | {label} | {p*100:.6f} |")
    v = report["z02_verification"]
    lines += ["",f"z=0.2: x_t={v['x_threshold']:.12g}; integrand x^gamma exp(-x)={v['unnormalized_tail_integrand_x_gamma_exp_minus_x']:.12g}.",
              f"Quadrature={v['quadrature_probability']:.14g}; series={v['series_probability']:.14g}; rejection={v['rejection_probability']:.14g}",
              f"with standard error {v['rejection_standard_error']:.8g}. The row status is {v['status']}.",""]
    knee = report["knee_fluences"]
    for z,f in zip(knee["z"],knee["fluence_jy_ms"]):
        lines.append(f"Knee fluence at z={z:g}: {f:.8f} Jy ms.")
    lines += [knee["interpretation"],"","The executed below-z=0.5 fractions (quadrature / rejection) are:",""]
    for name,r in report["population_naive_rejection"].items():
        lines.append(f"- {name}: {r['quadrature_fraction_detected_below_z05']*100:.4f}% / {r['fraction_detected_below_z05']*100:.4f}%.")
    lines += ["","Human third checks at z=1 and 1.4 agree at 10-15%; at z=0.2 approximately 1%.",
              "These are labeled corroboration. The executed numerical checks are the arbiter.",""]
    path = root/"DECISIONS.md"
    text = path.read_text(encoding="utf-8")
    left,rest = text.split("<!-- GENERATED_ANCHOR_EVIDENCE -->")
    _,right = rest.split("<!-- END_GENERATED_ANCHOR_EVIDENCE -->")
    path.write_text(left+"<!-- GENERATED_ANCHOR_EVIDENCE -->\n"+"\n".join(lines)+
                    "<!-- END_GENERATED_ANCHOR_EVIDENCE -->"+right,encoding="utf-8")


if __name__ == "__main__":
    main()
