"""Aggregate the module-ablation wave: is the world-model prior load-bearing?

Reads results/ablation/eval_ab_*.json (+ pred_ab_*.json) and prints, for estimation NMSE vs SNR:
  - in-distribution: full vs {prior, action, ssm, beamspace}
  - OOD (held-out scene): full vs {prior, ssm}
  - the KEY numbers: how much worse is "no prior" than "full" (the thesis test),
    and whether that gap is LARGER out-of-distribution.

    python scripts/aggregate-ablation.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path

A = Path("results/ablation")


def load(prefix):
    out = {}
    for f in sorted(glob.glob(str(A / f"{prefix}_ab_*.json"))):
        tag = Path(f).stem[len(prefix) + 4:]   # strip "<prefix>_ab_"
        out[tag] = json.loads(Path(f).read_text())
    return out


def pct(worse, base):
    return f"{100*(worse-base)/base:+.0f}%" if base else "n/a"


def table(evals, base_key, others, title):
    if base_key not in evals:
        print(f"\n[{title}] base run {base_key!r} missing"); return
    base = evals[base_key]
    snrs = sorted({int(s) for s in base}, key=int)
    cols = [base_key] + [k for k in others if k in evals]
    print(f"\n=== {title}: estimation NMSE vs SNR (Beam-WM) ===")
    print("SNR".rjust(4) + "".join(k.rjust(12) for k in cols) + "   (Δ vs full at that SNR)")
    for s in snrs:
        row = f"{s:>4}"
        for k in cols:
            row += f"{evals[k][str(s)]['beam']:>12.4f}"
        # delta of the 'prior'/no-prior run vs full
        deltas = []
        for k in cols[1:]:
            deltas.append(f"{k}:{pct(evals[k][str(s)]['beam'], base[str(s)]['beam'])}")
        print(row + "   " + " ".join(deltas))


def main():
    ev = load("eval")
    if not ev:
        print("no results in results/ablation/ yet"); return

    indist = {k[len("indist_"):]: v for k, v in ev.items() if k.startswith("indist_")}
    ood = {k[len("ood_"):]: v for k, v in ev.items() if k.startswith("ood_")}

    table(indist, "full", ["prior", "action", "ssm", "beamspace"],
          "IN-DISTRIBUTION")
    table(ood, "full", ["prior", "ssm"], "OOD (held-out scene)")

    # ---- headline: mean over SNR of the no-prior penalty, in-dist vs OOD ----
    def mean_penalty(group):
        if "full" not in group or "prior" not in group:
            return None
        snrs = [str(s) for s in group["full"]]
        rel = [(group["prior"][s]["beam"] - group["full"][s]["beam"]) / group["full"][s]["beam"]
               for s in snrs]
        return 100 * sum(rel) / len(rel)

    print("\n=== HEADLINE: does removing the world-model prior hurt? ===")
    pin = mean_penalty(indist); poo = mean_penalty(ood)
    if pin is not None:
        print(f"  in-distribution: no-prior is {pin:+.0f}% NMSE vs full (avg over SNR)")
    if poo is not None:
        print(f"  out-of-distribution: no-prior is {poo:+.0f}% NMSE vs full (avg over SNR)")
    if pin is not None and poo is not None:
        verdict = ("PRIOR IS LOAD-BEARING (and more so OOD)" if poo > pin > 3
                   else "PRIOR IS LOAD-BEARING" if pin > 3
                   else "PRIOR BUYS LITTLE — thesis at risk")
        print(f"  -> {verdict}")

    # prediction sanity: action/ssm should hurt prediction most
    (A / "SUMMARY_ablation.json").write_text(json.dumps({"indist": indist, "ood": ood}, indent=2))
    print(f"\nwrote {A/'SUMMARY_ablation.json'}")


if __name__ == "__main__":
    main()
