"""Aggregate the stress-matrix results into comparison tables.

Reads results/stress/{eval,pred}_*.json and prints:
  1. Estimation NMSE vs SNR: in-distribution (mean +/- std over seeds) vs MMSE vs LS
  2. Estimation NMSE vs SNR: per-scene OOD (leave-one-out) vs MMSE
  3. Prediction NMSE vs horizon: in-dist mean vs persistence vs AR(1)
  4. Win/loss summary vs MMSE (the headline claim)

    python scripts/aggregate-stress.py
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path

S = Path("results/stress")


def load(prefix):
    out = {}
    for f in sorted(glob.glob(str(S / f"{prefix}_*.json"))):
        tag = Path(f).stem[len(prefix) + 1:]
        out[tag] = json.loads(Path(f).read_text())
    return out


def mean_std(xs):
    n = len(xs); mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / n) if n > 1 else 0.0
    return mu, sd


def main():
    evals = load("eval")
    preds = load("pred")
    if not evals:
        print("no results in results/stress/ yet"); return

    indist = {k: v for k, v in evals.items() if k.startswith("indist")}
    ood = {k: v for k, v in evals.items() if k.startswith("ood_")}

    # ---- 1. in-distribution estimation, mean+/-std over seeds ----
    if indist:
        snrs = sorted({int(s) for v in indist.values() for s in v}, key=int)
        print("\n=== ESTIMATION (in-distribution) NMSE vs SNR ===")
        print(f"(mean +/- std over {len(indist)} seeds)")
        print(f"{'SNR':>4} {'LS':>10} {'MMSE':>10} {'Beam-WM':>18} {'win?':>6}")
        wins = 0
        for snr in snrs:
            ls = mean_std([indist[k][str(snr)]["ls"] for k in indist])[0]
            mm = mean_std([indist[k][str(snr)]["mmse"] for k in indist])[0]
            bmu, bsd = mean_std([indist[k][str(snr)]["beam"] for k in indist])
            w = "YES" if bmu < mm else "no"
            wins += bmu < mm
            print(f"{snr:>4} {ls:>10.4f} {mm:>10.4f} {bmu:>10.4f}+/-{bsd:<6.4f} {w:>6}")
        print(f"Beam-WM beats MMSE at {wins}/{len(snrs)} SNRs (in-dist)")

    # ---- 2. per-scene OOD estimation ----
    if ood:
        snrs = sorted({int(s) for v in ood.values() for s in v}, key=int)
        print("\n=== ESTIMATION (OOD, leave-one-scene-out) Beam-WM vs MMSE ===")
        hdr = "scene".ljust(30) + "".join(f"{s:>8}" for s in snrs)
        print(hdr)
        agg_win = agg_tot = 0
        for scene in sorted(ood):
            row = scene[:29].ljust(30)
            for snr in snrs:
                d = ood[scene][str(snr)]
                mark = "*" if d["beam"] < d["mmse"] else " "   # * = Beam wins
                agg_win += d["beam"] < d["mmse"]; agg_tot += 1
                row += f"{d['beam']:>7.4f}{mark}"
            print(row)
        # MMSE reference row
        print("-" * len(hdr))
        for scene in sorted(ood):
            row = ("  MMSE@" + scene[:22]).ljust(30)
            for snr in snrs:
                row += f"{ood[scene][str(snr)]['mmse']:>8.4f}"
            print(row); break  # print one representative MMSE row per scene would be noisy; show first
        print(f"(* = Beam-WM < MMSE)  OOD wins: {agg_win}/{agg_tot}")

    # ---- 3. prediction vs horizon (in-dist mean) ----
    pin = {k: v for k, v in preds.items() if k.startswith("indist")}
    if pin:
        ks = sorted({int(k) for v in pin.values() for k in v}, key=int)
        print("\n=== PREDICTION NMSE vs horizon k (in-dist, channel space) ===")
        print(f"{'k':>3} {'persist':>10} {'AR(1)':>10} {'Beam-WM':>10} {'winner':>10}")
        for k in ks:
            pr = mean_std([pin[t][str(k)]["persist"] for t in pin])[0]
            ar = mean_std([pin[t][str(k)]["ar1"] for t in pin])[0]
            bm = mean_std([pin[t][str(k)]["beam"] for t in pin])[0]
            win = min([("persist", pr), ("AR(1)", ar), ("Beam-WM", bm)], key=lambda x: x[1])[0]
            print(f"{k:>3} {pr:>10.4f} {ar:>10.4f} {bm:>10.4f} {win:>10}")

    # ---- write a machine-readable summary ----
    summary = {"indist": indist, "ood": ood, "pred": preds}
    (S / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {S/'SUMMARY.json'}")


if __name__ == "__main__":
    main()
