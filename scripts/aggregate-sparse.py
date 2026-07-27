"""Aggregate the sparse-pilot wave. The headline question: does the world-model prior win when
pilots are sparse (full vs noprior), and does it beat classical interpolation / masked-MMSE?

    python scripts/aggregate-sparse.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path

S = Path("results/sparse")


def load():
    return {Path(f).stem[len("eval_"):]: json.loads(Path(f).read_text())
            for f in sorted(glob.glob(str(S / "eval_*.json")))}


def main():
    ev = load()
    if not ev:
        print("no results in results/sparse/ yet"); return

    # group by (density, split): tags like sp_s4_full, sp_s4_noprior, sp_ood_s4_full
    def show(dens, full_key, nop_key, title):
        if full_key not in ev or nop_key not in ev:
            print(f"\n[{title}] missing {full_key!r}/{nop_key!r}"); return
        f, n = ev[full_key], ev[nop_key]
        snrs = sorted({int(s) for s in f}, key=int)
        print(f"\n=== {title} (stride {dens}) : estimation NMSE vs SNR ===")
        print(f"{'SNR':>4} {'interp':>9} {'mMMSE':>9} {'full':>9} {'noprior':>9}   prior lift")
        lifts = []
        for s in snrs:
            fr = f[str(s)]
            g_full, g_nop = fr["model"], n[str(s)]["model"]
            lift = 100 * (g_nop - g_full) / g_full if g_full else 0
            lifts.append(lift)
            best = min(fr["interp"], fr["mmse"])
            beat = "*" if g_full < best else " "
            print(f"{s:>4} {fr['interp']:>9.4f} {fr['mmse']:>9.4f} {g_full:>9.4f}{beat} {g_nop:>9.4f}"
                  f"   {lift:+.0f}%")
        print(f"  (* = full beats best classical)  avg prior lift = {sum(lifts)/len(lifts):+.0f}% "
              f"(full vs noprior)")

    show(2, "sp_s2_full", "sp_s2_noprior", "IN-DIST 50% pilots")
    show(4, "sp_s4_full", "sp_s4_noprior", "IN-DIST 25% pilots")
    show(8, "sp_s8_full", "sp_s8_noprior", "IN-DIST 12.5% pilots")
    show(4, "sp_ood_s4_full", "sp_ood_s4_noprior", "OOD 25% pilots")

    # headline: does the prior lift GROW as pilots get sparser?
    print("\n=== HEADLINE: world-model prior lift vs pilot density (in-dist) ===")
    for dens, fk, nk in [(2, "sp_s2_full", "sp_s2_noprior"),
                         (4, "sp_s4_full", "sp_s4_noprior"),
                         (8, "sp_s8_full", "sp_s8_noprior")]:
        if fk in ev and nk in ev:
            snrs = [str(s) for s in ev[fk]]
            lift = sum(100 * (ev[nk][s]["model"] - ev[fk][s]["model"]) / ev[fk][s]["model"]
                       for s in snrs) / len(snrs)
            print(f"  stride {dens} ({100//dens}% pilots): prior lifts NMSE {lift:+.0f}% (full vs noprior)")
    (S / "SUMMARY_sparse.json").write_text(json.dumps(ev, indent=2))
    print(f"\nwrote {S/'SUMMARY_sparse.json'}")


if __name__ == "__main__":
    main()
