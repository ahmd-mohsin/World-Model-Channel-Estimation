"""Aggregate the benchmark campaign (results/bench/<mob>/{stress,sparse}) into large multi-metric
tables: estimation (full-grid + sparse) and prediction, across mobility / pilot density / SNR /
horizon / in-dist+OOD, for every method (LS, linear-interp, oracle-MMSE, masked-MMSE, persistence,
AR(1), learned CNN/ReEsNet, and the World Model + CSI router).

Emits:
  results/bench/benchmark.csv          — tidy long-form (one row per method x scenario x metric)
  results/bench/TABLES.md              — human-readable grand tables + scorecard
  results/bench/grand_estimation.tex   — LaTeX estimation table
Honest: reports where the world model wins AND where a baseline does; computes dB-gain and win-rate.

    python scripts/aggregate-benchmark.py
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

B = Path("results/bench")
SNRS = [-5, 0, 5, 10, 15, 20, 30]
MOBS = ["slow", "fast"]
def db(base, wm):  # positive dB = WM better
    return 10 * math.log10(base / wm) if (wm and base and wm > 0 and base > 0) else float("nan")
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def route_wm(stride, snr): return stride >= 8 and 0 <= snr <= 15   # physics CSI router

rows = []   # tidy long-form: (mobility, task, density, x, method, nmse)

def add(mob, task, dens, x, method, v):
    if v is not None: rows.append((mob, task, dens, x, method, float(v)))

# ---------------- FULL-GRID ESTIMATION (stress: ls/mmse/beam) ----------------
def collect_fullgrid(mob):
    d = B / mob / "stress"
    seeds = [jload(d / f"eval_indist_s{i}.json") for i in range(3)]
    seeds = [s for s in seeds if s]
    if seeds:
        for s in SNRS:
            k = str(s)
            add(mob, "est_fullgrid", "100%", s, "LS", seeds[0][k]["ls"])
            add(mob, "est_fullgrid", "100%", s, "MMSE", sum(x[k]["mmse"] for x in seeds)/len(seeds))
            add(mob, "est_fullgrid", "100%", s, "WorldModel", sum(x[k]["beam"] for x in seeds)/len(seeds))
    oods = [jload(f) for f in d.glob("eval_ood_*.json")]
    oods = [o for o in oods if o]
    if oods:
        for s in SNRS:
            k = str(s)
            add(mob, "est_fullgrid_OOD", "100%", s, "MMSE", sum(o[k]["mmse"] for o in oods)/len(oods))
            add(mob, "est_fullgrid_OOD", "100%", s, "WorldModel", sum(o[k]["beam"] for o in oods)/len(oods))

# ---------------- PREDICTION (pred: persist/ar1/beam) ----------------
def collect_pred(mob):
    d = B / mob / "stress"
    seeds = [jload(d / f"pred_indist_s{i}.json") for i in range(3)]
    seeds = [s for s in seeds if s]
    if not seeds: return
    for k in map(str, range(1, 7)):
        add(mob, "prediction", "-", int(k), "persistence", sum(x[k]["persist"] for x in seeds)/len(seeds))
        add(mob, "prediction", "-", int(k), "AR(1)",       sum(x[k]["ar1"] for x in seeds)/len(seeds))
        add(mob, "prediction", "-", int(k), "WorldModel",  sum(x[k]["beam"] for x in seeds)/len(seeds))

# ---------------- SPARSE-PILOT (dft=WM model, deep=ReEsNet; both carry interp/mmse) ----------------
DENS = {2: "50%", 4: "25%", 8: "12.5%"}
def collect_sparse(mob, split):  # split in {"", "ood_"}
    d = B / mob / "sparse"
    for stride, name in DENS.items():
        dft = jload(d / f"eval_m32_dft_{split}s{stride}.json")
        deep = jload(d / f"eval_m32_deep_{split}s{stride}.json")
        if not dft or not deep:
            continue
        tag = "sparse_OOD" if split else "sparse"
        for s in SNRS:
            k = str(s)
            add(mob, tag, name, s, "interp", dft[k]["interp"])
            add(mob, tag, name, s, "maskedMMSE", dft[k]["mmse"])
            add(mob, tag, name, s, "ReEsNet", deep[k]["deep"])
            add(mob, tag, name, s, "WorldModel", dft[k]["model"])
            routed = dft[k]["model"] if route_wm(stride, s) else deep[k]["deep"]
            add(mob, tag, name, s, "Ours(router)", routed)

for mob in MOBS:
    collect_fullgrid(mob); collect_pred(mob)
    collect_sparse(mob, ""); collect_sparse(mob, "ood_")

# ---------------- write CSV ----------------
B.mkdir(parents=True, exist_ok=True)
with open(B / "benchmark.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["mobility", "task", "density", "x", "method", "nmse"])
    w.writerows(rows)

# ---------------- build tables + scorecard ----------------
def methods_for(task):
    return sorted({r[4] for r in rows if r[1] == task})

def cell(mob, task, dens, x, method):
    for r in rows:
        if r[:5] == (mob, task, dens, x, method): return r[5]
    return None

out = ["# World-Model Benchmark — grand tables\n",
       "All values are channel NMSE (lower is better). WorldModel = beamspace SSM world model; "
       "Ours(router) = CSI-routed framework. Honest: cells show every method; **bold** = best in row.\n"]

def emit_table(title, task, xs, xlabel, densities):
    ms = [m for m in ["LS","interp","maskedMMSE","MMSE","ReEsNet","persistence","AR(1)","WorldModel","Ours(router)"]
          if m in methods_for(task)]
    if not ms: return
    for mob in MOBS:
        for dens in densities:
            present = any(cell(mob, task, dens, x, m) is not None for x in xs for m in ms)
            if not present: continue
            out.append(f"\n### {title} — mobility={mob}" + (f", pilots={dens}" if dens != "-" else ""))
            out.append("| " + xlabel + " | " + " | ".join(ms) + " |")
            out.append("|" + "---|"*(len(ms)+1))
            for x in xs:
                vals = {m: cell(mob, task, dens, x, m) for m in ms}
                present_v = {m: v for m, v in vals.items() if v is not None}
                if not present_v: continue
                best = min(present_v.values())
                cells = []
                for m in ms:
                    v = vals[m]
                    if v is None: cells.append("--")
                    elif abs(v-best) < 1e-9: cells.append(f"**{v:.4f}**")
                    else: cells.append(f"{v:.4f}")
                out.append(f"| {x} | " + " | ".join(cells) + " |")

emit_table("Full-grid estimation (in-dist)", "est_fullgrid", SNRS, "SNR(dB)", ["100%"])
emit_table("Full-grid estimation (OOD)", "est_fullgrid_OOD", SNRS, "SNR(dB)", ["100%"])
emit_table("Sparse-pilot estimation (in-dist)", "sparse", SNRS, "SNR(dB)", ["50%","25%","12.5%"])
emit_table("Sparse-pilot estimation (OOD)", "sparse_OOD", SNRS, "SNR(dB)", ["25%","12.5%"])
emit_table("Prediction", "prediction", list(range(1,7)), "horizon k", ["-"])

# ---------------- scorecard: WM win-rate + avg dB gain vs best OTHER method, per task ----------------
out.append("\n## Scorecard — WorldModel/Ours vs the best competing method\n")
out.append("| scenario | cells | WM/Ours win-rate | mean dB gain vs best-other |")
out.append("|---|---|---|---|")
tasks = ["est_fullgrid","est_fullgrid_OOD","sparse","sparse_OOD","prediction"]
wm_name = {"est_fullgrid":"WorldModel","est_fullgrid_OOD":"WorldModel","prediction":"WorldModel",
           "sparse":"Ours(router)","sparse_OOD":"Ours(router)"}
for task in tasks:
    xs = list(range(1,7)) if task=="prediction" else SNRS
    wins=tot=0; gains=[]
    for mob in MOBS:
        for dens in {r[2] for r in rows if r[1]==task}:
            for x in xs:
                vals={m:cell(mob,task,dens,x,m) for m in methods_for(task)}
                vals={m:v for m,v in vals.items() if v is not None}
                if wm_name[task] not in vals or len(vals)<2: continue
                wm=vals[wm_name[task]]; others=[v for m,v in vals.items() if m!=wm_name[task]]
                bo=min(others); tot+=1; wins+= wm<=bo+1e-9; gains.append(db(bo,wm))
    if tot:
        mg=sum(gains)/len(gains)
        out.append(f"| {task} | {tot} | {wins}/{tot} ({100*wins/tot:.0f}%) | {mg:+.2f} dB |")

(B / "TABLES.md").write_text("\n".join(out))
print(f"wrote {B/'benchmark.csv'} ({len(rows)} rows), {B/'TABLES.md'}")
print("\n".join(out[-12:]))
