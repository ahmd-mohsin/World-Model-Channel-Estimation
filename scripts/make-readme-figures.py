"""Generate static README figures from the real 100k end-to-end run (results/e2e_100k/)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results/e2e_100k")
OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.facecolor": "#0f141d", "figure.facecolor": "#0a0e14",
                     "axes.edgecolor": "#2a3a4a", "text.color": "#e6edf3",
                     "axes.labelcolor": "#c9d4e0", "xtick.color": "#7d8794",
                     "ytick.color": "#7d8794", "grid.color": "#1e2a38"})

hist = json.loads((R / "metrics.json").read_text())
ev = json.loads((R / "eval.json").read_text())
X = [h["step"] for h in hist]

# ---- per-module loss panel ----
fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
panels = [("total", "Total loss", "#58a6ff"), ("jepa", "JEPA (world model)", "#3fb950"),
          ("vic", "VICReg (anti-collapse)", "#e3b341"), ("chan", "Channel (task)", "#bc8cff")]
for a, (k, title, c) in zip(ax, panels):
    a.plot(X, [h[k] for h in hist], color=c, lw=2)
    a.fill_between(X, [h[k] for h in hist], alpha=0.12, color=c)
    a.set_title(title, color="#c9d4e0"); a.set_xlabel("step"); a.grid(alpha=0.3)
fig.suptitle("End-to-end joint training — per-module losses (100k Sionna, 8× A100)", color="#e6edf3", fontsize=13)
fig.savefig(OUT / "e2e_losses.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# ---- channel-est NMSE vs SNR ----
sw = ev["channel_sweep"]; snrs = sorted(sw.keys(), key=float)
import numpy as np
x = np.arange(len(snrs)); w = 0.26
fig, a = plt.subplots(figsize=(9, 5))
a.bar(x - w, [sw[s]["ls"] for s in snrs], w, label="LS", color="#7d8794")
a.bar(x,     [sw[s]["mmse"] for s in snrs], w, label="MMSE", color="#e3b341")
a.bar(x + w, [sw[s]["sswm"] for s in snrs], w, label="SSWM (ours)", color="#58a6ff")
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([f"{s} dB" for s in snrs])
a.set_ylabel("NMSE (log, lower better)"); a.legend()
a.set_title("Channel estimation: SSWM vs LS / MMSE (100k held-out)", color="#e6edf3")
a.grid(alpha=0.3, axis="y")
fig.savefig(OUT / "e2e_channel_est.png", dpi=120, bbox_inches="tight"); plt.close(fig)

print("saved docs/assets/e2e_losses.png and docs/assets/e2e_channel_est.png")
