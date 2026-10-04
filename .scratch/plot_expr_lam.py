import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHODS = ("blend", "vote", "schedule")
COLORS = {"blend": "tab:blue", "vote": "tab:orange", "schedule": "tab:green"}

s = pd.read_csv("resources/52_composition_summary.csv")
wide = s.pivot_table(index=["method", "n"], columns="lam", values="joint").reset_index()
wide = wide.rename(columns={0.0: "lam0", 0.4: "lam04"})
print(wide)

lo = min(wide["lam0"].min(), wide["lam04"].min())
hi = max(wide["lam0"].max(), wide["lam04"].max())
pad = 0.03 * (hi - lo)
lim = (lo - pad, hi + pad)

fig, ax = plt.subplots(figsize=(5.5, 4.6))
ax.plot(lim, lim, color="gray", lw=0.8, ls="--", zorder=0, label="y = x (no fluency cost)")
for method in METHODS:
    m = wide[wide["method"] == method].sort_values("n")
    ax.plot(m["lam0"], m["lam04"], "-o", color=COLORS[method], label=method, zorder=2)
    for _, row in m.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["lam0"], row["lam04"]),
                    textcoords="offset points", xytext=(6, -2), fontsize=8,
                    color=COLORS[method])

ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("mean joint expression delta  ($\\lambda$ = 0)")
ax.set_ylabel("mean joint expression delta  ($\\lambda$ = 0.4)")
ax.set_title("Expression: $\\lambda$ = 0 vs $\\lambda$ = 0.4")
ax.legend(fontsize=8)
fig.tight_layout()
out = "resources/52_expression_lam0_vs_lam04.png"
fig.savefig(out, dpi=150)
print("saved", out)
