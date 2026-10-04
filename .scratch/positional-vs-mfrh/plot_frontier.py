"""Steering-vs-fluency frontier: mFRH (frame) vs positional vs pooled subspace,
on the shared pooled-r8 cosine metric over the notebook-32 concept pool.

Reads the positional pilot's full-run generations.jsonl (positional, pooled_rank8
arms) and the separately generated mFRH arm, all evaluated identically. Raw score
only (normalize_score=False), which is the axis mFRH lives on.
"""

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/rodrigo/projects/frh-multi")
SCRATCH = Path("/tmp/claude-1010/-home-rodrigo-projects-frh-multi/b92c27ff-2cfb-46e4-b2ee-dd9754efd859/scratchpad")
OUT = Path(__file__).parent / "positional_vs_mfrh_frontier.png"

# newest full-run dir (1360 rows)
run_dir = max(
    glob.glob(str(ROOT / "resources/positional_subspace_pilot_unquantized/*/")),
    key=lambda p: Path(p).stat().st_mtime,
)
rows = [json.loads(l) for l in open(Path(run_dir) / "generations.jsonl")]
rows += [json.loads(l) for l in open(SCRATCH / "mfrh_arm.jsonl")]


def on_target(r):
    c = r["cosine"]["shared_pooled_rank8"]
    return c["per_concept"][r["case"]]["delta"] if c["valid"] else None


LABEL = {"mfrh": "mFRH (frame)", "positional": "positional subspace", "pooled_rank8": "pooled subspace (r8)"}
COLOR = {"mfrh": "#4C72B0", "positional": "#C44E52", "pooled_rank8": "#55A868"}

stats = {}
for method in LABEL:
    for lam in (0.0, 0.4):
        cos = [on_target(r) for r in rows
               if r["method"] == method and r["lambda"] == lam and not r["normalize_score"]]
        cos = [c for c in cos if c is not None]
        ppl = [r["ppl"]["ppl_ratio"] for r in rows
               if r["method"] == method and r["lambda"] == lam and not r["normalize_score"]
               and r["ppl"]["valid"]]
        if cos and ppl:
            sem = st.pstdev(cos) / (len(cos) ** 0.5)
            stats[(method, lam)] = (st.median(ppl), st.mean(cos), sem, len(cos))

fig, ax = plt.subplots(figsize=(8.2, 5.6))
for method in LABEL:
    pts = [(stats[(method, lam)], lam) for lam in (0.0, 0.4) if (method, lam) in stats]
    xs = [p[0][0] for p in pts]
    ys = [p[0][1] for p in pts]
    es = [p[0][2] for p in pts]
    ax.plot(xs, ys, "-", color=COLOR[method], lw=1.6, zorder=1)
    ax.errorbar(xs, ys, yerr=es, fmt="o", color=COLOR[method], ms=9,
                capsize=3, label=LABEL[method], zorder=2)
    for (x, y, _, n), lam in pts:
        ax.annotate(f"λ={lam}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)

ax.axvline(1.0, ls=":", c="grey", lw=1)
ax.text(1.02, ax.get_ylim()[0], "baseline\nfluency", fontsize=7.5, color="grey", va="bottom")
ax.set_xlabel("fluency cost — median PPL ratio (steered / baseline, lower = better)")
ax.set_ylabel("steering — mean on-target cosine gain vs baseline\n(shared pooled-r8 reference, higher = better)")
ax.set_title("mFRH vs positional vs pooled subspace on the nb32 concept pool\n"
             "(Llama-3.1-8B bf16, 40 concepts × 2 prompts, k=4/24 steps)", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
for k, v in sorted(stats.items()):
    print(f"  {k[0]:14} lam={k[1]}: ppl_med={v[0]:.2f}  cos_delta={v[1]:+.4f}±{v[2]:.4f}  n={v[3]}")
