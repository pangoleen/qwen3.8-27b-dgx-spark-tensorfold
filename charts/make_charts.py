"""Chart and table for the X post and the README, from data/tf-vs-sglang.csv.
Run with any Python that has matplotlib:  python charts/make_charts.py"""
import csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).parent
rows = list(csv.DictReader(open(HERE.parent / "data" / "tf-vs-sglang.csv")))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TF, SG = "#2a78d6", "#eb6834"          # categorical slots 1 and 2, validated (CVD dE 24.7)
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})

def bars(ax, sel, title, unit, note):
    sel = sel[::-1]
    n = len(sel); h = 0.36; gap = 0.04
    vmax = max(max(float(r["tensorfold"]), float(r["sglang"])) for r in sel)
    for i, r in enumerate(sel):
        for off, key, col in ((h / 2 + gap / 2, "tensorfold", TF), (-h / 2 - gap / 2, "sglang", SG)):
            v = float(r[key])
            ax.add_patch(FancyBboxPatch((0, i + off - h / 2), v, h, boxstyle="round,pad=0,rounding_size=0.06",
                                        mutation_aspect=1 / (vmax / n) * 0.9, fc=col, ec="none"))
            ax.text(v + vmax * 0.015, i + off, f"{v:,.1f}".rstrip("0").rstrip(".") if v < 1000 else f"{v:,.0f}",
                    va="center", ha="left", fontsize=11.5, color=INK2)
    ax.set_xlim(0, vmax * 1.2); ax.set_ylim(-0.6, n - 0.4) if n > 1 else ax.set_ylim(-1.1, 1.1)
    ax.set_yticks(range(n)); ax.set_yticklabels([r["metric"] for r in sel])
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=22)
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=11, color=INK2, va="bottom")
    ax.set_xlabel(unit, fontsize=11)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

g = lambda name: [r for r in rows if r["group"] == name]
fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.62, wspace=0.42, left=0.15, right=0.97, top=0.76, bottom=0.13)
for ax in (a := [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]):
    ax.set_facecolor(SURF)
bars(a[0], g("decode"), "Decode", "tok/s", "higher is better")
bars(a[1], g("prefill"), "Prefill", "tok/s", "higher is better")
bars(a[2], g("longdecode"), "Decode after a long prompt", "tok/s", "higher is better")
bars(a[3], g("agent"), "Coding-agent task", "seconds", "lower is better: pi, main agent + 4 sub-agents")
fig.text(0.03, 0.935, "Qwen3.8-27B on one DGX Spark: TensorFold vs SGLang", fontsize=24, fontweight="bold")
fig.text(0.03, 0.885, "Same weights and drafter (RadixArk NVFP4 + incoai DFlash2). TensorFold native 1.2.9 decodes faster; "
         "SGLang prefills faster, and wins agent work.", fontsize=13.5, color=INK2)
for x, col, lab in ((0.03, TF, "TensorFold native 1.2.9"), (0.20, SG, "SGLang")):
    fig.patches.append(FancyBboxPatch((x, 0.833), 0.012, 0.02, boxstyle="round,pad=0,rounding_size=0.003",
                                      transform=fig.transFigure, fc=col, ec="none"))
    fig.text(x + 0.018, 0.843, lab, fontsize=13, va="center")
fig.text(0.03, 0.015, "One GB10, 2026-11-18, thinking off. Decode/prefill: sparkDash collectors, temperature 0. "
         "Agent: median of 6 runs, 6/6 pass on both. github.com/pangoleen/qwen3.8-27b-dgx-spark-tensorfold",
         fontsize=10.5, color=INK2)
fig.savefig(HERE / "tf-vs-sglang.png", facecolor=SURF)

# the table, as an image
def delta(r):
    t, s = float(r["tensorfold"]), float(r["sglang"])
    if r["better"] == "lower":
        return f"SGLang {100 * (t - s) / t:.0f}% faster"
    d = 100 * (t - s) / s
    return f"{d:+.0f}%"
order = g("decode") + g("longdecode") + g("prefill") + g("agent")
sec = {"decode": "Decode", "longdecode": "Decode after a long prompt", "prefill": "Prefill", "agent": "Agent work"}
cells, kinds, last = [], [], None
for r in order:
    if r["group"] != last:
        cells.append([sec[r["group"]], "", "", ""]); kinds.append("head"); last = r["group"]
    fmt = (lambda v: f"{float(v):,.0f}") if float(r["tensorfold"]) >= 1000 else (lambda v: f"{float(v):g}")
    cells.append([r["metric"], fmt(r["tensorfold"]) + " " + r["unit"], fmt(r["sglang"]) + " " + r["unit"], delta(r)])
    kinds.append("row")
fig2 = plt.figure(figsize=(12, 0.52 * len(cells) + 2.0), dpi=110, facecolor=SURF)
ax = fig2.add_axes([0.03, 0.02, 0.94, 0.80]); ax.axis("off"); ax.set_facecolor(SURF)
cols = ["", "TensorFold 1.2.9", "SGLang", "TensorFold vs SGLang"]
xs = [0.0, 0.47, 0.66, 0.84]
ax.set_xlim(0, 1); ax.set_ylim(len(cells) + 0.6, -0.8)
for x, c, col in zip(xs, cols, (INK, TF, SG, INK)):
    ax.text(x, -0.3, c, fontsize=13, fontweight="bold", color=INK, ha="left" if x == 0 else "left")
    if col != INK:
        ax.add_patch(FancyBboxPatch((x - 0.022, -0.47), 0.012, 0.36, boxstyle="round,pad=0,rounding_size=0.003", fc=col, ec="none"))
ax.plot([0, 1], [0.15, 0.15], color=INK2, lw=0.9)
for i, (row, k) in enumerate(zip(cells, kinds)):
    y = i + 0.75
    if k == "head":
        ax.text(0, y, row[0], fontsize=12.5, fontweight="bold", color=INK2); continue
    ax.plot([0, 1], [y + 0.45, y + 0.45], color=GRID, lw=0.7)
    win = row[3].startswith("+") or (row[3].startswith("SGLang") is False and not row[3].startswith("-"))
    for j, (x, txt) in enumerate(zip(xs, row)):
        bold = (j == 1 and row[3].startswith("+")) or (j == 2 and (row[3].startswith("-") or row[3].startswith("SGLang")))
        ax.text(x, y, txt, fontsize=12.5, color=INK, fontweight="bold" if bold else "normal")
fig2.text(0.03, 0.93, "Qwen3.8-27B on one DGX Spark, same weights and drafter", fontsize=17, fontweight="bold")
fig2.text(0.03, 0.885, "Bold = the faster engine for that row.  RadixArk NVFP4 + incoai DFlash2, thinking off, 2026-11-18.",
          fontsize=11, color=INK2)
fig2.savefig(HERE / "tf-vs-sglang-table.png", facecolor=SURF)
print("wrote", HERE / "tf-vs-sglang.png", HERE / "tf-vs-sglang-table.png")
