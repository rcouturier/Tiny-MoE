"""Generate the paper figures from the logged experimental numbers.
Colorblind-safe (Okabe-Ito) palette, one axis per panel, direct labels + legend,
recessive grid. Outputs PDF (paper) and PNG (preview) into this directory.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"))
    plt.close(fig)
    print("wrote", name)


# ---------- Fig 1: Finding 8 — branch importance vs tokens ----------
tok = np.array([98, 197, 295, 393, 491, 590, 688, 786, 885, 983])
sh = np.array([3.44, 2.37, 1.99, 1.93, 1.99, 2.07, 2.21, 2.35, 2.44, 2.45])
sh_s = np.array([.17, .17, .14, .18, .16, .16, .16, .17, .17, .18])
rt = np.array([0.82, 1.58, 1.98, 2.14, 2.25, 2.27, 2.38, 2.41, 2.49, 2.48])
rt_s = np.array([.03, .06, .06, .03, .01, .03, .03, .07, .10, .11])

fig, ax = plt.subplots(figsize=(5.2, 3.3))
style(ax)
for y, s, c, lab in [(sh, sh_s, BLUE, "remove shared expert"), (rt, rt_s, ORANGE, "remove all 8 routed")]:
    ax.fill_between(tok, y - s, y + s, color=c, alpha=0.15, zorder=1)
    ax.plot(tok, y, color=c, lw=2, marker="o", ms=5, zorder=3, label=lab)
ax.axvline(295, color=MUTED, ls="--", lw=1, zorder=2)
ax.annotate("crossover\n~295M", (295, 0.60), color=MUTED, fontsize=8, ha="center")
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("ablation impact  ΔPPL")
ax.set_title("Routed experts become load-bearing with training")
ax.set_xlim(60, 1010)
ax.set_ylim(0.4, 3.7)
ax.legend(loc="upper right", frameon=False, fontsize=9)
save(fig, "fig1_trajectory")


# ---------- Fig 2: Finding 9 — shared-LR shifts the crossover ----------
t2 = np.array([98, 197, 295, 393, 491])
cond = [("shared LR x1.0", BLUE, [0.80, 0.60, 0.51, 0.48, 0.48]),
        ("x0.3", ORANGE, [0.56, 0.49, 0.50, 0.52, 0.52]),
        ("x0.1", GREEN, [0.14, 0.10, 0.09, 0.10, 0.10])]
fig, ax = plt.subplots(figsize=(5.2, 3.3))
style(ax)
ax.axhline(0.5, color=MUTED, ls=":", lw=1, zorder=1)
ax.annotate("balanced (0.5)", (100, 0.515), color=MUTED, fontsize=8)
for lab, c, y in cond:
    ax.plot(t2, y, color=c, lw=2, marker="o", ms=5, zorder=3, label=lab)
    ax.annotate(lab, (t2[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                color=c, va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("shared branch share of ablation impact")
ax.set_title("Slowing the shared expert moves the crossover earlier")
ax.set_xlim(80, 620)
ax.set_ylim(0, 0.9)
save(fig, "fig2_mechanism")


# ---------- Fig 3: Finding 10 — non-monotonic #experts ----------
E = [4, 8, 16, 32]
minshare = [0.56, 0.48, 0.53, 0.55]
ppl = [3.28, 3.00, 2.94, 2.90]
xpos = np.arange(len(E))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.1))
for ax in (a1, a2):
    style(ax)
    ax.set_xticks(xpos)
    ax.set_xticklabels(E)
    ax.set_xlabel("number of routed experts")
bars = a1.bar(xpos, minshare, color=[GREEN if v < 0.5 else BLUE for v in minshare], width=0.6, zorder=3)
a1.axhline(0.5, color=MUTED, ls=":", lw=1)
for x, v in zip(xpos, minshare):
    a1.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
a1.set_ylabel("minimum shared share")
a1.set_ylim(0, 0.65)
a1.set_title("Only E=8 reaches balance")
a2.bar(xpos, ppl, color=VERM, width=0.6, zorder=3)
for x, v in zip(xpos, ppl):
    a2.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
a2.set_ylabel("final perplexity")
a2.set_ylim(2.7, 3.4)
a2.set_title("More experts, lower PPL")
save(fig, "fig3_experts")


# ---------- Fig 4: Finding 2 — load balancing controls specialization ----------
labels = ["0", "0.01", "0.1"]
nmi = [0.0338, 0.0085, 0.0010]
nmi_s = [0.0056, 0.0015, 0.0002]
imb = [0.187, 0.020, 0.018]
imb_s = [0.008, 0.001, 0.001]
xpos = np.arange(3)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.1))
for ax in (a1, a2):
    style(ax)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("aux-loss coefficient")
a1.bar(xpos, nmi, yerr=nmi_s, color=BLUE, width=0.6, zorder=3, capsize=3, ecolor=MUTED)
a1.set_ylabel("domain NMI(Expert; Domain)")
a1.set_title("Specialization")
a2.bar(xpos, imb, yerr=imb_s, color=ORANGE, width=0.6, zorder=3, capsize=3, ecolor=MUTED)
a2.set_ylabel("expert load imbalance")
a2.set_title("Load balance")
save(fig, "fig4_auxloss")

print("all figures written to", HERE)
