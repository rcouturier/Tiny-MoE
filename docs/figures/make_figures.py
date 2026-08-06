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


# ---------- Fig 3: the branch width ratio sets the split ----------
# (ratio I/I_shared, minimum shared share, E) for every configuration measured.
RUNS = [(2.0, 0.25, 4), (2.0, 0.28, 8), (2.0, 0.37, 8),
        (1.0, 0.47, 8), (1.0, 0.53, 16), (1.0, 0.55, 32), (1.0, 0.56, 4),
        (0.5, 0.63, 16),
        (0.25, 0.72, 32), (0.25, 0.79, 8), (0.25, 0.81, 8)]
E_COLOR = {4: ORANGE, 8: BLUE, 16: GREEN, 32: VERM}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
style(a1)
style(a2)

# panel (a): every run, ordered by the width ratio; bands per ratio level do not overlap
for r in sorted({r for r, _, _ in RUNS}):
    vals = [s for rr, s, _ in RUNS if rr == r]
    a1.plot([r, r], [min(vals), max(vals)], color=GRID, lw=6, solid_capstyle="round", zorder=1)
for e in (4, 8, 16, 32):
    pts = [(r, s) for r, s, ee in RUNS if ee == e]
    a1.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=E_COLOR[e],
               zorder=3, label=f"E={e}", edgecolor="white", linewidth=0.7)
a1.axhline(0.5, color=MUTED, ls=":", lw=1, zorder=2)
a1.annotate("balanced (0.5)", (0.235, 0.515), color=MUTED, fontsize=8)
a1.set_xscale("log", base=2)
a1.set_xticks([0.25, 0.5, 1.0, 2.0])
a1.set_xticklabels(["0.25", "0.5", "1", "2"])
a1.set_xlabel("branch width ratio  $I$ / $I_{shared}$")
a1.set_ylabel("minimum shared share")
a1.set_title("The width ratio orders every run")
a1.set_ylim(0.15, 0.9)
a1.legend(loc="lower left", frameon=False, fontsize=8, ncol=2, handletextpad=0.2)

# panel (b): the decisive contrast — routed branch byte-for-byte identical
IS = ["512", "1024", "4096"]
contrast = [0.28, 0.47, 0.81]
xpos = np.arange(3)
a2.bar(xpos, contrast, color=[GREEN if v < 0.5 else BLUE for v in contrast], width=0.6, zorder=3)
a2.axhline(0.5, color=MUTED, ls=":", lw=1)
for x, v in zip(xpos, contrast):
    a2.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=9)
a2.set_xticks(xpos)
a2.set_xticklabels(IS)
a2.set_xlabel("shared expert width $I_{shared}$")
a2.set_ylabel("minimum shared share")
a2.set_title("Routed branch identical (E=8, $I$=1024)")
a2.set_ylim(0, 0.95)
save(fig, "fig3_ratio")


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

# ---------- Fig 5: scale robustness (8L vs 16L) ----------
tok_s = np.array([98, 197, 295, 393, 491, 590, 688, 786, 885, 983])
share_8L = np.array([0.81, 0.60, 0.50, 0.47, 0.47, 0.48, 0.48, 0.49, 0.50, 0.50])
share_16L = np.array([0.84, 0.65, 0.58, 0.55, 0.54, 0.56, 0.56, 0.58, 0.59, 0.59])
lr_labels = ["x1.0", "x0.3", "x0.1"]
share98_8L = [0.80, 0.56, 0.14]
share98_16L = [0.82, 0.64, 0.21]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
style(a1); style(a2)
# panel (a): F8 crossover, scale dependence
a1.axhline(0.5, color=MUTED, ls=":", lw=1)
a1.plot(tok_s, share_8L, color=BLUE, lw=2, marker="o", ms=5, zorder=3, label="8 layers (134M)")
a1.plot(tok_s, share_16L, color=ORANGE, lw=2, marker="s", ms=5, zorder=3, label="16 layers (251M)")
a1.set_xlabel("training tokens (millions)")
a1.set_ylabel("shared branch share")
a1.set_title("Crossover is scale-dependent")
a1.set_ylim(0.4, 0.9)
a1.legend(loc="upper right", frameon=False, fontsize=9)
# panel (b): F9 mechanism reproduces at scale
x = np.arange(3); w = 0.38
a2.bar(x - w/2, share98_8L, w, color=BLUE, zorder=3, label="8 layers")
a2.bar(x + w/2, share98_16L, w, color=ORANGE, zorder=3, label="16 layers")
a2.set_xticks(x); a2.set_xticklabels(lr_labels)
a2.set_xlabel("shared-expert LR multiplier")
a2.set_ylabel("shared share at 98M tokens")
a2.set_title("Mechanism reproduces at 2x depth")
a2.set_ylim(0, 0.95)
a2.legend(loc="upper right", frameon=False, fontsize=9)
save(fig, "fig5_scale")


# ---------- Fig 6: hysteresis — a temporary handicap permanently reallocates ----------
tok_h = np.array([98, 197, 295, 393, 491, 590, 688, 786, 885, 983])
base_16 = np.array([0.84, 0.65, 0.58, 0.55, 0.54, 0.56, 0.56, 0.58, 0.59, 0.59])
ramp_16 = np.array([0.42, 0.23, 0.15, 0.14, 0.16, 0.17, 0.19, 0.20, 0.21, 0.21])
static_16 = np.array([0.21, 0.13, 0.14, 0.15, 0.16])  # run stopped at 491M

fig, ax = plt.subplots(figsize=(5.6, 3.4))
style(ax)
ax.axhline(0.5, color=MUTED, ls=":", lw=1, zorder=1)
ax.annotate("balanced (0.5)", (105, 0.515), color=MUTED, fontsize=8)
ax.axvspan(60, 295, color=GRID, alpha=0.45, zorder=0, lw=0)
ax.annotate("shared LR ramped\n$\\times$0.1 $\\rightarrow$ $\\times$1.0", (175, 0.86), color=MUTED,
            fontsize=8, ha="center")
ax.plot(tok_h, base_16, color=BLUE, lw=2, marker="o", ms=5, zorder=3)
ax.plot(tok_h, ramp_16, color=VERM, lw=2, marker="D", ms=5, zorder=3)
ax.plot(tok_h[:5], static_16, color=GREEN, lw=2, ls="--", marker="s", ms=5, zorder=3)
for lab, c, x, y, off in [("baseline $\\times$1.0", BLUE, tok_h[-1], base_16[-1], (7, 0)),
                          ("ramp 0.1$\\rightarrow$1.0", VERM, tok_h[-1], ramp_16[-1], (7, 0)),
                          ("static $\\times$0.1", GREEN, tok_h[2], static_16[2], (-2, -16))]:
    ax.annotate(lab, (x, y), xytext=off, textcoords="offset points",
                color=c, va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("shared branch share of ablation impact")
ax.set_title("The reallocation does not undo (16 layers, 251M)")
ax.set_xlim(60, 1180)
ax.set_ylim(0.03, 0.95)
save(fig, "fig6_hysteresis")

print("all figures written to", HERE)
