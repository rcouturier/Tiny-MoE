"""Generate the paper figures from the logged experimental numbers.

Figures are drawn at their final printed size (one IEEE column = 3.45 in) so
that no scaling happens in LaTeX and the type sizes below are the type sizes on
the page. Colourblind-safe (Okabe-Ito) palette, direct labels in preference to
legends, recessive grid. Outputs PDF (paper) and PNG (preview) into this
directory.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

COL = 3.45  # IEEE single column width, inches

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "lines.linewidth": 1.4, "lines.markersize": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(pad=0.25, w_pad=0.9)
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"))
    plt.close(fig)
    print("wrote", name)


TOK = np.array([98, 197, 295, 393, 491, 590, 688, 786, 885, 983])

# ---------- Fig 1: branch importance, and the crossover at two depths ----------
sh = np.array([3.44, 2.37, 1.99, 1.93, 1.99, 2.07, 2.21, 2.35, 2.44, 2.45])
sh_s = np.array([.17, .17, .14, .18, .16, .16, .16, .17, .17, .18])
rt = np.array([0.82, 1.58, 1.98, 2.14, 2.25, 2.27, 2.38, 2.41, 2.49, 2.48])
rt_s = np.array([.03, .06, .06, .03, .01, .03, .03, .07, .10, .11])
share_8L = np.array([0.81, 0.60, 0.50, 0.47, 0.47, 0.48, 0.48, 0.49, 0.50, 0.50])
share_16L = np.array([0.84, 0.65, 0.58, 0.55, 0.54, 0.56, 0.56, 0.58, 0.59, 0.59])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.62))
style(a1)
style(a2)

# (a) absolute ablation impact of each branch, 8 layers, four seeds
for y, s, c, lab, xy in [(sh, sh_s, BLUE, "shared", (688, 2.62)),
                         (rt, rt_s, ORANGE, "routed", (590, 1.86))]:
    a1.fill_between(TOK, y - s, y + s, color=c, alpha=0.15, zorder=1)
    a1.plot(TOK, y, color=c, marker="o", zorder=3)
    a1.annotate(lab, xy, color=c, fontsize=6.5, fontweight="bold", ha="center")
a1.axvline(295, color=MUTED, ls="--", lw=0.8, zorder=2)
a1.annotate("crossover", (330, 3.28), color=MUTED, fontsize=6, ha="left")
a1.set_xlabel("training tokens (M)")
a1.set_ylabel("ablation impact  $\\Delta$PPL")
a1.set_xlim(40, 1040)
a1.set_ylim(0.4, 3.75)
a1.set_xticks([200, 500, 800])
a1.set_title("(a) 8 layers", color=MUTED)

# (b) the same story as a share, at both depths
a2.axhline(0.5, color=MUTED, ls=":", lw=0.8, zorder=1)
a2.plot(TOK, share_8L, color=BLUE, marker="o", zorder=3)
a2.plot(TOK, share_16L, color=ORANGE, marker="s", zorder=3)
a2.annotate("16 layers", (983, 0.60), xytext=(-2, 5), textcoords="offset points",
            color=ORANGE, fontsize=6.5, fontweight="bold", ha="right")
a2.annotate("8 layers", (983, 0.50), xytext=(-2, -11), textcoords="offset points",
            color=BLUE, fontsize=6.5, fontweight="bold", ha="right")
a2.set_xlabel("training tokens (M)")
a2.set_ylabel("shared share  $s$")
a2.set_xlim(40, 1040)
a2.set_ylim(0.40, 0.90)
a2.set_xticks([200, 500, 800])
a2.set_title("(b) two depths", color=MUTED)
save(fig, "fig1_trajectory")


# ---------- Fig 2: the learning-rate handicap, at both depths ----------
t2 = np.array([98, 197, 295, 393, 491])
cond = [("$\\times1.0$", BLUE, [0.80, 0.60, 0.51, 0.48, 0.48]),
        ("$\\times0.3$", ORANGE, [0.56, 0.49, 0.50, 0.52, 0.52]),
        ("$\\times0.1$", GREEN, [0.14, 0.10, 0.09, 0.10, 0.10])]
share98_8L = [0.80, 0.56, 0.14]
share98_16L = [0.82, 0.64, 0.21]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.62))
style(a1)
style(a2)

# (a) trajectories under three multipliers, 8 layers
a1.axhline(0.5, color=MUTED, ls=":", lw=0.8, zorder=1)
a1.annotate("balanced", (110, 0.43), color=MUTED, fontsize=6)
for lab, c, y in cond:
    a1.plot(t2, y, color=c, marker="o", zorder=3)
    dy = {"$\\times1.0$": -5, "$\\times0.3$": 4}.get(lab, 0)
    a1.annotate(lab, (t2[-1], y[-1]), xytext=(4, dy), textcoords="offset points",
                color=c, va="center", fontsize=6.5, fontweight="bold")
a1.set_xlabel("training tokens (M)")
a1.set_ylabel("shared share  $s$")
a1.set_xlim(60, 720)
a1.set_ylim(0, 0.92)
a1.set_xticks([100, 300, 500])
a1.set_title("(a) 8 layers", color=MUTED)

# (b) the same manipulation at twice the depth, read at 98M tokens
x = np.arange(3)
w = 0.36
a2.bar(x - w / 2, share98_8L, w, color=BLUE, zorder=3, label="8 layers")
a2.bar(x + w / 2, share98_16L, w, color=ORANGE, zorder=3, label="16 layers")
a2.set_xticks(x)
a2.set_xticklabels(["$\\times1.0$", "$\\times0.3$", "$\\times0.1$"])
a2.set_xlabel("shared-expert LR multiplier")
a2.set_ylabel("$s$ at 98M tokens")
a2.set_ylim(0, 1.06)
a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
a2.legend(loc="upper right", frameon=False, fontsize=6, handlelength=1.1,
          handletextpad=0.4, borderaxespad=0.1)
a2.set_title("(b) both depths", color=MUTED)
save(fig, "fig2_mechanism")


# ---------- Fig 3: hysteresis — a temporary handicap reallocates durably ----------
base_16 = np.array([0.84, 0.65, 0.58, 0.55, 0.54, 0.56, 0.56, 0.58, 0.59, 0.59])
# Dose sequence: darker = longer handicap (ordinal, colourblind-safe Blues ramp).
DOSE = [  # (tokens at ramp end, trajectory over TOK, colour)
    (98,  [0.44, 0.25, 0.24, 0.24, 0.26, 0.28, 0.30, 0.32, 0.33, 0.33], "#9ECAE1"),
    (197, [0.44, 0.18, 0.16, 0.17, 0.19, 0.21, 0.22, 0.24, 0.25, 0.26], "#6BAED6"),
    (295, [0.42, 0.24, 0.15, 0.14, 0.15, 0.17, 0.18, 0.20, 0.21, 0.21], "#3182BD"),
    (491, [0.36, 0.24, 0.19, 0.16, 0.14, 0.15, 0.16, 0.18, 0.19, 0.19], "#08519C"),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.68))
style(a1)
style(a2)

# (a) trajectories; the tick marks where each handicap was lifted
a1.axhline(0.5, color=MUTED, ls=":", lw=0.8, zorder=1)
a1.plot(TOK, base_16, color=VERM, marker="o", ms=2.5, zorder=4)
a1.annotate("no handicap", (983, 0.59), xytext=(-2, 6), textcoords="offset points",
            color=VERM, ha="right", fontsize=6.5, fontweight="bold")
for tk, traj, c in DOSE:
    a1.plot(TOK, traj, color=c, lw=1.2, marker="D", ms=2.2, zorder=3)
    a1.plot([tk], [traj[[98, 197, 295, 491].index(tk)]], marker="|", ms=9, mew=1.8,
            color=c, zorder=6)
a1.annotate("98M", (983, 0.33), xytext=(-2, 4), textcoords="offset points",
            color="#6BAED6", ha="right", fontsize=6, fontweight="bold")
a1.annotate("491M", (983, 0.19), xytext=(-2, -11), textcoords="offset points",
            color="#08519C", ha="right", fontsize=6, fontweight="bold")
a1.set_xlabel("training tokens (M)")
a1.set_ylabel("shared share  $s$")
a1.set_xlim(40, 1040)
a1.set_ylim(0.05, 0.95)
a1.set_xticks([200, 500, 800])
a1.set_title("(a) 16 layers", color=MUTED)

# (b) the dose-response itself
xs = [0] + [tk for tk, _, _ in DOSE]
ys = [base_16[-1]] + [traj[-1] for _, traj, _ in DOSE]
a2.plot(xs, ys, color=MUTED, lw=1.0, zorder=2)
a2.scatter([0], [ys[0]], s=26, color=VERM, zorder=4, edgecolor="white", linewidth=0.6)
for tk, traj, c in DOSE:
    a2.scatter([tk], [traj[-1]], s=26, color=c, zorder=4, edgecolor="white", linewidth=0.6)
for i, (xv, yv) in enumerate(zip(xs, ys)):
    a2.annotate(f"{yv:.2f}", (xv, yv), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=6)
ASYMPTOTE = 0.14  # static x0.1, never restored, at 983M tokens
a2.axhline(ASYMPTOTE, color=MUTED, ls="--", lw=1.0, zorder=1)
a2.annotate("never lifted: 0.14", (0, ASYMPTOTE), xytext=(0, -9),
            textcoords="offset points", color=MUTED, fontsize=6, ha="left")
a2.set_xlabel("handicap duration (M)")
a2.set_ylabel("$s$ at 983M tokens")
a2.set_xlim(-60, 610)
a2.set_ylim(0.06, 0.72)
a2.set_xticks([0, 200, 400])
a2.set_title("(b) dose-response", color=MUTED)
save(fig, "fig3_hysteresis")


# ---------- Fig 4: the branch width ratio sets the split ----------
SH2048_MIN = 0.70  # E=8, I=1024, I_shared=2048 (camera-ready run, two seeds)
# (ratio I/I_shared, minimum shared share, E) for every configuration measured.
RUNS = [(2.0, 0.25, 4), (2.0, 0.28, 8), (2.0, 0.37, 8),
        (1.0, 0.47, 8), (1.0, 0.53, 16), (1.0, 0.55, 32), (1.0, 0.56, 4),
        (0.5, 0.63, 16), (0.5, SH2048_MIN, 8),
        (0.25, 0.72, 32), (0.25, 0.79, 8), (0.25, 0.81, 8)]
E_COLOR = {4: ORANGE, 8: BLUE, 16: GREEN, 32: VERM}
SEED = 0.03  # largest seed spread on s measured across the two-seed configurations

fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.62))
style(a1)
style(a2)

# (a) every run, ordered by the width ratio; the bands per ratio do not overlap
for r in sorted({r for r, _, _ in RUNS}):
    vals = [s for rr, s, _ in RUNS if rr == r]
    a1.plot([r, r], [min(vals), max(vals)], color=GRID, lw=4.5,
            solid_capstyle="round", zorder=1)
for e in (4, 8, 16, 32):
    pts = [(r, s) for r, s, ee in RUNS if ee == e]
    a1.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=SEED, fmt="none",
                ecolor=E_COLOR[e], elinewidth=0.6, capsize=1.2, alpha=0.7, zorder=2)
    a1.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, color=E_COLOR[e],
               zorder=3, label=f"$E$={e}", edgecolor="white", linewidth=0.5)
a1.axhline(0.5, color=MUTED, ls=":", lw=0.8, zorder=2)
a1.set_xscale("log", base=2)
a1.set_xticks([0.25, 0.5, 1.0, 2.0])
a1.set_xticklabels(["0.25", "0.5", "1", "2"])
a1.set_xlabel("width ratio  $I/I_{\\mathrm{shared}}$")
a1.set_ylabel("minimum shared share")
a1.set_ylim(0.12, 0.93)
a1.legend(loc="lower left", frameon=False, fontsize=6, ncol=2, handletextpad=0.05,
          columnspacing=0.6, borderaxespad=0.2, labelspacing=0.25)
a1.set_title("(a) twelve runs", color=MUTED)

# (b) the decisive contrast: the routed branch is identical across all four
contrast = [0.28, 0.47, SH2048_MIN, 0.81]
xpos = np.arange(4)
a2.bar(xpos, contrast, color=[GREEN if v < 0.5 else BLUE for v in contrast],
       width=0.6, zorder=3)
a2.axhline(0.5, color=MUTED, ls=":", lw=0.8, zorder=1)
for xv, v in zip(xpos, contrast):
    a2.annotate(f"{v:.2f}", (xv, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=6.5)
a2.set_xticks(xpos)
a2.set_xticklabels(["512", "1024", "2048", "4096"])
a2.set_xlabel("shared width  $I_{\\mathrm{shared}}$")
a2.set_ylim(0, 0.95)
a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
a2.set_title("(b) routed identical", color=MUTED)
save(fig, "fig4_ratio")


# ---------- Fig 5: does reallocation buy routed specialization? ----------
# Width lever, E=8 with the routed branch byte-identical (I=1024), at 983M tokens.
w_share = [0.82, 0.50, 0.34]
w_nmi = [0.0103, 0.0174, 0.0226]
w_lab = ["4096", "1024", "512"]
# Learning-rate lever at matched step (491M tokens), both depths.
lr16_share, lr16_nmi = [0.67, 0.56, 0.16], [0.0108, 0.0135, 0.0151]
lr8_share, lr8_nmi = [0.48, 0.52, 0.10], [0.0191, 0.0195, 0.0175]

fig, ax = plt.subplots(figsize=(COL, 2.05))
style(ax)
ax.plot(w_share, w_nmi, color=BLUE, marker="o", ms=4.5, zorder=4,
        label="width lever, $I_{\\mathrm{shared}}$ 4096$\\rightarrow$512"
              " (routed identical)")
ax.plot(lr16_share, lr16_nmi, color=GREEN, lw=1.2, ls="--", marker="s", ms=3.5,
        zorder=3, label="LR lever, 16 layers, $m$ 1.0$\\rightarrow$0.1")
ax.plot(lr8_share, lr8_nmi, color=VERM, lw=1.2, ls=":", marker="^", ms=4,
        zorder=3, label="LR lever, 8 layers, $m$ 1.0$\\rightarrow$0.1")
ax.invert_xaxis()
ax.set_xlabel("shared branch share $s$   (routed engagement $\\rightarrow$)")
ax.set_ylabel("routed domain NMI")
ax.set_ylim(0.0086, 0.0250)
ax.set_xlim(0.87, 0.05)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False,
          fontsize=6, handlelength=1.6, handletextpad=0.5, borderaxespad=0,
          labelspacing=0.3)
save(fig, "fig5_specialization")

print("all figures written to", HERE)
