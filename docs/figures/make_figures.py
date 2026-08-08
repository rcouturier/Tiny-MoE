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
# Dose sequence: darker = longer handicap (ordinal, colourblind-safe Blues ramp).
DOSE = [  # (ramp_end step, tokens at ramp end, trajectory over tok_h, colour)
    (3000,   98, [0.44, 0.25, 0.24, 0.24, 0.26, 0.28, 0.30, 0.32, 0.33, 0.33], "#9ECAE1"),
    (6000,  197, [0.44, 0.18, 0.16, 0.17, 0.19, 0.21, 0.22, 0.24, 0.25, 0.26], "#6BAED6"),
    (9000,  295, [0.42, 0.24, 0.15, 0.14, 0.15, 0.17, 0.18, 0.20, 0.21, 0.21], "#3182BD"),
    (15000, 491, [0.36, 0.24, 0.19, 0.16, 0.14, 0.15, 0.16, 0.18, 0.19, 0.19], "#08519C"),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
style(a1)
style(a2)

# panel (a): trajectories — the reallocation survives the handicap being lifted
a1.axhline(0.5, color=MUTED, ls=":", lw=1, zorder=1)
a1.annotate("balanced (0.5)", (105, 0.515), color=MUTED, fontsize=8)
a1.plot(tok_h, base_16, color=VERM, lw=2, marker="o", ms=4, zorder=4)
a1.annotate("no handicap", (tok_h[-1], base_16[-1]), xytext=(-4, 9), textcoords="offset points",
            color=VERM, ha="right", fontsize=8, fontweight="bold")
for _, tk, traj, c in DOSE:
    a1.plot(tok_h, traj, color=c, lw=1.8, marker="D", ms=3.5, zorder=3,
            label=f"{tk}M")
    a1.plot([tk], [traj[[98, 197, 295, 491].index(tk)]], marker="|", ms=13, mew=2.2,
            color=c, zorder=5)
leg = a1.legend(loc="center right", frameon=False, fontsize=7.5, ncol=2,
                title="handicap held for", handlelength=1.4, columnspacing=1.0,
                handletextpad=0.4, labelspacing=0.3)
leg.get_title().set_fontsize(7.5)
leg.get_title().set_color(MUTED)
a1.set_xlabel("training tokens (millions)")
a1.set_ylabel("shared branch share")
a1.set_title("Longer handicap, lower final share")
a1.set_xlim(60, 1010)
a1.set_ylim(0.05, 0.92)

# panel (b): the dose-response itself — final share vs how long the handicap lasted
xs = [0] + [tk for _, tk, _, _ in DOSE]
ys = [base_16[-1]] + [traj[-1] for _, _, traj, _ in DOSE]
a2.plot(xs, ys, color=MUTED, lw=1.5, ls="-", zorder=2)
a2.scatter([0], [base_16[-1]], s=55, color=VERM, zorder=4, edgecolor="white", linewidth=0.8)
for (_, tk, traj, c) in DOSE:
    a2.scatter([tk], [traj[-1]], s=55, color=c, zorder=4, edgecolor="white", linewidth=0.8)
for x, y in zip(xs, ys):
    a2.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8.5)
a2.annotate("no handicap", (0, ys[0]), xytext=(6, -12), textcoords="offset points",
            color=VERM, fontsize=8, fontweight="bold")
ASYMPTOTE = 0.14  # static x0.1, never restored, at 983M tokens
a2.axhline(ASYMPTOTE, color=MUTED, ls="--", lw=1.2, zorder=1)
a2.annotate("never restored ($\\infty$): 0.14", (555, ASYMPTOTE), xytext=(0, 5),
            textcoords="offset points", color=MUTED, fontsize=8, ha="right")
a2.set_xlabel("handicap duration (M tokens)")
a2.set_ylabel("shared share at 983M tokens")
a2.set_title("Dose-response, at flat perplexity")
a2.set_xlim(-45, 570)
a2.set_ylim(0.08, 0.70)
save(fig, "fig6_hysteresis")


# ---------- Fig 7: does reallocation buy routed specialization? ----------
# Width lever, E=8 with the routed branch byte-identical (I=1024), at 983M tokens.
w_share = [0.82, 0.50, 0.34]
w_nmi = [0.0103, 0.0174, 0.0226]
w_lab = ["4096", "1024", "512"]
# Learning-rate lever at matched step (491M tokens), both depths.
lr16_share, lr16_nmi = [0.67, 0.56, 0.16], [0.0108, 0.0135, 0.0151]
lr8_share, lr8_nmi = [0.48, 0.52, 0.10], [0.0191, 0.0195, 0.0175]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
style(ax)
ax.plot(w_share, w_nmi, color=BLUE, lw=2, marker="o", ms=7, zorder=4,
        label="width lever (routed branch identical)")
for x, y, l, off in zip(w_share, w_nmi, w_lab, [(4, 10), (-6, -14), (-4, 10)]):
    ax.annotate("$I_{shared}$=" + l, (x, y), xytext=off, textcoords="offset points",
                ha="left" if off[0] > 0 else "right", fontsize=7.5, color=BLUE)
ax.plot(lr16_share, lr16_nmi, color=GREEN, lw=1.8, ls="--", marker="s", ms=5.5, zorder=3,
        label="LR lever, 16 layers")
ax.plot(lr8_share, lr8_nmi, color=VERM, lw=1.8, ls=":", marker="^", ms=6, zorder=3,
        label="LR lever, 8 layers")
ax.invert_xaxis()
ax.set_xlabel("shared branch share  (routed engagement increases $\\rightarrow$)")
ax.set_ylabel("routed domain NMI")
ax.set_title("Only the width lever reliably buys specialization")
ax.set_ylim(0.009, 0.025)
ax.legend(loc="upper left", frameon=False, fontsize=8)
save(fig, "fig7_specialization")

print("all figures written to", HERE)
