# Results — Expert Specialization in a Small Multi-Domain MoE

*Working draft. Numbers are from the pilot-scale study on a single RTX A6000; some
sections marked "pending" are being filled by the running interaction/multi-axis/scale
experiments.*

## Setup

We train a 134M-parameter decoder-only MoE (8 layers, 8 routed experts + 1 shared expert,
top-2 routing, MLA attention) from scratch on a three-domain mixture with **known per-token
labels**: FineWeb-Edu (`web`), Cosmopedia-v2 (`edu`), OpenWebMath (`math`). All pilots use a
single-GPU plain-PyTorch trainer (3,000 steps ≈ 49M tokens unless noted, batch 32, seq 512,
cosine LR to zero). Each condition is replicated over **4 seeds**.

Analysis is run in inference mode (no capacity dropping) on held-out per-domain text
(80k tokens/domain, packed within domain so every token has one domain label).

**Metrics.** *Specialization:* soft (gate-weighted, top-2) normalized mutual information
`NMI(Expert; Domain) = I(E;D)/H(D)`, averaged over layers. *Balance:* mean per-layer standard
deviation of top-1 expert load. *Causal importance:* per-domain ΔPPL from ablating a whole
branch. **Controls:** a label-permutation null (shuffled domain labels) and an untrained-model
floor; between-condition **exact permutation tests** on seed-level NMI.

All reported NMI values sit far above the permutation-null (≈ 0.000 in every layer), so the
measured domain information is real, though (see below) modest in absolute size.

## Finding 1 — The shared expert is the workhorse; routed experts are largely redundant

Using a *fair, like-for-like* causal test (ablate the **entire** routed branch — all 8 experts
— vs. ablate the **entire** shared branch), on the default `aux=1e-2` models (4 seeds):

| removed branch | ΔPPL (mean ± std) |
|---|---|
| all 8 routed experts | **+0.55 ± 0.02** |
| the single shared expert | **+4.00 ± 0.15** |

Removing all eight routed experts costs almost nothing; removing the lone shared expert costs
~7× more. The shared expert carries the bulk of the feed-forward computation, and the routed
experts are largely redundant when it is present. When the shared expert is absent, the routed
branch does step up (removing it then costs +8.0 PPL), i.e. the capability is substitutable.

> Single-expert ablation is misleading here: removing one routed expert costs ≈ 0 because top-2
> routing reroutes to the survivors. The whole-branch comparison is the correct causal probe.

## Finding 2 — The load-balancing loss causally controls domain specialization

Sweeping the auxiliary load-balancing coefficient (4 seeds each), everything else fixed:

| `aux_coef` | soft NMI(E;D) | load imbalance | base PPL |
|---|---|---|---|
| **0** | **0.0338 ± 0.0056** | 0.187 ± 0.008 | 4.49 |
| 1e-2 (default) | 0.0085 ± 0.0015 | 0.020 ± 0.001 | 4.47 |
| 1e-1 | 0.0010 ± 0.0002 | 0.018 ± 0.001 | 4.47 |

The effect is **monotonic** and every pairwise permutation test is significant at the 4-vs-4
floor (p = 0.029, i.e. perfect rank separation between all three levels). Removing the balancing
loss multiplies domain-NMI ~4× and load imbalance ~9×, **at no perplexity cost** at this scale.
Balance and specialization are in direct tension, set by a single coefficient.

**Mechanism.** Without balancing pressure, experts are free to become both imbalanced *and*
domain-niched; the two rise together.

## Negative result — Removing the shared expert does *not* sharpen specialization (retracted)

At 3 seeds, no-shared models showed cleanly higher NMI (0.0100 vs 0.0078, perfect rank
separation) — suggesting "removing the shared expert sharpens routed specialization." This
**did not replicate**: at 4 seeds (default `aux=1e-2`) the difference collapses to
0.0093 ± 0.0013 vs 0.0085 ± 0.0015, permutation p = 0.543. Worse for the naive hypothesis,
once balancing is turned off (`aux=0`, where the effect is measurable) the sign **reverses**:
removing the shared expert *lowers* specialization (Finding 4). The original D1 hypothesis is
not merely unsupported — it is backwards. The 3-seed separation was small-sample noise, caught
by replication.

## Finding 3 — Experts specialize by surface form first, domain second, position not at all

soft NMI(Expert; axis) on the `aux=0` models (4 seeds; permutation control in parens):

| axis | soft NMI |
|---|---|
| **surface class** (word / number / punct / space) | **0.050 ± 0.006** (ctl 0.0001) |
| domain | 0.034 ± 0.006 (ctl 0.0000) |
| token frequency | 0.028 ± 0.003 (ctl 0.0001) |
| sequence position | 0.0003 ± 0.0000 (ctl 0.0000) |

Experts organize **primarily by token surface form**; domain is a secondary axis; position
plays no role. This reproduces, in a controlled labelled setting, the Mixtral observation that
experts specialize more by token/syntax than by topic.

**Disentanglement (conditional MI).** Because these axes are correlated (math carries more
digits/symbols), we check whether one explains the others via NMI(Expert; X | Y). Domain
routing information is **fully retained** under conditioning: `I(E;domain|surface)` = 100% of
`I(E;domain)` (and 100% conditioning on frequency). Surface is likewise ~unattenuated by domain
(105%). Only frequency is partly absorbed by surface (82%). So surface, domain, and frequency
are **separable axes** — domain specialization is genuine and *not* an artifact of surface-form
correlation, and surface is the largest independent axis.

## Finding 4 — Shared and routed experts are *complements* for specialization

2×2 {shared, no-shared} × {aux=0, aux=1e-2}, 4 seeds. At `aux=1e-2` there is no difference
(both crushed by balancing, p=0.543). At **`aux=0`**, where specialization is visible:

| | routed soft NMI | remove all routed | remove shared | PPL |
|---|---|---|---|---|
| **with shared** | **0.034 ± 0.006** | +0.46 | +4.81 | 4.49 |
| no shared | 0.015 ± 0.001 | +6.50 | — | 4.60 |

Permutation test Δ=+0.019, **p=0.029**. **Removing the shared expert *lowers* routed
specialization (0.034 → 0.015), the opposite of the naive D1 hypothesis.** The shared expert
acts as a common-capability backstop that *frees* routed experts to specialize (and to become
imbalanced: load imbalance 0.187 with shared vs 0.109 without). This is the DeepSeekMoE
intuition — shared and routed are **complements, not substitutes** — but it is only visible
once the load-balancing loss is off. Note that shared-branch computational dominance persists
here too (+4.81 vs +0.46).

## Finding 5 — Scale: shared dominance strengthens; domain-NMI does not clearly grow

`aux=0`, training budgets 3k / 6k / 12k steps (≈ 49M / 98M / 197M tokens; 2 seeds):

| budget | PPL | routed soft NMI | remove shared ΔPPL |
|---|---|---|---|
| 3k | 4.48 | 0.033 ± 0.005 | +4.91 |
| 6k | 4.04 | 0.034 ± 0.001 | +5.34 |
| 12k | 3.59 | 0.041 ± 0.005 | +5.89 |

Perplexity drops as expected. **Shared-expert dominance grows monotonically** with more tokens
(+4.9 → +5.9). **Domain-NMI does not increase significantly** (pairwise p ≥ 0.33 over 2 seeds),
suggesting the modest domain specialization is partly intrinsic to this scale rather than mere
undertraining. (Two seeds, narrow token range — indicative, not conclusive.)

## Finding 6 — Shared dominance is not a granularity artifact (robust over #experts and top-k)

Sweeping the number of routed experts (top-2, `aux=1e-2`, 2 seeds; monotonic trends, tiny std,
though 2-seed permutation p floors at 0.333):

| #experts | remove all routed | remove shared | soft NMI | load imbalance | PPL |
|---|---|---|---|---|---|
| 4 | +0.43 | +4.41 | 0.0047 | 0.037 | 4.51 |
| 8 | +0.54 | +3.97 | 0.0071 | 0.021 | 4.46 |
| 16 | +0.69 | +3.54 | 0.0118 | 0.014 | 4.43 |
| 32 | +0.78 | +3.10 | 0.0176 | 0.009 | 4.42 |

**Shared dominance persists at every granularity** — even with 32 routed experts, removing the
shared expert costs +3.1 PPL vs +0.78 for removing all 32 routed experts (~4×). So Finding 1 is
*not* an artifact of too many experts for too little data. Secondary trends: the routed branch
grows gradually more useful with more experts (+0.43→+0.78) and more specialized (NMI ~4×), with
lower per-expert imbalance but higher collective specialization.

Top-k (8 experts): at **top-1**, domain specialization collapses to ≈ the control (soft NMI
0.0007) and the routed branch is even less important (+0.16) while the shared expert dominates
more (+6.2). Top-2 is needed for routed experts to contribute at all.

## Finding 7 — The shared-expert monopoly is benign, not a pathology (falsified hypothesis)

We hypothesized that the shared expert *progressively monopolizes* computation and *starves*
the routed experts — a training pathology. Two experiments (3 seeds, 6000 steps each):

**The race (baseline, aux=1e-2).** The shared branch dominates from very early (93% of total
ablation impact by step 1000) and stays dominant, but its relative share *decreases* slightly
over training (0.93 → 0.82) — routed experts slowly catch up, they are not increasingly
starved. Routed NMI is flat (~0.007) throughout. (This tempers an earlier coarse-sweep
impression that shared dominance strictly grows.)

**The intervention (per-token shared-branch dropout, p=0.3).** This mechanically breaks the
monopoly: at step 6000 the shared branch becomes irrelevant (remove-shared ΔPPL +0.08) and the
routed branch carries everything (remove-routed +5.63). But the payoff is null:

| | soft NMI | base PPL |
|---|---|---|
| baseline | 0.0072 ± 0.0019 | **4.02** |
| shared-dropout 0.3 | 0.0085 ± 0.0003 | 4.05 |

Specialization does not meaningfully increase (p=0.60) and PPL is slightly *worse* (+0.03,
consistent across seeds). **Forcing the computation out of the shared expert reproduces the same
loss without unlocking specialization.** The shared/routed split is therefore a near-neutral
reparametrization: the shared expert is simply the convenient locus for common computation, not
a mechanism suppressing otherwise-useful routed experts. The hypothesis is falsified. Practical
implication: at this scale the routed experts are over-parametrized regardless of the shared
expert — their under-use reflects a lack of *needed* specialized computation, not suppression.

## Finding 8 — Shared dominance / routed redundancy is largely an undertraining artifact

Branch importance as a function of tokens seen, over a **constant-batch** (64), **4-seed**
training trajectory (mean ± std; remove = ablate the whole branch, measure ΔPPL on held-out):

| tokens | held-out PPL | remove shared | remove all 8 routed | routed NMI | shared share |
|---|---|---|---|---|---|
| 98M | 4.18 | **+3.44 ± 0.17** | +0.82 ± 0.03 | 0.0121 | **0.81** |
| 197M | 3.63 | +2.37 ± 0.17 | +1.58 ± 0.06 | 0.0148 | 0.60 |
| **295M** | 3.41 | +1.99 ± 0.14 | +1.98 ± 0.06 | 0.0161 | **0.50** |
| 491M | 3.20 | +1.99 ± 0.16 | +2.25 ± 0.01 | 0.0168 | 0.47 |
| 688M | 3.07 | +2.21 ± 0.16 | +2.38 ± 0.03 | 0.0177 | 0.48 |
| 983M | 3.00 | +2.45 ± 0.18 | +2.48 ± 0.11 | 0.0174 | 0.50 |

At **98M tokens** (the regime of our pilots and of many toy studies) the shared expert carries
**81%** of total ablation impact — the routed experts look nearly redundant. The crossover to
balance occurs around **~295M tokens**, after which the shared and routed branches are equally
important (share ≈ 0.5), stable through 1B. All 4 seeds follow the same trajectory with tight
error bars; routed specialization grows in parallel (NMI 0.0008 → 0.017, plateauing ~300M).

**The "shared monopoly" and "routed redundancy" of Findings 1/6 are undertraining artifacts.**
This is the study's most useful, statistically-supported takeaway: small-scale MoE
interpretability trained on too few tokens (as ours was initially, and as many pilot studies
are) systematically overstates expert redundancy and shared-expert dominance — here the effect
inverts within ~300M tokens.

## Distillation (preliminary, single-seed)

On-the-fly KD from a Mistral-7B teacher into the Tiny-MoE (α=0.9, τ=2.0), 98M tokens, compared
against a batch/step-matched from-scratch CE control:

| 98M-token model | soft NMI | PPL | remove routed | remove shared | ratio |
|---|---|---|---|---|---|
| CE control (batch 24) | 0.0063 | 3.92 | +1.03 | +3.85 | 3.7× |
| distilled (batch 24) | 0.0095 | 4.09 | +1.74 | +4.40 | 2.5× |

Against the matched control, distillation appears to **engage the routed branch more** (+1.03→
+1.74) and reduce the shared/routed imbalance (3.7×→2.5×), with slightly more domain NMI — but
at a small next-token PPL cost (α=0.9 optimizes teacher-matching, not likelihood). **Single seed
per condition; not statistically established.** We also found held-out NMI/PPL are sensitive to
batch size at fixed tokens (CE batch-24 vs batch-64: NMI 0.0063 vs 0.0141), an uncontrolled
confound in the earlier sweeps — see Limitations.

## Finding 9 — The monopoly is a gradient-exposure race, controllable by one knob

*Why* does the shared expert dominate early (Finding 8)? Hypothesis: a gradient-exposure race —
the shared expert receives gradient from 100% of tokens, each routed expert from only ~k/E
(=25%), so the shared expert wins the early optimization race and the residual stream leans on
it; routed experts activate only once it saturates. **Test:** slow the shared expert's learning
(its own optimizer group at `lr × mult`) and see whether the routed-activation crossover shifts.
Two seeds, 15k steps; values at **98M tokens** (the undertrained regime):

| shared LR | remove routed ΔPPL | remove shared ΔPPL | shared share | PPL |
|---|---|---|---|---|
| ×1.0 (baseline) | +0.85 | +3.46 | **0.80** | 4.16 |
| ×0.3 | +1.94 | +2.48 | 0.56 | 4.10 |
| ×0.1 | **+4.46** | +0.72 | **0.14** | 4.10 |

Slowing the shared expert makes the routed branch load-bearing **dramatically earlier** — a 5×
increase in routed importance at fixed tokens — shifting the crossover from ~295M tokens
(baseline) to <100M (×0.3) to before training even settles (×0.1, routed dominate from the
start). The effect is monotonic, dose-dependent, tight error bars. **Autonomous routed
maturation is refuted: the routed experts were being out-competed, not merely immature.**

Crucially, **held-out PPL is unchanged** across conditions (if anything slightly better at ×0.1/
×0.3). The shared expert is therefore not *necessary* for this computation — the model defaults
to placing it there because that expert trains fastest. **A single knob (shared-expert LR)
reallocates computation to the routed experts at no quality cost**, controlling when/whether a
small MoE actually uses its experts — useful for interpretability, expert pruning, or modularity.

Together, Findings 8+9 give observation → mechanism → control: routed "redundancy" is a
gradient-race artifact with a known threshold and a causal lever.

## Finding 10 — #experts dependence is non-monotonic (RETRACTED, see Finding 11)

> **Retracted 2026-08-06.** Both halves of this finding were confounded. The sweep held
> per-expert width fixed, so total routed capacity and total parameter count grew with E while
> the branch **width ratio** stayed at 1.0 throughout — and Finding 11 shows the width ratio is
> what governs the split. All four points below therefore sit inside the single ratio-1.0 band
> (0.47–0.56), and the "Goldilocks optimum at E=8" is within that band's residual spread. The
> perplexity gain with E was a parameter-count effect: at matched parameters perplexity is flat
> (2.99 / 3.00 / 3.00 / 3.02 for E = 4/8/16/32). Expert count is not what sets the split.
> The original text is kept below for the record.

The gradient-race mechanism (Finding 9) predicts the crossover token-count should scale ∝ E/k
(each routed expert sees k/E of the gradient). We tested E ∈ {4,8,16,32} (top-2, batch 64,
full trajectories). The minimum shared "share" of ablation impact reached during training:

| #experts | min shared share | crosses 0.5? | final PPL |
|---|---|---|---|
| 4 | 0.56 | no (plateau) | 3.28 |
| **8** | **0.48** | **yes** | 3.00 |
| 16 | 0.53 | no | 2.94 |
| 32 | 0.55 | no | 2.90 |

The simple law **does not hold**: only E=8 fully crosses to routed-balance; the relationship is
**non-monotonic** with a "Goldilocks" optimum at E=8. The mechanism's prediction holds on the
high-E side (E=8→16→32: shared grows more dominant, 0.48→0.53→0.55, as each routed expert's
gradient share k/E shrinks) but **fails at E=4**, which stays shared-dominant despite the highest
per-expert gradient share — implicating a second factor, total routed capacity (E=4 has half
E=8's routed params). There is also a practical tension: more experts give **better perplexity**
(2.90 at E=32) but **more shared-dominated computation** (routed experts individually
under-used). (Caveat: E=4 trained to 491M vs 983M for others, but plateaued flat by 295M.)

Net: the #experts dependence is a nuanced secondary result, not a clean law. The core
contribution remains Findings 8 (observation) + 9 (mechanism + control).

## Finding 11 — The branch split is set by the routed:shared width ratio, not by #experts

If not E (Finding 10, retracted), what determines *where* the shared/routed split lands? Eleven
8-layer configurations, top-2, 983M tokens, 1–2 seeds each. Write `I` for each routed expert's
intermediate width and `I_shared` for the shared expert's. **Two contrasts disentangle the
candidates, both at fixed E=8, k=2** (so expert count, routing and per-expert gradient share are
constant):

**Varying routed width** (`I_shared` = 1024 throughout):

| `I` | 256 | 1024 | 2048 |
|---|---|---|---|
| min shared share | 0.79 | 0.47 | 0.37 |
| params | 58.2M | 133.7M | 234.4M |

**Varying shared width** (`I` = 1024 throughout — the routed branch is byte-for-byte identical,
same active capacity k·I = 2048):

| `I_shared` | 512 | 1024 | 4096 |
|---|---|---|---|
| min shared share | **0.28** | 0.47 | **0.81** |
| params | 127.4M | 133.7M | 171.5M |

**The second contrast is decisive**: nothing about the routed branch changes and the split still
moves by 0.53. **Absolute routed capacity does not govern the split.** What both contrasts share
is the ratio `I / I_shared`. Grouping every configuration measured by that ratio — including the
four points of the retracted E-sweep, all of which sat at ratio 1.0 — gives non-overlapping bands:

| ratio `I`/`I_shared` | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| min shared share | 0.72 / 0.79 / 0.81 | 0.63 | 0.47 / 0.53 / 0.55 / 0.56 | 0.25 / 0.28 / 0.37 |

The ordering holds across **8× in ratio, 8× in expert count (E ∈ {4,…,32}) and 4× in parameter
count (58M–234M)**. Within a band the residual spread is ~0.1 and is *not* consistently explained
by E: at ratio 2.0 more experts *raise* the share (E=4 → 0.25 vs E=8 → 0.37), at ratio 0.25 more
experts *lower* it (E=32 → 0.72 vs E=8 → 0.79). Seed noise on this quantity is ~0.02 (measured by
adding seed 1 to `e8i256` / `e8i2048`), an order of magnitude below the effect.

At matched total routed capacity (E·I = 8192, equal parameters), E moves neither the split much
nor perplexity at all: final PPL 2.99 / 3.00 / 3.00 / 3.02 for E = 4/8/16/32.

**And it is nearly free.** Across the seven 127M–234M configurations the minimum shared share
spans 0.25 → 0.81 while final PPL spans only 2.95 → 3.02. Combined with Finding 9: the gradient
race decides *who gets there first*, the width ratio decides *how much each branch can absorb* —
and where a small MoE puts its feed-forward computation is close to a free design choice.

Full per-checkpoint tables, provenance and reproduction commands: `docs/branch-split-runs.md`.

## Finding 12 — A temporary shared-LR handicap permanently reallocates computation (hysteresis)

Finding 9's knob was applied for all of training. If the split is decided by an *early* race, a
handicap covering only that race should suffice. Test on the 16-layer model (251M params,
1 seed): ramp the shared-expert LR multiplier from ×0.1 up to ×1.0 by step 9000 (~295M tokens),
after which every parameter trains at the base rate.

| tokens | baseline ×1.0 | static ×0.1 | ramp 0.1→1.0 |
|---|---|---|---|
| 98M | 0.84 | 0.21 | 0.42 |
| 295M (ramp ends) | 0.58 | 0.14 | 0.15 |
| 491M | 0.54 | 0.16 | 0.16 |
| 983M | 0.59 | — | **0.21** |
| final PPL | 2.90 | — | 2.89 |

**The reallocation does not undo.** By the end of the ramp the ramped run has converged onto the
static-×0.1 trajectory, and once the handicap is gone it stays there — 0.21 at 983M vs the
baseline's 0.59, at equal perplexity. The shared branch does slowly re-assert (0.14 → 0.21 between
295M and 983M), far more slowly than the baseline's rise. This is what a race-then-commit dynamic
predicts, and it makes the knob a *training-schedule* intervention rather than a permanent tax on
the shared expert.

It also removes Finding 8's / the paper's caveat that the 16-layer model cannot be brought to
routed balance within 1B tokens: left alone it cannot, but it can be driven well past balance and
kept there.

**Seed replication.** Pooling a second seed into the ramp_end=9000 point above changes nothing:
minimum share 0.14, 0.21 at 983M, identical to the single seed, with ±0.01–0.10 PPL spread on the
ablation deltas. The single-seed caveat originally attached to this finding is retired.

### Dose-response over the handicap duration

Varying only `--shared-ramp-end` (16 layers, mult ×0.1, 983M tokens, seed 0 except where noted):

| handicap held for | 0 (baseline) | 98M | 197M | 295M | 491M | ∞ |
|---|---|---|---|---|---|---|
| min share | 0.54 | 0.24 | 0.16 | 0.14 | 0.14 | 0.08 |
| **share @983M** | **0.59** | **0.33** | **0.26** | **0.21** | **0.19** | **0.14** |
| final PPL | 2.90 | 2.88 | 2.88 | 2.88 | 2.89 | 2.91 |
| seeds | 1 | 1 | 1 | 2 | 1 | 1 |

Three readings:

1. **The effect is graded in the duration**, monotonically — 0.33 → 0.26 → 0.21 → 0.19 → 0.14. The
   hysteresis is a dose-response, not an all-or-nothing switch. Combined with Finding 9's
   dose-response over handicap *strength*, the race has two independent, monotone control axes.
2. **A brief handicap already does much of the work.** Lifting it at 98M tokens — one tenth of the
   budget — lands at 0.33 against the baseline's 0.59, more than half the distance from the
   baseline to the never-released limit of 0.14.
3. **The response flattens but does not plateau.** Increments shrink by 0.07, 0.05 and 0.02 across
   successive endpoints, and the 295M → 491M gap is at the seed-noise floor. But the never-released
   limit lies a further 0.05 below the longest ramp, so no plateau is reached anywhere in the swept
   range. *(Corrected 2026-08-09: this was previously stated as saturation past ~295M tokens, which
   the ∞ point refutes.)*

**Releasing the handicap beats keeping it.** Every ramped condition ends at 2.88, against 2.90 for
the untouched baseline and 2.91 for the never-released run — the only condition worse than doing
nothing. The knob is a training schedule, not a standing tax on the shared expert.

**Remaining caveats.** The 98M / 197M / 491M / ∞ points are single-seed (only 295M is replicated);
all are at one depth and one handicap strength. Full per-checkpoint tables in
`docs/branch-split-runs.md`.

## Finding 13 — Reallocation buys routed specialization, but only via the width lever

The study is motivated by routed experts being the object of interpretability work, and Findings 9
and 11 give two ways to make them carry more computation at no perplexity cost. Do the resulting
routed experts actually *specialize* more? The `routed NMI` column has been recorded throughout;
this reads it properly, at matched training steps and matched architecture.

**The width lever: yes, and cleanly.** Holding the routed branch byte-for-byte identical (E=8,
I=1024) and varying only the shared expert's width, at 983M tokens:

| `I_shared` | 4096 | 1024 | 512 |
|---|---|---|---|
| shared share @983M | 0.82 | 0.50 | 0.34 |
| routed domain NMI | 0.0103 | 0.0174 | **0.0226** |

A factor **2.2** in domain specialization with the routed branch untouched. Widening the E=8 set to
let `I` vary too keeps the ordering nearly intact — NMI 0.0103 / 0.0130 / 0.0174 / 0.0239 / 0.0226
at shares 0.82 / 0.80 / 0.50 / 0.40 / 0.34 — though those runs differ in routed capacity, so the
byte-identical contrast above is the one to trust.

**The learning-rate lever: only while the model is still shared-dominated.** At matched step
(491M tokens), the two depths disagree:

| | ×1.0 | ×0.3 | ×0.1 |
|---|---|---|---|
| 16 layers — share | 0.67 | 0.56 | 0.16 |
| 16 layers — NMI | 0.0108 | 0.0135 | **0.0151** |
| 8 layers — share | 0.48 | 0.52 | 0.10 |
| 8 layers — NMI | 0.0191 | 0.0195 | **0.0175** |

At 16 layers, where the baseline is still shared-dominated at 491M, slowing the shared expert
raises routed NMI monotonically. At 8 layers, where the baseline has *already* crossed to balance,
pushing further into routed dominance **lowers** it. The dose-response series at 983M is likewise
muddled: NMI 0.0123 / 0.0111 / 0.0118 / 0.0142 / 0.0145 for handicaps of 0 / 98M / 197M / 295M /
491M — monotone among the ramps but with the baseline sitting inside the range, not below it.

**Reading.** Engagement and specialization are related but not the same axis. Routed domain
specialization tracks how much the routed branch has been genuinely trained, which rises as it
takes over computation — but it saturates around balance and can decline once the shared branch is
starved past it. Practically this sharpens the paper's own recommendation: **if the goal is routed
experts that are better objects of study, the branch width ratio is the lever, not the learning
rate.** The learning-rate knob buys engagement early and reliably; it does not reliably buy
interpretability.

**Caveats.** Single seed on the 16-layer LR runs; NMI grows mechanically with E (Finding 6) so
cross-E comparisons are excluded here.

### Multi-axis check (2026-08-10): the width-lever gain is not domain-only

Surface-form / frequency / position NMI on the width trio at 983M (2 seeds each,
`analysis.multi_axis`, in `meanablate.log`):

| axis | sh4096 (s=0.82) | base1b (0.50) | sh512 (0.34) | gain |
|---|---|---|---|---|
| surface form | 0.1184 | 0.1407 | 0.1413 | +19%, saturating |
| frequency | 0.0391 | 0.0510 | 0.0563 | +44%, monotone |
| domain | 0.0103 | 0.0200 | 0.0226 | +119%, monotone |
| position | 0.0007 | 0.0003 | 0.0003 | nil (control) |

Every meaningful axis moves with routed engagement; none moves against. The gain is largest on
domain, moderate on frequency, modest and saturating on surface form — which remains the dominant
axis in absolute terms (0.14 ≫ 0.02), reproducing Finding 3's hierarchy. The "legibility" claim
therefore does not rest on the domain axis alone, though its headline factor (2.2×) is
domain-specific and must be quoted as such.

## Finding 14 — Routing granularity `k` is the strongest lever, and the only one that isn't free

Finding 11 established the width ratio at **top-2 throughout**, so it is strictly an ordering at
fixed routing. Test: hold `I = I_shared = 1024` (ratio 1.0) and E=8, vary only `--top-k`.
8 layers, 983M tokens, 2 seeds each; `k=2` is the published `base1b`.

| `k` | 1 | 2 | 4 |
|---|---|---|---|
| min shared share | **0.97** | 0.47 | **0.17** |
| share @983M | 0.97 | 0.50 | 0.19 |
| final PPL | 3.13 | 3.00 | 2.99 |
| routed domain NMI | 0.0003 | ~0.017 | 0.0235 |
| Δrm routed @983M | +0.29 | +2.48 | +3.95 |

1. **`k` enters, and dominates.** At identical widths, top-1 → top-4 swings the split 0.97 → 0.17
   — further than the 0.53 obtained by quadrupling `I_shared`. Finding 11 must be stated as
   holding *at fixed routing*.
2. **This lever is not free**, unlike every other one in this study. Top-1 costs 0.13 PPL; top-4
   doubles routed FLOPs per token. The width manipulations moved the split across 0.25–0.81 inside
   a 0.07 PPL band.
3. **The "active width fraction" repair fails.** `x = I_shared/(I_shared + k·I)` folds `k` in and
   does organize the `k≥2` data (x = 0.20 / 0.33 / 0.50 / 0.67 → splits 0.17–0.37 / 0.47–0.56 /
   0.63 / 0.72–0.81). But `e16cap` (E=16, k=2, I=512) and this `k=1` run have **identical x =
   0.50** and split 0.63 vs 0.97. No scalar of this form covers both.
4. **Top-1 is a degenerate regime, not an extreme point.** The router cannot blend two experts;
   ablating the whole routed branch costs +0.29 PPL, and routed domain NMI stays near zero while
   *declining* over training (0.0053 → 0.0003). Consistent with Finding 6's early top-1 note and
   with the known difficulty of single-expert routing.

**Net (superseded the same day — see below).** For `k≥2`, the split is ordered by how much
*active* width each branch applies. Top-1 should be treated separately.

### The matched-active-width runs refute the active-width scalar too (`gran.log`, 2026-08-09)

Both runs hold `k·I = 2048` and `I_shared = 1024` — the same active widths and the same
`x = 0.333` as `base1b` (0.47) — and change only how the routed width is delivered. 1 seed each.

| | `k=4, I=512` | `base1b` (k=2, I=1024) | `k=1, I=2048` |
|---|---|---|---|
| min share | **0.29** | 0.47 | **0.95** |
| share @983M | 0.32 | 0.50 | 0.95 |
| final PPL | 3.06 | 3.00 | 3.11 |
| routed NMI @983M | **0.0301** | ~0.017 | 0.0003 |
| params | ~96M | 133.7M | ~230M |

1. **Granularity acts beyond active width.** At identical `x`, four experts of 512 land at 0.29 —
   outside the ratio-1.0 band, 0.18 below `base1b`, 9× the seed noise. The active-width scalar is
   refuted even for `k≥2`; it only *appeared* to organize the k≥2 data because k was 2 everywhere.
2. **Top-1's collapse is not a width effect.** One expert of 2048 — double `base1b`'s per-token
   routed width — still gives 0.95. It is the inability to mix experts that kills the routed
   branch, not missing capacity.
3. **`k4i512` sets the study-wide NMI record** (0.0301): the same active width delivered as more,
   narrower experts is also the most domain-specialized configuration measured anywhere.

**Final statement (what the paper says).** No single scalar tested — width ratio or active-width
fraction — subsumes width and routing at once. Jointly: at fixed granularity the width proportion
orders every configuration; at fixed active width, finer granularity lowers the shared share
further; top-1 never engages the routed branch at any width.

## Finding 15 — The branch split is not a norm artifact: mean-ablation replicates, and norm dissociates from impact

An external comment identified the study's sharpest methodological exposure: `s` is a ratio of
**zero-ablation** ΔPPLs, and zeroing a branch in a residual stream perturbs the network by an
amount that scales with the removed branch's output norm — which both headline levers (width, LR)
modulate directly. Test: replace each ablated branch by its per-layer **mean output** over the
eval stream instead of zeros (`analysis.meanablate`), and separately report the **norm share**
`‖f_sh‖/(‖f_sh‖+‖routed‖)` — the quantity the objection says `s` secretly measures.

Result over all nine key configurations (2–4 seeds each, final checkpoint):

| config | `s_zero` | `s_mean` | norm share |
|---|---|---|---|
| LR ×0.1 (491M) | 0.10 | 0.11 | 0.323 |
| k4 | 0.19 | 0.21 | 0.439 |
| sh512 | 0.34 | 0.35 | 0.445 |
| e8i2048 | 0.40 | 0.43 | 0.483 |
| base1b | 0.50 | 0.54 | 0.513 |
| LR ×0.3 (491M) | 0.52 | 0.55 | 0.453 |
| e8i256 | 0.80 | 0.81 | 0.610 |
| sh4096 | 0.82 | 0.86 | 0.670 |
| k1 | 0.97 | 0.97 | 0.759 |

1. **Mean-ablation replicates zero-ablation within 0.04 everywhere**, across a span of 0.10 → 0.97.
   The ordering is identical; nothing about the paper's conclusions is measurement-dependent.
2. **Norm share and ablation share dissociate, in both directions.** At k=1 the routed branch
   carries 24% of the output norm but ~3% of the ablation impact (+0.29 vs +8.89): norm without
   function. At LR ×0.1 the shared branch carries 32% of the norm but 10% of the impact. `s` is
   therefore not a norm readout — a branch can have substantial norm and no useful computation.
3. The norm share is monotone in `s` but far *less* extreme (0.32–0.76 vs 0.10–0.97): ablation
   amplifies real functional differences, it does not manufacture them.

Raw table in `meanablate.log`; the zero-vs-mean full trajectories of the width trio and the
surface-form NMI runs are in the same log.

### The scalar control (2026-08-11): a pure norm intervention does not move the split

Train with the shared expert's output multiplied by a fixed, non-learned α — widths and learning
rates untouched (`--shared-scale`, `alpha.log`). Under Adam a constant gradient scale largely
cancels in the adaptive normalization, so α is close to a *pure contribution-norm* intervention.
1 seed each, 8 layers, 983M tokens:

| | α=0.5 | α=1.0 (`base1b`) | α=2.0 | width lever, for comparison |
|---|---|---|---|---|
| min share | 0.43 | 0.47 | 0.42 | 0.28 (`sh512`) / 0.81 (`sh4096`) |
| share @983M | 0.46 | 0.50 | 0.43 | 0.34 / 0.82 |
| final PPL | 2.99 | 3.00 | 3.02 | 3.01 / 2.97 |
| routed NMI | 0.0172 | 0.0174 | 0.0122 | 0.0226 / 0.0103 |

Halving or doubling the shared branch's contribution norm leaves the split within ±0.07 of the
baseline; the width manipulations move it by ±0.35. **"Width" is not "norm"** — the width lever's
effect is representational capacity, not output scale. This was the external comment's own
proposed decisive test, and it comes out against the norm-confound reading. (Curiosity: both α
directions *lower* the share slightly, and α=2.0 lowers routed NMI; not interpreted further.)

## Limitations

Pilot scale (134M params, ≤197M tokens, 8 experts, 512 context, single hardware). Even
unconstrained (`aux=0`), domain-NMI is modest (0.034 = 3.4% of domain entropy): domain
specialization is present and significant but not dominant — consistent with prior reports that
experts specialize more by token/syntax than topic. Conditional-NMI estimates are finite-sample
biased and not relied upon. The held-out eval uses a skip-offset of the training streams rather
than a fully disjoint split. **Batch-size confound:** held-out NMI and branch importance are
sensitive to batch size at fixed token count (CE, 98M tokens: batch-24 NMI 0.0063 / remove-routed
+1.03 vs batch-64 NMI 0.0141 / +0.84); sweeps in this document used mixed batch sizes (32/64), so
cross-experiment magnitude comparisons should be read with care — the within-experiment,
matched-batch comparisons and the tokens-vs-importance trajectory (Finding 8) are the reliable
ones. Distillation results are single-seed and not statistically established. **Finding 11**
characterizes an *ordering* over the width ratio at a single depth and token budget, not a
functional form — whether the split depends on the ratio alone or on ratio and depth jointly is
untested. **Finding 12** is single-seed at one ramp endpoint and one depth. **Confounding:** our
own retraction of Finding 10 came from letting total capacity ride along with the swept variable;
architecture sweeps here now match capacity or parameters explicitly.

## Framing

A controlled, causal, seed-replicated account of expert organization in a small shared-expert
MoE:

1. **The shared expert is the computational workhorse** — removing all 8 routed experts costs
   far less than removing the lone shared expert (+0.5 vs +4–6 PPL), and its dominance
   *strengthens* with scale.
2. **The load-balancing loss is the master control on specialization** — sweeping it moves
   domain-NMI ~4× monotonically, at no perplexity cost.
3. **Specialization is by surface form first, domain second** — experts sort tokens more by
   lexical class than by topic; position is irrelevant.
4. **Shared and routed experts are complements, not substitutes** — the shared expert provides
   a common-capability backstop that *enables* routed experts to specialize; removing it
   *reduces* specialization once balancing is off.
5. **The split is a design choice, not a property of the architecture** — its *timing* is set by
   the gradient race (and a temporary LR handicap moves it permanently), its *level* by the
   routed:shared width ratio; across 0.25→0.81 of shared share, perplexity moves by <0.1.

The methodological contribution is the pairing of a domain-labelled probe with causal branch
ablation, a load-balancing sweep, and strict seed replication — which, notably, overturned three
of our own conclusions (the "no-shared sharpens" separation, the "substitutes" framing, and the
non-monotonic #experts law). The first two were small-sample or under-controlled artifacts that
replication and the `aux=0` interaction corrected; the third was a capacity confound that only a
width-matched sweep exposed.
