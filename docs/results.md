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

| handicap held for | 0 (baseline) | 98M | 197M | 295M | 491M |
|---|---|---|---|---|---|
| min share | 0.54 | 0.24 | 0.16 | 0.14 | 0.14 |
| **share @983M** | **0.59** | **0.33** | **0.26** | **0.21** | **0.19** |
| final PPL | 2.90 | 2.88 | 2.88 | 2.88 | 2.89 |
| seeds | 1 | 1 | 1 | 2 | 1 |

Three readings:

1. **The effect is graded in the duration**, monotonically — 0.33 → 0.26 → 0.21 → 0.19. The
   hysteresis is a dose-response, not an all-or-nothing switch. Combined with Finding 9's
   dose-response over handicap *strength*, the race has two independent, monotone control axes.
2. **A brief handicap already does most of the work.** Lifting it at 98M tokens — one tenth of
   the budget — still lands at 0.33 against the baseline's 0.59: roughly two thirds of the total
   achievable reallocation, bought in the first tenth of training.
3. **The curve saturates.** The 295M → 491M gap is 0.02, equal to the seed noise measured on this
   quantity elsewhere in the study; **those two points are not distinguishable**. Past ~295M
   tokens, extending the handicap buys nothing measurable.

Perplexity is flat across the entire series (2.88–2.89) and slightly better than the baseline's
2.90 throughout — consistent with Finding 9.

**Remaining caveats.** The 98M / 197M / 491M points are single-seed (only 295M is replicated); all
are at one depth and one handicap strength. Full per-checkpoint tables in
`docs/branch-split-runs.md`.

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
