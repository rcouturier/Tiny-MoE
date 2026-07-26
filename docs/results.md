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

## Limitations

Pilot scale (134M params, ≤197M tokens, 8 experts, 512 context, single hardware). Even
unconstrained (`aux=0`), domain-NMI is modest (0.034 = 3.4% of domain entropy): domain
specialization is present and significant but not dominant — consistent with prior reports that
experts specialize more by token/syntax than topic. Conditional-NMI estimates are finite-sample
biased and not relied upon. The held-out eval uses a skip-offset of the training streams rather
than a fully disjoint split.

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

The methodological contribution is the pairing of a domain-labelled probe with causal branch
ablation, a load-balancing sweep, and strict seed replication — which, notably, overturned two
of our own early conclusions (the "no-shared sharpens" separation and the "substitutes"
framing). Both were small-sample or under-controlled artifacts that replication and the
`aux=0` interaction corrected.
