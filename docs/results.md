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
separation). This **did not replicate**: at 4 seeds the difference collapses to
0.0093 ± 0.0013 vs 0.0085 ± 0.0015, permutation p = 0.543 (overlapping). We report this as a
null result — the 3-seed separation was small-sample noise, caught by replication. No claim
that the shared expert suppresses routed specialization *by displacement*; the suppression
lever is the load-balancing loss (Finding 2), not the shared expert's presence.

## Interaction (shared × aux) — pending

Does the shared expert still dominate at `aux=0` (where routed experts specialize more)? And
does removing it change specialization once balancing is off? 2×2 {shared, no-shared} ×
{aux=0, aux=1e-2}, 4 seeds. *(filled by the running experiment.)*

## What axis do experts specialize by? — pending

soft NMI(Expert; axis) for axis ∈ {domain, surface-class, token-frequency, sequence-position}
on the `aux=0` models, to test whether the (modest) specialization is by topic or, as in
Mixtral, more by token/surface form. *(filled by the running experiment.)*

## Scale — pending

Domain-NMI and shared-branch dominance at `aux=0` for training budgets of 3k / 6k / 12k steps
(≈ 49M / 98M / 197M tokens), to see whether the effects strengthen or fade with more tokens.
*(filled by the running experiment.)*

## Limitations

Pilot scale (134M params, ≤197M tokens, 8 experts, 512 context, single hardware). Even
unconstrained (`aux=0`), domain-NMI is modest (0.034 = 3.4% of domain entropy): domain
specialization is present and significant but not dominant — consistent with prior reports that
experts specialize more by token/syntax than topic. Conditional-NMI estimates are finite-sample
biased and not relied upon. The held-out eval uses a skip-offset of the training streams rather
than a fully disjoint split.

## Framing

Two robust, causal, replicated findings at small scale: (1) in a shared-expert MoE the shared
expert does most of the FFN work and the routed experts are near-redundant; (2) the degree of
routed domain-specialization is directly controlled by the load-balancing coefficient, at no
perplexity cost. Together they suggest the routed experts are "under-used" both because the
shared expert absorbs the common computation and because balancing represses their
specialization — a controlled, quantified account of a tension usually discussed only at scale.
