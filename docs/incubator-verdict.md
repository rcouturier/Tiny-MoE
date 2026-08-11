# Why the "incubator" idea does not work

*Verdict note, 2026-08-11. Follow-up idea assessed against the Tiny-MoE study and killed by one
targeted experiment. Total cost: 8 GPU-hours.*

**The idea.** A DeepSeekMoE-style shared branch is a genuine dense sub-network: every token passes
through it and it needs no router, so it can be detached as-is — unlike the routed branch, which
cannot run without its router and its full expert pool. The parent paper shows it is also
gradient-favoured and matures first. Hence: train the big MoE as an *incubator*, then detach
`embeddings + attention + norms + shared expert` and ship that as a small standalone "child",
treating the routed experts as scaffolding discarded at deployment.

The intuition is sound. The arithmetic is not.

---

## 1. We had already measured the child, without noticing

The pivot of the whole assessment: **ablating the entire routed branch *is* the child's forward
pass.** It leaves exactly `embeddings + attention + norms + shared expert`. So

```
PPL(detached child) = PPL(parent) + Δrm_routed
```

and `Δrm_routed` is a column the study has printed since its first experiment. No new training was
needed to evaluate the proposal.

For the reference model (`base1b`, 8 layers, E=8, k=2, I=I_shared=1024):

| quantity | value |
|---|---|
| parent PPL @983M | 3.00 |
| Δrm routed | +2.48 |
| **child PPL** | **5.48** |
| child size | 33M (20.4M trunk + 12.6M shared expert) |
| measured dense trained solo, 58M | **3.11** |

A 33M dense trained directly would be somewhat worse than the 58M one — perhaps 3.3–3.4, *this is
an interpolation, not measured* — but nowhere near 5.48. The detached child is not "slightly
degraded"; it is far worse than a model one could simply have trained.

## 2. The structural reason: it is nearly a tautology

> `Δrm_routed` is large **exactly when** the routed branch is doing useful work.

So either the routed experts contribute — and removing them is expensive, so the child is bad — or
they do not — and the child is fine, but 100M parameters of scaffolding were trained for nothing
and the parent is worse than a plain dense model.

**The quantity measuring extraction quality is the same one measuring the scaffolding's
uselessness.** There is no free corner.

The data show both horns. The best extraction in the whole study comes from the top-1 model, whose
child reaches **3.42** — nearly respectable. But that parent tops out at **3.13**, i.e. *worse than
the 58M dense at 3.11*. One would have trained 134M parameters to obtain a 33M model neither better
nor cheaper than one trained directly.

| parent | parent PPL | child PPL | child size | shared share `s` |
|---|---|---|---|---|
| top-1 (`k1r1`) | 3.13 | **3.42** | 33M | 0.97 |
| `sh4096` | 2.97 | 4.15 | 71M | 0.82 |
| `e8i256` | 3.12 | 4.51 | 33M | 0.80 |
| reference (`base1b`) | 3.00 | 5.48 | 33M | 0.50 |
| top-4 (`k4r1`) | 2.99 | 6.94 | 33M | 0.19 |

Child quality tracks `s` perfectly — by construction, since the two are complements of the same
ratio.

## 3. Why the two branches never duplicate each other

The deeper reason is architectural: the residual stream **adds** the two branches. Neither has any
incentive to learn what the other already provides; joint training pushes them to **divide** the
labour, each doing what the other does not. They are additive complements, not redundant copies.

**Extraction fails for the very reason the architecture works.**

For the incubator to succeed, the shared branch would have to *absorb* what the routed experts
know. That is distillation, and it does not happen spontaneously through co-training.

## 4. The last escape hatch, and its closing

One possibility remained: find a regime with a **healthy parent and a dominant shared branch**.
Exactly one lever had never been tested — *accelerating* the shared expert instead of slowing it
(the published sweep only handicaps: m = 0.1, 0.3, 1.0). We ran m = 3 and m = 10.

| multiplier `m` | 0.1 | 0.3 | **1.0** | 3.0 | 10.0 |
|---|---|---|---|---|---|
| shared share `s` @983M | 0.10 | ~0.52 | **0.50** | 0.17 | 0.20 |
| parent PPL | — | — | 3.00 | 3.02 | 3.09 |
| **child PPL** | — | — | 5.48 | **7.01** | **7.95** |
| routed NMI | — | — | 0.0174 | 0.0096 | 0.0058 |

It does not help — it goes the other way. **Accelerating the shared expert makes it *less*
important, not more.** `s` peaks at the neutral multiplier and collapses on both sides, and the
child gets worse still. No divergence: m=10 trains stably, it is simply useless.

(Independently interesting, and kept as follow-up material in `results.md` Finding 16: this
refutes the naive reading of the gradient-exposure mechanism — "more gradient on the shared branch
⇒ more shared dominance" is false. A plausible but untested account: an over-accelerated shared
expert converges early onto a mediocre solution and stops learning, while the still-plastic routed
experts absorb the rest. Same race, but winning too fast loses.)

## 5. What would have to be true

The falsifiable form: the incubator needs **a configuration where removing the routed branch is
cheap even though it was genuinely contributing.** Eighteen configurations say no such point
exists, and §3 explains why this is a property of the architecture rather than a sampling gap.

## 6. Compute accounting, for completeness

Two framings were on the table:

- *"An efficient way to get a small model"* — no. Training the child directly is cheaper and gives
  a better model.
- *"A free deployable by-product"* — the strong framing, and it requires the child to be usable
  as-is. At 5.48 against ~3.3 it is not. Add fine-tuning and it is no longer free, and the claim
  collapses to "a good initialisation", a crowded and much weaker result.

## Bottom line

Worth the pilot, not the paper — and the decisive pilot was not the proposed
incubated-vs-solo-vs-distilled comparison (three training runs, measuring a consequence) but the
single untested lever behind it (two runs, testing the cause). Existing data already argued
against; boosting closed the last door.

**No impact on the submitted paper.** It demonstrates the *handicap* direction causally and never
claims symmetry — the phrasing "settles it in either direction" was drafted and rejected during
review, which these data vindicate.

---

*Raw tables: `boost.log` (m>1 runs), `branch-split-runs.md` (all configurations),
`results.md` Finding 16.*
