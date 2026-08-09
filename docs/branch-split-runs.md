# Raw run record: what sets the shared/routed branch split

Experiments run **2026-08-02 → 2026-08-05** on a single RTX A6000 (`cluster5`), after the
version of `paper.md` currently in this repository was written. **These results supersede
parts of §5 and all of §6 of that paper — see "What this changes" below. Nothing here has
been integrated into `paper.md` / `results.md` yet.**

Raw logs live on the training host at `~/tiny_moe_study/{followup,capacity,collapse,collapse_s1,ratio}.log`,
alongside the driver scripts `run_{followup,capacity,collapse,collapse_s1,ratio}.sh`.
Checkpoints are under `~/tiny_moe_study/<run>/step_*`.

## Common setup

All runs below: 8 transformer layers unless noted, top-k 2, batch 64, sequence length 512,
lr 3e-4, warmup 1500, aux-coef 1e-2, 30000 steps ≈ 983M tokens, checkpoint every 3000 steps.
Analysis is `analysis.race` with `--tokens-per-domain 60000 --skip 3000`.
`I` denotes the per-expert (routed) intermediate size, `I_shared` the shared expert's width.
The shared branch is a single expert active on every token; each routed expert sees ~k/E of tokens.
"Active routed capacity" means k·I, the routed width a single token actually flows through.

The reported quantity is `shared/(sh+rt)` — the shared branch's share of total ablation impact.
`±` is the spread across seeds; `±0.00` means a single seed, **not** zero variance.

## Summary

Minimum share reached over the trajectory, and share at 983M tokens.

| run | layers | E | I | I_shared | ratio I/I_shared | active k·I | params | seeds | min share | @983M | final PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `e8i2048` | 8 | 8 | 2048 | 1024 | 2.0 | 4096 | 234.4M | 2 | 0.37 | 0.40 | 2.95 |
| `e4wide` | 8 | 4 | 2048 | 1024 | 2.0 | 4096 | 133.7M | 2 | **0.25** | 0.28 | 2.99 |
| `sh512` | 8 | 8 | 1024 | 512 | 2.0 | 2048 | 127.4M | 2 | 0.28 | 0.34 | 3.01 |
| `base1b` (ref) | 8 | 8 | 1024 | 1024 | 1.0 | 2048 | 133.7M | 4 | 0.47 | 0.50 | 3.00 |
| `e16cap` | 8 | 16 | 512 | 1024 | 0.5 | 1024 | 133.7M | 2 | 0.63 | 0.67 | 3.00 |
| `e32cap` | 8 | 32 | 256 | 1024 | 0.25 | 512 | 133.8M | 2 | 0.72 | 0.75 | 3.02 |
| `e8i256` | 8 | 8 | 256 | 1024 | 0.25 | 512 | 58.2M | 2 | 0.79 | 0.80 | 3.12 |
| `sh4096` | 8 | 8 | 1024 | 4096 | 0.25 | 2048 | 171.5M | 2 | **0.81** | 0.82 | 2.97 |
| `scale16_1b` (ref) | 16 | 8 | 1024 | 1024 | 1.0 | 2048 | 251.0M | 1 | 0.54 | 0.59 | 2.90 |
| `dyn16` | 16 | 8 | 1024 | 1024 | 1.0 | 2048 | 251.0M | 2 | 0.14 | 0.21 | 2.88 |

`base1b` and `scale16_1b` are the already-published baselines (`finding8.log`, `extend.log`),
repeated here as reference rows. `dyn16` is the dynamic shared-LR run (see below), not an
architecture variant.

Perplexities are **not** comparable across rows with different parameter counts
(58.2M → 234.4M). Only the share is.

### Grouped by width ratio

| ratio I/I_shared | 2.0 | 1.0 | 0.5 | 0.25 |
|---|---|---|---|---|
| min share observed | 0.25 / 0.28 / 0.37 | 0.47 / 0.53 / 0.55 / 0.56 | 0.63 | 0.72 / 0.79 / 0.81 |

The ratio-1.0 entries include the four ratio-1.0 points of the published E-sweep
(E ∈ {4,8,16,32} at I = I_shared = 1024; see `results.md` Finding 10). The bands do not
overlap: the ratio orders every configuration measured, across 8× in ratio, 4× in parameter
count and 8× in expert count. Within a ratio level the residual spread is ~0.1 and is **not**
consistently explained by E, by absolute capacity, or by total routed parameters — at ratio 2.0
more experts raises the share (E=4 → 0.25 vs E=8 → 0.37) while at ratio 0.25 more experts
lowers it (E=8 → 0.79 vs E=32 → 0.72).

### The two disentangling contrasts

Both hold E=8 and k=2 fixed, so expert count and routing are constant throughout.

**Varying routed width** (`e8i256` / `base1b` / `e8i2048`, I_shared pinned at 1024):

| I | 256 | 1024 | 2048 |
|---|---|---|---|
| min share | 0.79 | 0.47 | 0.37 |

**Varying shared width** (`sh4096` / `base1b` / `sh512`, I pinned at 1024 — routed active
capacity is *identical* in all three, 2048):

| I_shared | 4096 | 1024 | 512 |
|---|---|---|---|
| min share | 0.81 | 0.47 | 0.28 |

The second contrast is the decisive one: the routed branch is byte-for-byte the same in all
three runs, and the split still moves by 0.53. Absolute routed capacity therefore does not
govern the split. The two contrasts agree wherever they meet the same ratio (0.79 vs 0.81 at
ratio 0.25; 0.37 vs 0.28 at ratio 2.0) despite parameter counts differing by up to 4×.

## What this changes

1. **Finding 10 as published is wrong on two counts.** Its non-monotonic #experts curve
   (0.56 / 0.48 / 0.53 / 0.55 for E = 4/8/16/32) held per-expert width fixed, so total routed
   capacity grew with E and confounded the comparison; and its reported perplexity gain with E
   (3.28 → 2.90) was a parameter-count effect — at matched parameters the perplexity is flat
   (2.99 / 3.00 / 3.00 / 3.02 for E = 4/8/16/32 in `e4wide`/`base1b`/`e16cap`/`e32cap`).
   Expert count is not what governs the split; the branch width ratio is.
2. **The ∝E/k prediction of Finding 9 is not supported** as the determinant of where the split
   lands. The gradient-exposure race remains causally demonstrated by the shared-LR control;
   what these runs refute is the extrapolation from that mechanism to a law over E.
3. **§5's caveat that the 16-layer model never crosses to balance within 1B tokens is removed**
   by the `dyn16` result below.

## Dynamic shared-LR (`dyn16`) — hysteresis

16 layers, 251.0M params, shared-expert LR multiplier ramped from ×0.1 up to ×1.0 by step 9000
(~295M tokens), **2 seeds** (seed 1 added 2026-08-08, see the dose-response section below).
Compare to the published 16-layer runs: baseline ×1.0 and static ×0.1.

| tokens | baseline ×1.0 | static ×0.1 | ramp 0.1→1.0 |
|---|---|---|---|
| 98M | 0.84 | 0.21 | 0.42 |
| 295M (ramp ends) | 0.58 | 0.14 | 0.15 |
| 491M | 0.54 | 0.16 | 0.16 |
| 983M | 0.59 | — | 0.21 |
| final PPL | 2.90 | — | 2.89 |

Once the handicap is lifted the reallocation does not undo: the share stays at 0.15–0.21
through 983M tokens instead of returning toward the baseline's 0.59, at unchanged perplexity.
The shared branch does slowly re-assert (0.14 → 0.21 between 295M and 983M), far more slowly
than the baseline's rise.

The two caveats stated here originally — single seed, single ramp endpoint — are both addressed
by the run series below. The single-depth caveat stands.

## Dose-response over the handicap duration

Experiments run **2026-08-07 → 2026-08-08**. All: 16 layers, 251.0M params, `--shared-lr-mult 0.1`,
30000 steps ≈ 983M tokens, seed 0 unless noted; only `--shared-ramp-end` varies. The ramp is
linear in step, `mult = 0.1 + 0.9·min(1, step/ramp_end)`, so `ramp_end` is the step by which the
shared expert is back at the base learning rate. Driver `run_dose.sh`, log `dose.log`.

| ramp end (step) | handicap lifted at | seeds | min share | **@983M** | final PPL |
|---|---|---|---|---|---|
| — (baseline ×1.0) | never applied | 1 | 0.54 | **0.59** | 2.90 |
| 3000 | 98M tokens | 1 | 0.24 | **0.33** | 2.88 |
| 6000 | 197M | 1 | 0.16 | **0.26** | 2.88 |
| 9000 (`dyn16`) | 295M | 2 | 0.14 | **0.21** | 2.88 |
| 15000 | 491M | 1 | 0.14 | **0.19** | 2.89 |
| ∞ (`m01long`, static ×0.1) | never lifted | 1 | 0.08 | **0.14** | 2.91 |

Three readings:

1. **The effect is graded in the duration.** 0.33 → 0.26 → 0.21 → 0.19 → 0.14 as the handicap runs
   longer, monotonically. The hysteresis is a dose-response, not an all-or-nothing switch.
2. **A brief handicap already does much of the work.** Lifting it at 98M tokens — 10% of the token
   budget — lands at 0.33 against the baseline's 0.59, more than half the distance from the
   baseline to the never-released limit of 0.14.
3. **The response flattens but does not plateau.** Increments shrink by 0.07, 0.05 and 0.02 across
   successive endpoints, and the 295M → 491M gap of 0.02 is at the seed-noise floor. But the
   never-released limit sits a further 0.05 below the longest ramp, so there is no plateau
   anywhere in the swept range. *(An earlier version of this record claimed the curve saturated
   past ~295M tokens; the ∞ point shows that was too strong.)*

**Releasing the handicap is marginally better than keeping it.** Every ramped condition ends at
2.88, against 2.90 for the untouched baseline and 2.91 for the never-released run — which is the
only condition here worse than doing nothing. The intervention is a schedule, not a standing tax.

**Caveats.** The 3000 / 6000 / 15000 / ∞ points are single-seed; only the 9000 point is
replicated. All are at one depth (16 layers) and one handicap strength (×0.1); the strength axis is
the separate dose-response already published as Finding 9.

#### `m01long_s0` — static ×0.1, never released, to 983M (`asymptote.log`)

```
     0 |   10.41 |   +0.50±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.13 |   +0.95±0.00 |   +3.37±0.00 |     0.0127 |          0.22
  6000 |    3.58 |   +0.58±0.00 |   +4.78±0.00 |     0.0151 |          0.11
  9000 |    3.35 |   +0.45±0.00 |   +5.11±0.00 |     0.0155 |          0.08
 12000 |    3.22 |   +0.46±0.00 |   +5.30±0.00 |     0.0150 |          0.08
 15000 |    3.13 |   +0.51±0.00 |   +5.21±0.00 |     0.0154 |          0.09
 18000 |    3.05 |   +0.59±0.00 |   +5.12±0.00 |     0.0149 |          0.10
 21000 |    2.99 |   +0.69±0.00 |   +5.07±0.00 |     0.0153 |          0.12
 24000 |    2.94 |   +0.76±0.00 |   +5.00±0.00 |     0.0155 |          0.13
 27000 |    2.92 |   +0.84±0.00 |   +5.01±0.00 |     0.0153 |          0.14
 30000 |    2.91 |   +0.84±0.00 |   +5.00±0.00 |     0.0153 |          0.14
```

### Seed replication of the published point (`dyn16`, ramp_end 9000)

Pooling seed 1 with the published seed 0 changes nothing: minimum share 0.14 and 0.21 at 983M in
both. Seed spread on the ablation deltas is ±0.01–0.10 PPL. The single-seed result stands.

### Per-checkpoint tables (dose-response)

Columns as above: `step | basePPL | Δrm shared | Δrm routed | routed NMI | shared/(sh+rt)`.

#### `dyn16_r3000_s0` — ramp ends at 98M tokens (`dose.log`)

```
     0 |   10.41 |   +0.50±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.12 |   +1.35±0.00 |   +1.75±0.00 |     0.0099 |          0.44
  6000 |    3.55 |   +0.90±0.00 |   +2.75±0.00 |     0.0116 |          0.25
  9000 |    3.32 |   +0.90±0.00 |   +2.89±0.00 |     0.0112 |          0.24
 12000 |    3.19 |   +0.96±0.00 |   +2.99±0.00 |     0.0115 |          0.24
 15000 |    3.11 |   +1.04±0.00 |   +2.95±0.00 |     0.0113 |          0.26
 18000 |    3.02 |   +1.15±0.00 |   +2.94±0.00 |     0.0113 |          0.28
 21000 |    2.96 |   +1.27±0.00 |   +2.97±0.00 |     0.0114 |          0.30
 24000 |    2.91 |   +1.39±0.00 |   +2.96±0.00 |     0.0112 |          0.32
 27000 |    2.89 |   +1.47±0.00 |   +3.02±0.00 |     0.0111 |          0.33
 30000 |    2.88 |   +1.49±0.00 |   +3.01±0.00 |     0.0111 |          0.33
```

#### `dyn16_r6000_s0` — ramp ends at 197M tokens (`dose.log`)

```
     0 |   10.41 |   +0.50±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.12 |   +1.53±0.00 |   +1.95±0.00 |     0.0112 |          0.44
  6000 |    3.56 |   +0.69±0.00 |   +3.21±0.00 |     0.0116 |          0.18
  9000 |    3.32 |   +0.67±0.00 |   +3.42±0.00 |     0.0123 |          0.16
 12000 |    3.19 |   +0.67±0.00 |   +3.40±0.00 |     0.0120 |          0.17
 15000 |    3.11 |   +0.77±0.00 |   +3.34±0.00 |     0.0120 |          0.19
 18000 |    3.02 |   +0.87±0.00 |   +3.33±0.00 |     0.0116 |          0.21
 21000 |    2.96 |   +0.96±0.00 |   +3.36±0.00 |     0.0119 |          0.22
 24000 |    2.91 |   +1.07±0.00 |   +3.30±0.00 |     0.0119 |          0.24
 27000 |    2.89 |   +1.14±0.00 |   +3.36±0.00 |     0.0119 |          0.25
 30000 |    2.88 |   +1.16±0.00 |   +3.35±0.00 |     0.0118 |          0.26
```

#### `dyn16_s0` + `dyn16_s1` — ramp ends at 295M tokens, 2 seeds pooled (`dose.log`)

```
     0 |   10.41 |   +0.48±0.01 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.12 |   +1.59±0.01 |   +2.23±0.05 |     0.0123 |          0.42
  6000 |    3.55 |   +0.99±0.05 |   +3.20±0.10 |     0.0150 |          0.24
  9000 |    3.33 |   +0.62±0.01 |   +3.57±0.07 |     0.0158 |          0.15
 12000 |    3.19 |   +0.59±0.01 |   +3.66±0.09 |     0.0151 |          0.14
 15000 |    3.10 |   +0.66±0.03 |   +3.62±0.05 |     0.0146 |          0.15
 18000 |    3.02 |   +0.73±0.01 |   +3.57±0.04 |     0.0148 |          0.17
 21000 |    2.96 |   +0.82±0.01 |   +3.64±0.01 |     0.0147 |          0.18
 24000 |    2.91 |   +0.90±0.02 |   +3.58±0.01 |     0.0144 |          0.20
 27000 |    2.89 |   +0.97±0.02 |   +3.62±0.01 |     0.0143 |          0.21
 30000 |    2.88 |   +0.98±0.03 |   +3.63±0.04 |     0.0142 |          0.21
```

#### `dyn16_r15000_s0` — ramp ends at 491M tokens (`dose.log`)

```
     0 |   10.41 |   +0.50±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.13 |   +1.41±0.00 |   +2.53±0.00 |     0.0121 |          0.36
  6000 |    3.57 |   +1.04±0.00 |   +3.34±0.00 |     0.0144 |          0.24
  9000 |    3.34 |   +0.86±0.00 |   +3.57±0.00 |     0.0145 |          0.19
 12000 |    3.20 |   +0.71±0.00 |   +3.64±0.00 |     0.0148 |          0.16
 15000 |    3.12 |   +0.63±0.00 |   +3.75±0.00 |     0.0152 |          0.14
 18000 |    3.03 |   +0.67±0.00 |   +3.79±0.00 |     0.0143 |          0.15
 21000 |    2.96 |   +0.73±0.00 |   +3.74±0.00 |     0.0146 |          0.16
 24000 |    2.92 |   +0.81±0.00 |   +3.69±0.00 |     0.0147 |          0.18
 27000 |    2.89 |   +0.88±0.00 |   +3.78±0.00 |     0.0146 |          0.19
 30000 |    2.89 |   +0.90±0.00 |   +3.79±0.00 |     0.0145 |          0.19
```

## Raw tables

Columns: `step | basePPL | Δrm shared | Δrm routed | routed NMI | shared/(sh+rt)`.

### D1 — `e4wide`, E=4 @ I=2048, I_shared=1024, 2 seeds (`followup.log`)

```
     0 |   10.43 |   +0.49±0.01 |   +0.00±0.00 |     0.0004 |          1.00
  3000 |    4.17 |   +2.76±0.18 |   +1.07±0.02 |     0.0112 |          0.72
  6000 |    3.62 |   +1.54±0.03 |   +2.19±0.02 |     0.0172 |          0.41
  9000 |    3.40 |   +1.21±0.03 |   +2.69±0.01 |     0.0187 |          0.31
 12000 |    3.28 |   +1.12±0.04 |   +3.06±0.06 |     0.0194 |          0.27
 15000 |    3.19 |   +1.11±0.05 |   +3.24±0.06 |     0.0199 |          0.26
 18000 |    3.12 |   +1.13±0.04 |   +3.32±0.04 |     0.0209 |          0.25
 21000 |    3.06 |   +1.20±0.04 |   +3.42±0.03 |     0.0211 |          0.26
 24000 |    3.02 |   +1.27±0.03 |   +3.47±0.03 |     0.0210 |          0.27
 27000 |    2.99 |   +1.32±0.04 |   +3.47±0.02 |     0.0207 |          0.28
 30000 |    2.99 |   +1.33±0.04 |   +3.47±0.02 |     0.0208 |          0.28
```

### D2 — `dyn16`, 16 layers, shared-LR ramp 0.1→1.0 by step 9000, 1 seed (`followup.log`)

```
     0 |   10.41 |   +0.50±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.12 |   +1.58±0.00 |   +2.18±0.00 |     0.0127 |          0.42
  6000 |    3.56 |   +0.95±0.00 |   +3.10±0.00 |     0.0143 |          0.23
  9000 |    3.33 |   +0.63±0.00 |   +3.50±0.00 |     0.0154 |          0.15
 12000 |    3.20 |   +0.58±0.00 |   +3.58±0.00 |     0.0147 |          0.14
 15000 |    3.11 |   +0.68±0.00 |   +3.57±0.00 |     0.0145 |          0.16
 18000 |    3.03 |   +0.74±0.00 |   +3.53±0.00 |     0.0144 |          0.17
 21000 |    2.96 |   +0.83±0.00 |   +3.63±0.00 |     0.0144 |          0.19
 24000 |    2.92 |   +0.92±0.00 |   +3.57±0.00 |     0.0144 |          0.20
 27000 |    2.89 |   +0.99±0.00 |   +3.63±0.00 |     0.0146 |          0.21
 30000 |    2.89 |   +1.00±0.00 |   +3.67±0.00 |     0.0145 |          0.21
```

### `e16cap`, E=16 @ I=512, I_shared=1024, 2 seeds (`capacity.log`)

```
     0 |   10.42 |   +0.50±0.01 |   +0.00±0.00 |     0.0017 |          1.00
  3000 |    4.15 |   +3.67±0.01 |   +0.79±0.01 |     0.0163 |          0.82
  6000 |    3.62 |   +3.15±0.10 |   +1.30±0.02 |     0.0173 |          0.71
  9000 |    3.40 |   +2.89±0.08 |   +1.61±0.08 |     0.0188 |          0.64
 12000 |    3.28 |   +2.92±0.04 |   +1.72±0.04 |     0.0184 |          0.63
 15000 |    3.20 |   +3.05±0.04 |   +1.80±0.09 |     0.0184 |          0.63
 18000 |    3.13 |   +3.21±0.03 |   +1.80±0.06 |     0.0191 |          0.64
 21000 |    3.07 |   +3.45±0.05 |   +1.84±0.06 |     0.0186 |          0.65
 24000 |    3.03 |   +3.65±0.07 |   +1.86±0.06 |     0.0187 |          0.66
 27000 |    3.01 |   +3.77±0.06 |   +1.89±0.09 |     0.0187 |          0.67
 30000 |    3.00 |   +3.79±0.07 |   +1.90±0.09 |     0.0188 |          0.67
```

### `e32cap`, E=32 @ I=256, I_shared=1024, 2 seeds (`capacity.log`)

```
     0 |   10.42 |   +0.48±0.01 |   -0.00±0.00 |     0.0032 |          1.00
  3000 |    4.15 |   +3.63±0.26 |   +0.74±0.02 |     0.0226 |          0.83
  6000 |    3.62 |   +3.48±0.22 |   +1.16±0.06 |     0.0237 |          0.75
  9000 |    3.41 |   +3.48±0.07 |   +1.33±0.03 |     0.0229 |          0.72
 12000 |    3.30 |   +3.63±0.15 |   +1.40±0.03 |     0.0237 |          0.72
 15000 |    3.22 |   +3.84±0.17 |   +1.48±0.01 |     0.0227 |          0.72
 18000 |    3.15 |   +4.08±0.25 |   +1.50±0.03 |     0.0226 |          0.73
 21000 |    3.09 |   +4.40±0.24 |   +1.52±0.01 |     0.0228 |          0.74
 24000 |    3.05 |   +4.64±0.27 |   +1.54±0.01 |     0.0226 |          0.75
 27000 |    3.03 |   +4.78±0.29 |   +1.57±0.00 |     0.0225 |          0.75
 30000 |    3.02 |   +4.81±0.29 |   +1.58±0.00 |     0.0225 |          0.75
```

### `e8i256`, E=8 @ I=256, I_shared=1024, 2 seeds (`collapse_s1.log`)

```
     0 |   10.42 |   +0.50±0.01 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.23 |   +4.70±0.17 |   +0.44±0.00 |     0.0103 |          0.91
  6000 |    3.72 |   +4.38±0.04 |   +0.79±0.01 |     0.0134 |          0.85
  9000 |    3.51 |   +4.30±0.20 |   +0.98±0.01 |     0.0140 |          0.81
 12000 |    3.39 |   +4.39±0.28 |   +1.11±0.01 |     0.0139 |          0.80
 15000 |    3.31 |   +4.55±0.31 |   +1.19±0.01 |     0.0133 |          0.79
 18000 |    3.24 |   +4.82±0.34 |   +1.25±0.02 |     0.0136 |          0.79
 21000 |    3.19 |   +5.10±0.34 |   +1.30±0.01 |     0.0137 |          0.80
 24000 |    3.15 |   +5.32±0.37 |   +1.33±0.03 |     0.0132 |          0.80
 27000 |    3.13 |   +5.46±0.38 |   +1.38±0.01 |     0.0130 |          0.80
 30000 |    3.12 |   +5.48±0.38 |   +1.39±0.02 |     0.0130 |          0.80
```

### `e8i2048`, E=8 @ I=2048, I_shared=1024, 2 seeds (`collapse_s1.log`)

```
     0 |   10.42 |   +0.51±0.01 |   -0.00±0.00 |     0.0009 |          1.00
  3000 |    4.15 |   +2.87±0.05 |   +1.06±0.05 |     0.0146 |          0.73
  6000 |    3.62 |   +2.01±0.04 |   +2.00±0.00 |     0.0181 |          0.50
  9000 |    3.38 |   +1.66±0.11 |   +2.37±0.02 |     0.0202 |          0.41
 12000 |    3.25 |   +1.55±0.08 |   +2.62±0.07 |     0.0225 |          0.37
 15000 |    3.16 |   +1.55±0.07 |   +2.64±0.06 |     0.0226 |          0.37
 18000 |    3.08 |   +1.62±0.06 |   +2.73±0.13 |     0.0233 |          0.37
 21000 |    3.02 |   +1.76±0.06 |   +2.79±0.10 |     0.0238 |          0.39
 24000 |    2.98 |   +1.85±0.08 |   +2.79±0.10 |     0.0243 |          0.40
 27000 |    2.95 |   +1.93±0.07 |   +2.86±0.14 |     0.0240 |          0.40
 30000 |    2.95 |   +1.95±0.07 |   +2.86±0.14 |     0.0239 |          0.40
```

Seed 0 alone gave min share 0.81 and 0.35 for these two configurations; adding seed 1 moved
them by 0.02. Seed noise on this quantity is therefore ~0.02, well below the effects reported
above.

### `sh512`, E=8 @ I=1024, I_shared=512, 2 seeds (`ratio.log`)

```
     0 |   10.49 |   +0.43±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.15 |   +1.96±0.12 |   +1.66±0.06 |     0.0157 |          0.54
  6000 |    3.61 |   +1.41±0.11 |   +2.64±0.11 |     0.0201 |          0.35
  9000 |    3.41 |   +1.27±0.11 |   +2.94±0.07 |     0.0231 |          0.30
 12000 |    3.29 |   +1.20±0.14 |   +3.14±0.12 |     0.0231 |          0.28
 15000 |    3.20 |   +1.21±0.15 |   +3.12±0.08 |     0.0220 |          0.28
 18000 |    3.13 |   +1.32±0.16 |   +3.11±0.03 |     0.0231 |          0.30
 21000 |    3.07 |   +1.41±0.18 |   +3.20±0.09 |     0.0231 |          0.31
 24000 |    3.03 |   +1.52±0.19 |   +3.18±0.06 |     0.0229 |          0.32
 27000 |    3.01 |   +1.59±0.19 |   +3.20±0.03 |     0.0227 |          0.33
 30000 |    3.01 |   +1.61±0.20 |   +3.19±0.03 |     0.0226 |          0.34
```

### `sh4096`, E=8 @ I=1024, I_shared=4096, 2 seeds (`ratio.log`)

```
     0 |   10.42 |   +0.51±0.02 |   -0.00±0.00 |     0.0009 |          1.00
  3000 |    4.16 |   +5.31±0.02 |   +0.25±0.01 |     0.0080 |          0.95
  6000 |    3.68 |   +4.89±0.06 |   +0.56±0.01 |     0.0099 |          0.90
  9000 |    3.41 |   +4.63±0.23 |   +0.76±0.02 |     0.0098 |          0.86
 12000 |    3.27 |   +4.44±0.25 |   +0.88±0.03 |     0.0099 |          0.83
 15000 |    3.18 |   +4.51±0.17 |   +0.97±0.02 |     0.0104 |          0.82
 18000 |    3.10 |   +4.70±0.21 |   +1.04±0.04 |     0.0101 |          0.82
 21000 |    3.04 |   +4.93±0.16 |   +1.12±0.02 |     0.0104 |          0.81
 24000 |    2.99 |   +5.12±0.15 |   +1.14±0.04 |     0.0102 |          0.82
 27000 |    2.97 |   +5.26±0.12 |   +1.18±0.03 |     0.0103 |          0.82
 30000 |    2.97 |   +5.29±0.12 |   +1.18±0.03 |     0.0103 |          0.82
```

Seed 1 of `sh512` and `sh4096` finished after the first version of this record was written;
the tables above are the pooled two-seed analyses. Adding seed 1 moved the minimum share by
0.03 (`sh512`, 0.31 → 0.28) and 0.01 (`sh4096`, 0.82 → 0.81) — consistent with the ~0.02 seed
noise measured on `e8i256` / `e8i2048`, and it widens the shared-width contrast slightly rather
than narrowing it.

## Routing granularity (`topk.log`, 2026-08-09)

8 layers, E=8, `I = I_shared = 1024` (width ratio 1.0) throughout; only `--top-k` varies.
`k=2` is the published `base1b`. 2 seeds each.

| `k` | min share | @983M | final PPL | routed NMI @983M | params |
|---|---|---|---|---|---|
| 1 | **0.97** | 0.97 | 3.13 | 0.0003 | 133.7M |
| 2 (`base1b`) | 0.47 | 0.50 | 3.00 | ~0.017 | 133.7M |
| 4 | **0.17** | 0.19 | 2.99 | 0.0235 | 133.7M |

Parameter count is identical across the three; only the number of experts activated per token
changes. This is the largest single effect measured anywhere in the study.

### `k1r1` — E=8, k=1, I=I_shared=1024, 2 seeds

```
     0 |   10.41 |   +0.49±0.02 |   -0.00±0.00 |     0.0017 |          1.00
  3000 |    4.23 |   +6.66±0.04 |   +0.12±0.00 |     0.0053 |          0.98
  6000 |    3.72 |   +7.79±0.08 |   +0.15±0.01 |     0.0070 |          0.98
  9000 |    3.53 |   +8.04±0.06 |   +0.16±0.00 |     0.0040 |          0.98
 12000 |    3.41 |   +8.44±0.01 |   +0.17±0.00 |     0.0047 |          0.98
 15000 |    3.33 |   +8.51±0.27 |   +0.20±0.01 |     0.0035 |          0.98
 18000 |    3.26 |   +8.70±0.23 |   +0.22±0.00 |     0.0028 |          0.98
 21000 |    3.20 |   +8.90±0.21 |   +0.25±0.00 |     0.0017 |          0.97
 24000 |    3.16 |   +8.84±0.21 |   +0.27±0.00 |     0.0007 |          0.97
 27000 |    3.14 |   +8.88±0.18 |   +0.29±0.00 |     0.0004 |          0.97
 30000 |    3.13 |   +8.89±0.17 |   +0.29±0.00 |     0.0003 |          0.97
```

Note the routed NMI *falling* over training (0.0070 → 0.0003): with top-1 the routed branch does
not merely stay unimportant, it loses what little domain structure it had.

### `k4r1` — E=8, k=4, I=I_shared=1024, 2 seeds

```
     0 |   10.41 |   +0.49±0.02 |   -0.00±0.00 |     0.0003 |          1.00
  3000 |    4.17 |   +2.47±0.04 |   +1.31±0.09 |     0.0134 |          0.65
  6000 |    3.64 |   +1.35±0.05 |   +2.66±0.02 |     0.0176 |          0.34
  9000 |    3.41 |   +0.99±0.02 |   +3.30±0.08 |     0.0199 |          0.23
 12000 |    3.28 |   +0.88±0.01 |   +3.79±0.01 |     0.0211 |          0.19
 15000 |    3.19 |   +0.82±0.00 |   +3.87±0.08 |     0.0220 |          0.18
 18000 |    3.12 |   +0.82±0.00 |   +3.89±0.10 |     0.0232 |          0.17
 21000 |    3.06 |   +0.87±0.01 |   +3.95±0.09 |     0.0238 |          0.18
 24000 |    3.02 |   +0.91±0.02 |   +3.94±0.06 |     0.0237 |          0.19
 27000 |    3.00 |   +0.94±0.01 |   +3.94±0.08 |     0.0234 |          0.19
 30000 |    2.99 |   +0.95±0.01 |   +3.95±0.08 |     0.0235 |          0.19
```

### Why the "active width fraction" does not rescue a single scalar

`x = I_shared/(I_shared + k·I)` folds `k` in and orders the `k≥2` data:

| `x` | 0.20 | 0.333 | 0.50 | 0.667 |
|---|---|---|---|---|
| observed min share | 0.17 / 0.25 / 0.28 / 0.37 | 0.47 / 0.53 / 0.55 / 0.56 | 0.63 | 0.72 / 0.79 / 0.81 |

but `e16cap` (E=16, k=2, I=512) and `k1r1` (E=8, k=1, I=1024) both sit at `x = 0.50` and split
0.63 against 0.97. Top-1 is outside the family — and the matched-width runs below show `x` fails
inside the family too.

### Matched active width (`gran.log`, 2026-08-09): granularity is a separate axis

Both at `k·I = 2048`, `I_shared = 1024`, hence `x = 0.333` — identical to `base1b` (0.47). 1 seed.

| run | k | I | min share | @983M | final PPL | routed NMI @983M |
|---|---|---|---|---|---|---|
| `k4i512` | 4 | 512 | **0.29** | 0.32 | 3.06 | **0.0301** (study max) |
| `base1b` (ref) | 2 | 1024 | 0.47 | 0.50 | 3.00 | ~0.017 |
| `k1i2048` | 1 | 2048 | **0.95** | 0.95 | 3.11 | 0.0003 |

`k4i512` refutes the active-width scalar within `k≥2` (0.29 vs 0.47 at identical `x`);
`k1i2048` shows top-1's collapse is not a width effect (0.95 with double the per-token width).

#### `k4i512_s0` — E=8, k=4, I=512, I_shared=1024 (`gran.log`)

```
     0 |   10.42 |   +0.49±0.00 |   -0.00±0.00 |     0.0003 |          1.00
  3000 |    4.20 |   +2.85±0.00 |   +1.09±0.00 |     0.0157 |          0.72
  6000 |    3.67 |   +1.80±0.00 |   +2.25±0.00 |     0.0245 |          0.44
  9000 |    3.45 |   +1.42±0.00 |   +2.65±0.00 |     0.0265 |          0.35
 12000 |    3.33 |   +1.29±0.00 |   +3.06±0.00 |     0.0284 |          0.30
 15000 |    3.25 |   +1.27±0.00 |   +3.10±0.00 |     0.0296 |          0.29
 18000 |    3.18 |   +1.32±0.00 |   +3.22±0.00 |     0.0305 |          0.29
 21000 |    3.12 |   +1.36±0.00 |   +3.23±0.00 |     0.0304 |          0.30
 24000 |    3.09 |   +1.44±0.00 |   +3.28±0.00 |     0.0302 |          0.30
 27000 |    3.07 |   +1.51±0.00 |   +3.26±0.00 |     0.0302 |          0.32
 30000 |    3.06 |   +1.51±0.00 |   +3.26±0.00 |     0.0301 |          0.32
```

#### `k1i2048_s0` — E=8, k=1, I=2048, I_shared=1024 (`gran.log`)

```
     0 |   10.42 |   +0.52±0.00 |   -0.00±0.00 |     0.0017 |          1.00
  3000 |    4.23 |   +5.96±0.00 |   +0.18±0.00 |     0.0043 |          0.97
  6000 |    3.79 |   +7.13±0.00 |   +0.22±0.00 |     0.0050 |          0.97
  9000 |    3.53 |   +7.81±0.00 |   +0.25±0.00 |     0.0038 |          0.97
 12000 |    3.41 |   +8.37±0.00 |   +0.26±0.00 |     0.0041 |          0.97
 15000 |    3.32 |   +8.07±0.00 |   +0.29±0.00 |     0.0055 |          0.97
 18000 |    3.24 |   +7.95±0.00 |   +0.33±0.00 |     0.0031 |          0.96
 21000 |    3.18 |   +8.03±0.00 |   +0.37±0.00 |     0.0019 |          0.96
 24000 |    3.14 |   +8.09±0.00 |   +0.40±0.00 |     0.0007 |          0.95
 27000 |    3.11 |   +8.09±0.00 |   +0.42±0.00 |     0.0003 |          0.95
 30000 |    3.11 |   +8.08±0.00 |   +0.42±0.00 |     0.0003 |          0.95
```

## Reproduction

```bash
# D1 (e4wide) — routed capacity matched to the E=8 baseline at equal total parameters
python -m analysis.pilot_train --num-layers 8 --num-experts 4 --top-k 2 \
  --moe-intermediate 2048 --shared-intermediate 1024 --steps 30000 --batch 64 --seq-len 512 \
  --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s --out e4wide_s$s

# D2 (dyn16) — shared-LR ramped back to parity by step 9000
python -m analysis.pilot_train --num-layers 16 --num-experts 8 --top-k 2 --steps 30000 \
  --batch 64 --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 \
  --shared-lr-mult 0.1 --shared-ramp-end 9000 --seed 0 --out dyn16_s0

# capacity-matched #experts sweep: E*I held at 8192
python -m analysis.pilot_train --num-layers 8 --num-experts $E --top-k 2 \
  --moe-intermediate $((8192/E)) --shared-intermediate 1024 --steps 30000 --batch 64 \
  --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s \
  --out e${E}cap_s$s

# routed-width contrast at E=8 (I in 256, 1024, 2048)
python -m analysis.pilot_train --num-layers 8 --num-experts 8 --top-k 2 \
  --moe-intermediate $I --shared-intermediate 1024 --steps 30000 --batch 64 --seq-len 512 \
  --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s --out e8i${I}_s$s

# shared-width contrast at E=8 (I_shared in 512, 1024, 4096), routed branch identical
python -m analysis.pilot_train --num-layers 8 --num-experts 8 --top-k 2 \
  --moe-intermediate 1024 --shared-intermediate $IS --steps 30000 --batch 64 --seq-len 512 \
  --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s --out sh${IS}_s$s

# dose-response over the handicap duration (16 layers, mult 0.1, only ramp_end varies)
for r in 3000 6000 15000; do
  python -m analysis.pilot_train --num-layers 16 --num-experts 8 --top-k 2 --steps 30000 \
    --batch 64 --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 \
    --shared-lr-mult 0.1 --shared-ramp-end $r --seed 0 --out dyn16_r${r}_s0
done
# second seed on the published ramp_end=9000 point
python -m analysis.pilot_train --num-layers 16 --num-experts 8 --top-k 2 --steps 30000 \
  --batch 64 --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 \
  --shared-lr-mult 0.1 --shared-ramp-end 9000 --seed 1 --out dyn16_s1
# the infinite-ramp asymptote: static x0.1 out to 983M
python -m analysis.pilot_train --num-layers 16 --num-experts 8 --top-k 2 --steps 30000 \
  --batch 64 --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 \
  --shared-lr-mult 0.1 --seed 0 --out m01long_s0

# analysis (pool seeds with a comma-separated list)
python -m analysis.race --runs <run_dir>[,<run_dir>...] --tokens-per-domain 60000 --skip 3000
```
